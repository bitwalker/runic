//! Attribute and type verification under mutation.
//!
//! A valid operation is given one faulty attribute or type at a time, wrapped in a random number
//! of holders (lists, and parameters of signatures) among random valid siblings, in a dictionary
//! entry or in the `signature` property. Each must be rejected with a message naming the
//! operation and the site; the same tree without the fault must verify.

use alloc::{format, rc::Rc, string::String, vec, vec::Vec};

use super::element_model_tests::{Desc, build, random_desc, type_pool};
use crate::{
    AttributeRef, CallConv, Context, Int, Op, Operation, StructType, Type,
    attributes::{AttrList, NamedAttribute},
    dialects::{
        builtin::{
            attributes::{
                AbiParam, BoolAttr, DeclaredEffects, DeclaredEffectsAttr, IntAttr, List, ListAttr,
                LocalVariable, LocalVariableAttr, MemoryEffectDescriptor, Signature, SignatureAttr,
                TypeArrayAttr, TypeAttr,
            },
            types::{EnumType, EnumVariant},
        },
        debuginfo::attributes::{Subprogram, SubprogramAttr, Variable, VariableAttr},
    },
    effects::MemoryEffect,
    interner::Symbol,
    testing::{Test, label_messages, text::Rng},
    types::TypeImpl,
};

fn sym(s: &str) -> Symbol {
    Symbol::intern(s)
}

fn has_symref(desc: &Desc) -> bool {
    match desc {
        Desc::SymRef => true,
        Desc::List(entries) => entries.iter().any(has_symref),
        Desc::Sig(params, results) => {
            params.iter().chain(results).any(|(_, attrs)| attrs.iter().any(has_symref))
        }
        _ => false,
    }
}

/// A random valid attribute: no unresolved symbol reference.
fn valid(cx: &Rc<Context>, rng: &mut Rng, pool: &[Type]) -> AttributeRef {
    loop {
        let desc = random_desc(rng, pool, 2);
        if !has_symref(&desc) {
            return build(cx, &desc).attr;
        }
    }
}

fn list(cx: &Rc<Context>, entries: impl IntoIterator<Item = AttributeRef>) -> AttributeRef {
    let mut list = AttrList::new();
    for entry in entries {
        list.push_back(entry.borrow().as_attr().as_attr_ref());
    }
    cx.create_attribute::<ListAttr, _>(List::from(list)).as_attribute_ref()
}

fn broken_enum(cx: &Context) -> Type {
    let wide = EnumType::new(
        None,
        cx.isize(),
        [EnumVariant::new(sym("a")).with_value(1 << 40), EnumVariant::new(sym("b"))],
    )
    .unwrap();
    cx.enum_type(wide.replace_subtypes(&mut |_| cx.i32()))
}

/// What a fault is, and what the message must say about it.
struct Fault {
    name: &'static str,
    attr: AttributeRef,
    /// Whether the fault is a type the attribute holds (reported by the type walk, at the
    /// top-level site, as "holds type") rather than the attribute itself.
    held_type: bool,
    /// A fragment the message must contain.
    says: String,
}

fn faults(cx: &Rc<Context>, other: &Rc<Context>) -> Vec<Fault> {
    let open = cx.ptr_type(cx.rec_var(0));
    let foreign = other.i64();
    let broken = broken_enum(cx);
    let broken_deep = cx.ptr_type(cx.struct_type(StructType::new([cx.i8().into(), broken.into()])));
    let mut out = Vec::new();

    // widths
    out.push(Fault {
        name: "an integer constant built with the wrong width",
        attr: cx
            .create_attribute_with_type::<IntAttr, _>(Int::from(-1i32), cx.i8())
            .as_attribute_ref(),
        held_type: false,
        says: "holding a 32-bit value, but a constant of that type is 8 bits wide".into(),
    });
    let mut edited = cx.int_attr(Int::from(1i32), cx.i32()).as_attribute_ref();
    *edited.borrow_mut().value_mut().as_any_mut().downcast_mut::<Int>().unwrap() = Int::from(1i8);
    out.push(Fault {
        name: "an integer constant edited in place",
        attr: edited,
        held_type: false,
        says: "holding a 8-bit value, but a constant of that type is 32 bits wide".into(),
    });
    out.push(Fault {
        name: "an integer constant of a type that has no constants",
        attr: cx
            .create_attribute_with_type::<IntAttr, _>(Int::from(1i32), cx.f32())
            .as_attribute_ref(),
        held_type: false,
        says: "no integer constant has that type".into(),
    });

    // signatures
    let never_param =
        Signature::from_parts(CallConv::default(), vec![AbiParam::new(cx.never_type())], vec![]);
    out.push(Fault {
        name: "a never parameter",
        attr: cx.create_attribute::<SignatureAttr, _>(never_param).as_attribute_ref(),
        held_type: false,
        says: "is an invalid signature: parameter 0 has type `never`".into(),
    });
    let never_result = Signature::from_parts(
        CallConv::default(),
        vec![],
        vec![AbiParam::new(cx.i32()), AbiParam::new(cx.never_type())],
    );
    out.push(Fault {
        name: "never among results",
        attr: cx.create_attribute::<SignatureAttr, _>(never_result).as_attribute_ref(),
        held_type: false,
        says: "is an invalid signature".into(),
    });
    for (what, ty, problem) in [
        ("open", open, "which is open"),
        ("foreign", foreign, "which belongs to another context"),
    ] {
        let signature =
            Signature::from_parts(CallConv::default(), vec![AbiParam::new(cx.i32())], vec![]);
        let mut attr = cx.create_attribute::<SignatureAttr, _>(signature);
        attr.borrow_mut().as_value_mut().params_mut()[0].ty = ty;
        out.push(Fault {
            name: if what == "open" {
                "an open parameter type"
            } else {
                "a parameter type of another context"
            },
            attr: attr.as_attribute_ref(),
            held_type: false,
            says: format!("is an invalid signature: parameter 0 has type `{ty}`, {problem}"),
        });
    }

    // types held in values
    for (name, ty, problem) in [
        ("a type attribute holding an open type", open, "which is open"),
        (
            "a type attribute holding a type of another context",
            foreign,
            "which belongs to another context",
        ),
        ("a type attribute holding an invalid enum", broken, "which is invalid"),
        (
            "a type attribute holding an invalid enum in a struct in a pointer",
            broken_deep,
            "which is invalid",
        ),
    ] {
        let mut attr = cx.create_attribute::<TypeAttr, _>(cx.i32());
        *attr.borrow_mut().as_value_mut() = ty;
        out.push(Fault {
            name,
            attr: attr.as_attribute_ref(),
            held_type: true,
            says: format!(
                "holds type `{}`, {problem}",
                if ty == broken_deep { broken } else { ty }
            ),
        });
    }
    for (name, ty, problem) in [
        ("a type array holding an open type", open, "which is open"),
        (
            "a type array holding a type of another context",
            foreign,
            "which belongs to another context",
        ),
        ("a type array holding an invalid enum", broken, "which is invalid"),
    ] {
        let mut attr = cx.create_attribute::<TypeArrayAttr, _>([cx.i32()]);
        attr.borrow_mut().as_value_mut().push(ty);
        out.push(Fault {
            name,
            attr: attr.as_attribute_ref(),
            held_type: true,
            says: format!("holds type `{ty}`, {problem}"),
        });
    }
    for (name, ty, problem) in [
        ("a debug variable of an open type", open, "which is open"),
        (
            "a debug variable of a type of another context",
            foreign,
            "which belongs to another context",
        ),
    ] {
        let mut attr = cx.create_attribute::<VariableAttr, _>(Variable::default());
        attr.borrow_mut().as_value_mut().ty = Some(ty);
        out.push(Fault {
            name,
            attr: attr.as_attribute_ref(),
            held_type: true,
            says: format!("holds type `{ty}`, {problem}"),
        });
    }
    {
        let mut attr = cx.create_attribute::<SubprogramAttr, _>(Subprogram::new(
            sym("f"),
            sym("f.rs"),
            1,
            None,
        ));
        attr.borrow_mut().as_value_mut().ty = Some(open);
        out.push(Fault {
            name: "a subprogram of an open type",
            attr: attr.as_attribute_ref(),
            held_type: true,
            says: format!("holds type `{open}`, which is open"),
        });
    }

    let local = cx
        .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(0, open))
        .as_attribute_ref();
    out.push(Fault {
        name: "a local of an open type",
        attr: local,
        held_type: true,
        says: format!("holds type `{open}`, which is open"),
    });

    // own types
    let mut flag = cx.create_attribute::<BoolAttr, _>(true).as_attribute_ref();
    flag.borrow_mut().as_attr_mut().set_type_unchecked(other.i1());
    out.push(Fault {
        name: "a fixed own type of another context",
        attr: flag,
        held_type: false,
        says: "has type `i1`, which belongs to another context".into(),
    });

    // effects
    let both = DeclaredEffects::from([MemoryEffectDescriptor {
        effect: MemoryEffect::Read,
        argument: Some(0),
        result: Some(0),
    }]);
    out.push(Fault {
        name: "an effect on an argument and a result",
        attr: cx.create_attribute::<DeclaredEffectsAttr, _>(both).as_attribute_ref(),
        held_type: false,
        says: "names both an argument and a result".into(),
    });
    out
}

/// Wraps `inner` in `depth` holders, each a list or a signature's parameter attribute, with
/// random valid siblings.
fn wrap(
    cx: &Rc<Context>,
    rng: &mut Rng,
    pool: &[Type],
    mut inner: AttributeRef,
    depth: usize,
) -> AttributeRef {
    for _ in 0..depth {
        inner = if rng.below(2) == 0 {
            let mut entries: Vec<AttributeRef> =
                (0..rng.below(3)).map(|_| valid(cx, rng, pool)).collect();
            let at = rng.below(entries.len() as u64 + 1) as usize;
            entries.insert(at, inner);
            list(cx, entries)
        } else {
            let sibling = valid(cx, rng, pool);
            let param = AbiParam::new_with_attrs(
                pool[rng.below(4) as usize],
                [NamedAttribute::new("note", inner), NamedAttribute::new("other", sibling)],
            );
            let results = vec![AbiParam::new(pool[0])];
            let signature = Signature::from_parts(CallConv::default(), vec![param], results);
            cx.create_attribute::<SignatureAttr, _>(signature).as_attribute_ref()
        };
    }
    inner
}

fn verify_error(op: &Operation) -> Result<(), String> {
    op.verify().map_err(|err| label_messages(err.as_diagnostic()).join(" | "))
}

/// Every fault at every depth, in a dictionary entry and in the `signature` property, is
/// rejected with a message naming the operation and the site; the operation verifies before the
/// fault is installed and after it is removed.
#[test]
fn every_fault_at_every_depth_is_rejected_by_name() {
    let mut checked = 0;
    for seed in crate::testing::test_seeds(12) {
        let mut rng = Rng(seed ^ 0xfa17);
        for depth in 0..5 {
            let test = Test::new("f", |cx| ([cx.i32()], []));
            let mut function = test.function();
            let cx = test.context_rc();
            let other = Rc::new(Context::default());
            let pool = type_pool(&cx);
            // valid random attributes around the fault
            for i in 0..2 {
                let attr = valid(&cx, &mut rng, &pool);
                function
                    .borrow_mut()
                    .as_operation_mut()
                    .set_attribute(sym(&format!("ok{i}")), attr);
            }
            verify_error(function.borrow().as_operation())
                .unwrap_or_else(|err| panic!("seed {seed:#x}: valid attributes rejected: {err}"));
            let original_signature =
                function.borrow().as_operation().get_attribute(sym("signature")).unwrap();

            for fault in faults(&cx, &other) {
                // (1) in the dictionary entry `e`
                let wrapped = wrap(&cx, &mut rng, &pool, fault.attr, depth);
                function.borrow_mut().as_operation_mut().set_attribute("e", wrapped);
                let message = match verify_error(function.borrow().as_operation()) {
                    Err(message) => message,
                    Ok(()) => panic!("seed {seed:#x}: {} at depth {depth} is accepted", fault.name),
                };
                let site = if fault.held_type || depth == 0 {
                    "attribute 'e' of 'builtin.function'"
                } else {
                    "an attribute held by attribute 'e' of 'builtin.function'"
                };
                assert!(
                    message.contains(site) && message.contains(&fault.says),
                    "seed {seed:#x}: {} at depth {depth}: {message}\n expected `{site}` .. `{}`",
                    fault.name,
                    fault.says
                );
                function.borrow_mut().as_operation_mut().remove_attribute("e");
                verify_error(function.borrow().as_operation()).unwrap();
                checked += 1;

                // (2) in the `signature` property, as an attribute of the parameter
                let wrapped = wrap(&cx, &mut rng, &pool, fault.attr.borrow().dyn_clone(), depth);
                let param =
                    AbiParam::new_with_attrs(cx.i32(), [NamedAttribute::new("note", wrapped)]);
                let signature = Signature::from_parts(CallConv::default(), vec![param], vec![]);
                let signature =
                    cx.create_attribute::<SignatureAttr, _>(signature).as_attribute_ref();
                function
                    .borrow_mut()
                    .as_operation_mut()
                    .set_property(sym("signature"), signature)
                    .unwrap();
                let message = match verify_error(function.borrow().as_operation()) {
                    Err(message) => message,
                    Ok(()) => panic!(
                        "seed {seed:#x}: {} at depth {depth} in the signature is accepted",
                        fault.name
                    ),
                };
                let site = if fault.held_type {
                    "property 'signature' of 'builtin.function'"
                } else {
                    "an attribute held by property 'signature' of 'builtin.function'"
                };
                assert!(
                    message.contains(site) && message.contains(&fault.says),
                    "seed {seed:#x}: {} at depth {depth} in the signature: {message}\n expected \
                     `{site}` .. `{}`",
                    fault.name,
                    fault.says
                );
                function
                    .borrow_mut()
                    .as_operation_mut()
                    .set_property(sym("signature"), original_signature)
                    .unwrap();
                verify_error(function.borrow().as_operation()).unwrap();
                checked += 1;
            }
        }
    }
    std::eprintln!("{checked} faults checked");
    assert!(checked > 2000, "{checked}");
}

//! What an attribute holds and the deep walk of an operation, against a model.
//!
//! The model is `Desc`: a description of an attribute that says, independently of
//! `visit_elements`, what it holds. `build` makes the attribute from a description and returns
//! the tree of handles (`Built`); the expectations are computed from that tree.

use alloc::{format, rc::Rc, string::String, vec, vec::Vec};

use crate::{
    AttributeRef, CallConv, CompactString, Context, FxHashSet, Int, Op, StructType, Type,
    WalkResult,
    attributes::{AttrList, ElementReplacer, NamedAttribute},
    dialects::{
        builtin::attributes::{
            AbiParam, Array, BoolAttr, I32ArrayAttr, I32Attr, IntAttr, List, ListAttr,
            LocalVariable, LocalVariableAttr, Signature, SignatureAttr, StringAttr, SymbolRef,
            SymbolRefAttr, TypeArrayAttr, TypeAttr, ZextAttr,
        },
        debuginfo::attributes::{Subprogram, SubprogramAttr, Variable, VariableAttr},
    },
    interner::Symbol,
    testing::{
        Test,
        elements::{Element, Identity, elements},
        text::Rng,
    },
};

/// A description of an attribute: what it is, and so what it holds.
#[derive(Clone, Debug)]
pub(crate) enum Desc {
    Bool(bool),
    I32(i32),
    Str(u8),
    SymRef,
    /// An integer constant: its own type is its one element. The value has the width of a
    /// constant of the first type; the attribute's type is the second (a raw replacement retypes
    /// a constant and keeps its value, as documented).
    Int(u64, Type, Type),
    TypeA(Type),
    TypeArray(Vec<Type>),
    I32Array(Vec<i32>),
    List(Vec<Desc>),
    /// For each parameter and then each result: its type, then its attributes.
    Sig(Vec<(Type, Vec<Desc>)>, Vec<(Type, Vec<Desc>)>),
    /// The uniqued `zext` marker; only ever a parameter attribute here (a marker cannot be linked
    /// into two lists).
    Zext,
    Variable(Option<Type>),
    Subprogram(Option<Type>),
    UnboundLocal(Type),
}

/// What a built attribute holds directly, in order.
#[derive(Clone)]
pub(crate) enum Held {
    Ty(Type),
    Attr(Built),
}

#[derive(Clone)]
pub(crate) struct Built {
    pub attr: AttributeRef,
    pub held: Vec<Held>,
}

fn sym(s: &str) -> Symbol {
    Symbol::intern(s)
}

/// The types attributes are built over: scalars, aggregates, and members of recursion groups.
pub(crate) fn type_pool(cx: &Context) -> Vec<Type> {
    let (i8, i32, i64, isize) = (cx.i8(), cx.i32(), cx.i64(), cx.isize());
    let pair = cx.struct_type(StructType::named(sym("pair"), [i64.into(), i8.into()]));
    let list = {
        let body = cx.struct_type(StructType::named(
            sym("list"),
            [i64.into(), cx.ptr_type(cx.rec_var(0)).into()],
        ));
        cx.rec_group(&[body]).unwrap()[0]
    };
    let (a, b) = {
        let a = cx.struct_type(StructType::named(
            sym("a"),
            [isize.into(), cx.ptr_type(cx.rec_var(1)).into()],
        ));
        let b = cx.struct_type(StructType::named(
            sym("b"),
            [cx.ptr_type(cx.rec_var(0)).into(), i64.into()],
        ));
        let members = cx.rec_group(&[a, b]).unwrap();
        (members[0], members[1])
    };
    vec![
        i8,
        i32,
        i64,
        isize,
        cx.ptr_type(i64),
        cx.array_type(i64, 3),
        pair,
        cx.ptr_type(pair),
        list,
        cx.ptr_type(list),
        a,
        b,
        cx.function_type([i64, a], [list]),
    ]
}

pub(crate) fn random_desc(rng: &mut Rng, pool: &[Type], depth: u32) -> Desc {
    let ty = |rng: &mut Rng| pool[rng.below(pool.len() as u64) as usize];
    let n = if depth == 0 { 10 } else { 14 };
    match rng.below(n) {
        0 => Desc::Bool(rng.below(2) == 0),
        1 => Desc::I32(rng.below(100) as i32),
        2 => Desc::Str(rng.below(5) as u8),
        3 => Desc::SymRef,
        4 => Desc::Int(rng.next_u64(), pool[2], pool[2]),
        5 => Desc::TypeA(ty(rng)),
        6 => Desc::TypeArray((0..rng.below(4)).map(|_| ty(rng)).collect()),
        7 => Desc::I32Array((0..rng.below(4)).map(|i| i as i32).collect()),
        8 => Desc::Variable((rng.below(3) != 0).then(|| ty(rng))),
        9 => {
            if rng.below(2) == 0 {
                Desc::Subprogram((rng.below(3) != 0).then(|| ty(rng)))
            } else {
                Desc::UnboundLocal(ty(rng))
            }
        }
        10 | 11 => {
            Desc::List((0..rng.below(4)).map(|_| random_desc(rng, pool, depth - 1)).collect())
        }
        _ => {
            let side = |rng: &mut Rng| -> Vec<(Type, Vec<Desc>)> {
                (0..rng.below(3))
                    .map(|_| {
                        let attrs = (0..rng.below(3))
                            .map(|_| {
                                if rng.below(3) == 0 {
                                    Desc::Zext
                                } else {
                                    random_desc(rng, pool, depth - 1)
                                }
                            })
                            .collect();
                        (ty(rng), attrs)
                    })
                    .collect()
            };
            let params = side(rng);
            let results = side(rng);
            Desc::Sig(params, results)
        }
    }
}

/// The names parameter attributes get, in the order `AbiParam` keeps them (sorted by name).
fn param_attr_names(count: usize) -> Vec<Symbol> {
    let mut names: Vec<Symbol> = (0..count).map(|i| sym(&format!("pa{i}"))).collect();
    names.sort();
    names
}

/// Builds the attribute `desc` describes, with the tree of what it holds.
pub(crate) fn build(cx: &Rc<Context>, desc: &Desc) -> Built {
    let leaf = |attr: AttributeRef| Built {
        attr,
        held: Vec::new(),
    };
    match desc {
        Desc::Bool(value) => leaf(cx.create_attribute::<BoolAttr, _>(*value).as_attribute_ref()),
        Desc::I32(value) => leaf(cx.create_attribute::<I32Attr, _>(*value).as_attribute_ref()),
        Desc::Str(n) => leaf(
            cx.create_attribute::<StringAttr, _>(CompactString::from(format!("s{n}")))
                .as_attribute_ref(),
        ),
        Desc::SymRef => {
            leaf(cx.create_attribute::<SymbolRefAttr, _>(SymbolRef::default()).as_attribute_ref())
        }
        Desc::Zext => leaf(cx.get_marker::<ZextAttr>().as_attribute_ref()),
        Desc::Int(bits, value_ty, ty) => {
            let width = Int::width_for_type(*value_ty).expect("an integer type");
            let value = Int::from_bits(64, u128::from(*bits)).trunc(width.min(64)).zext(width);
            Built {
                attr: cx.create_attribute_with_type::<IntAttr, _>(value, *ty).as_attribute_ref(),
                held: vec![Held::Ty(*ty)],
            }
        }
        Desc::TypeA(ty) => Built {
            attr: cx.create_attribute::<TypeAttr, _>(*ty).as_attribute_ref(),
            held: vec![Held::Ty(*ty)],
        },
        Desc::TypeArray(types) => Built {
            attr: cx
                .create_attribute::<TypeArrayAttr, _>(Array::from(types.clone()))
                .as_attribute_ref(),
            held: types.iter().copied().map(Held::Ty).collect(),
        },
        Desc::I32Array(values) => leaf(
            cx.create_attribute::<I32ArrayAttr, _>(Array::from(values.clone()))
                .as_attribute_ref(),
        ),
        Desc::Variable(ty) => {
            let mut variable = Variable::new(sym("x"), sym("f.rs"), 2, Some(3));
            variable.ty = *ty;
            Built {
                attr: cx.create_attribute::<VariableAttr, _>(variable).as_attribute_ref(),
                held: ty.iter().copied().map(Held::Ty).collect(),
            }
        }
        Desc::Subprogram(ty) => {
            let mut subprogram = Subprogram::new(sym("f"), sym("f.rs"), 1, None);
            subprogram.ty = *ty;
            Built {
                attr: cx.create_attribute::<SubprogramAttr, _>(subprogram).as_attribute_ref(),
                held: ty.iter().copied().map(Held::Ty).collect(),
            }
        }
        Desc::UnboundLocal(ty) => Built {
            attr: cx
                .create_attribute_with_type::<LocalVariableAttr, _>(LocalVariable::default(), *ty)
                .as_attribute_ref(),
            held: vec![Held::Ty(*ty)],
        },
        Desc::List(entries) => {
            let built: Vec<Built> = entries.iter().map(|entry| build(cx, entry)).collect();
            let mut list = AttrList::new();
            for entry in &built {
                list.push_back(entry.attr.borrow().as_attr().as_attr_ref());
            }
            Built {
                attr: cx.create_attribute::<ListAttr, _>(List::from(list)).as_attribute_ref(),
                held: built.into_iter().map(Held::Attr).collect(),
            }
        }
        Desc::Sig(params, results) => {
            let mut held = Vec::new();
            let mut side = |side: &[(Type, Vec<Desc>)]| -> Vec<AbiParam> {
                side.iter()
                    .map(|(ty, attrs)| {
                        held.push(Held::Ty(*ty));
                        let names = param_attr_names(attrs.len());
                        let built: Vec<Built> = attrs.iter().map(|a| build(cx, a)).collect();
                        let param = AbiParam::new_with_attrs(
                            *ty,
                            names.iter().zip(&built).map(|(n, b)| NamedAttribute::new(*n, b.attr)),
                        );
                        held.extend(built.into_iter().map(Held::Attr));
                        param
                    })
                    .collect()
            };
            let params = side(params);
            let results = side(results);
            let signature = Signature::from_parts(CallConv::default(), params, results);
            Built {
                attr: cx.create_attribute::<SignatureAttr, _>(signature).as_attribute_ref(),
                held,
            }
        }
    }
}

fn direct(built: &Built) -> Vec<Element> {
    built
        .held
        .iter()
        .map(|held| match held {
            Held::Ty(ty) => Element::Type(*ty),
            Held::Attr(built) => Element::Attribute(built.attr),
        })
        .collect()
}

fn same_handles(got: &[Element], expected: &[Element]) -> bool {
    got.len() == expected.len()
        && got.iter().zip(expected).all(|(got, expected)| match (got, expected) {
            (Element::Type(a), Element::Type(b)) => a == b,
            (Element::Attribute(a), Element::Attribute(b)) => AttributeRef::ptr_eq(a, b),
            _ => false,
        })
}

/// Checks `built` and everything under it: the visit lists exactly what the description says, by
/// handle; the identity is asked the same and answers `None`.
fn check_visits(built: &Built, seed: u64) {
    let got = elements(&built.attr);
    let expected = direct(built);
    assert!(
        same_handles(&got, &expected),
        "seed {seed:#x}: {:?} lists {got:?}, expected {expected:?}",
        built.attr.borrow()
    );
    let mut identity = Identity::default();
    let unchanged = built.attr.borrow().replace_elements(&mut identity);
    assert!(
        unchanged.is_none(),
        "seed {seed:#x}: the identity rebuilt {:?}",
        built.attr.borrow()
    );
    assert!(same_handles(&identity.0, &expected), "seed {seed:#x}: identity was asked");
    for held in &built.held {
        if let Held::Attr(inner) = held {
            check_visits(inner, seed);
        }
    }
}

/// `i64 -> i32` on the description.
fn map_desc(cx: &Context, desc: &Desc, map: &dyn Fn(&Context, Type) -> Type) -> Desc {
    let m = |ty: &Type| map(cx, *ty);
    let side = |side: &[(Type, Vec<Desc>)]| -> Vec<(Type, Vec<Desc>)> {
        side.iter()
            .map(|(ty, attrs)| (m(ty), attrs.iter().map(|a| map_desc(cx, a, map)).collect()))
            .collect()
    };
    match desc {
        Desc::Int(bits, value_ty, ty) => Desc::Int(*bits, *value_ty, m(ty)),
        Desc::TypeA(ty) => Desc::TypeA(m(ty)),
        Desc::TypeArray(types) => Desc::TypeArray(types.iter().map(m).collect()),
        Desc::List(entries) => Desc::List(entries.iter().map(|e| map_desc(cx, e, map)).collect()),
        Desc::Sig(params, results) => Desc::Sig(side(params), side(results)),
        Desc::Variable(ty) => Desc::Variable(ty.as_ref().map(m)),
        Desc::Subprogram(ty) => Desc::Subprogram(ty.as_ref().map(m)),
        Desc::UnboundLocal(ty) => Desc::UnboundLocal(m(ty)),
        other => other.clone(),
    }
}

/// A shallow type map (the handle itself only), applied to every type at any depth of an
/// attribute by recursing into the attributes it holds.
struct DeepReplacer<'a> {
    cx: &'a Context,
    map: &'a dyn Fn(&Context, Type) -> Type,
    asked_types: usize,
}

impl ElementReplacer for DeepReplacer<'_> {
    fn replace_type(&mut self, ty: Type) -> Type {
        self.asked_types += 1;
        (self.map)(self.cx, ty)
    }

    fn replace_attribute(&mut self, attr: &AttributeRef) -> Option<AttributeRef> {
        attr.borrow().replace_elements(self)
    }
}

fn swap_i64_i8(cx: &Context, ty: Type) -> Type {
    // only the handle itself: `ptr<i64>` is kept. i64 and i8 swap so that a kept sibling exists.
    if ty == cx.i64() {
        cx.int_type(64 + 13)
    } else {
        ty
    }
}

/// The printed form of an attribute, through a dictionary entry of a scratch operation.
fn printed(attr: &AttributeRef) -> String {
    format!("{:?}", attr.borrow())
}

/// Visits and identity replacement against the description; a deep replacement gives an
/// attribute equal to the one built from scratch from the mapped description, and leaves the
/// original listing what it listed.
#[test]
fn random_attributes_visit_and_replace_as_described() {
    for seed in crate::testing::test_seeds(40) {
        let test = Test::new("f", |_| ([] as [Type; 0], [] as [Type; 0]));
        let cx = test.context_rc();
        let pool = type_pool(&cx);
        let mut rng = Rng(seed);
        for _ in 0..40 {
            let desc = random_desc(&mut rng, &pool, 3);
            let built = build(&cx, &desc);
            check_visits(&built, seed);
            let before = printed(&built.attr);

            let mut replacer = DeepReplacer {
                cx: &cx,
                map: &swap_i64_i8,
                asked_types: 0,
            };
            let replaced = built.attr.borrow().replace_elements(&mut replacer);
            let mapped = map_desc(&cx, &desc, &swap_i64_i8);
            let expected = build(&cx, &mapped);
            let changes = printed(&expected.attr) != before;
            match replaced {
                None => assert!(
                    !changes,
                    "seed {seed:#x}: {desc:?} was not rebuilt, expected {:?}",
                    expected.attr.borrow()
                ),
                Some(new) => {
                    assert!(changes, "seed {seed:#x}: {desc:?} was rebuilt though nothing changed");
                    assert!(!AttributeRef::ptr_eq(&new, &built.attr));
                    assert_eq!(printed(&new), printed(&expected.attr), "seed {seed:#x}: {desc:?}");
                    assert!(new == expected.attr, "seed {seed:#x}: {desc:?}: not equal");
                    assert_eq!(new.name(), built.attr.name());
                }
            }
            // the original is as it was
            assert_eq!(printed(&built.attr), before, "seed {seed:#x}: the original changed");
            check_visits(&built, seed);
        }
    }
}

/// The attributes under `built`, a holder before what it holds.
fn preorder(built: &Built, out: &mut Vec<AttributeRef>, seen: &mut FxHashSet<usize>) {
    if !seen.insert(AttributeRef::as_ptr(&built.attr).cast::<()>().addr()) {
        return;
    }
    out.push(built.attr);
    for held in &built.held {
        if let Held::Attr(inner) = held {
            preorder(inner, out, seen);
        }
    }
}

fn direct_types(built: &Built, out: &mut Vec<(usize, Type)>, seen: &mut FxHashSet<usize>) {
    let id = AttributeRef::as_ptr(&built.attr).cast::<()>().addr();
    if !seen.insert(id) {
        return;
    }
    for held in &built.held {
        if let Held::Ty(ty) = held {
            out.push((id, *ty));
        }
    }
    for held in &built.held {
        if let Held::Attr(inner) = held {
            direct_types(inner, out, seen);
        }
    }
}

fn reachable(roots: impl IntoIterator<Item = Type>) -> FxHashSet<Type> {
    let mut seen = FxHashSet::default();
    let mut stack: Vec<Type> = roots.into_iter().collect();
    while let Some(ty) = stack.pop() {
        if seen.insert(ty) {
            ty.visit_subtypes(&mut |held| stack.push(held));
        }
    }
    seen
}

/// The deep walk of an operation that holds random attributes in a property (`signature` of
/// a function) and in dictionary entries reaches every attribute once, a holder first, and every
/// type once, recursive types included.
#[test]
fn the_walk_reaches_every_attribute_and_type_once() {
    for seed in crate::testing::test_seeds(40) {
        let mut test = Test::named("walk").in_module("m");
        let mut owner = test.define_function("owner", &[], &[]);
        let cx = test.context_rc();
        let pool = type_pool(&cx);
        let mut rng = Rng(seed);

        // a property: the signature (a random one; the function is not verified here)
        let sig = loop {
            let desc = random_desc(&mut rng, &pool, 3);
            if matches!(desc, Desc::Sig(..)) {
                break build(&cx, &desc);
            }
        };
        let mut entries: Vec<(Symbol, Built)> = (0..1 + rng.below(5))
            .map(|i| {
                let desc = random_desc(&mut rng, &pool, 3);
                (sym(&format!("e{i}")), build(&cx, &desc))
            })
            .collect();
        // one attribute held at two sites: the second entry is the first again
        if entries.len() > 1 && rng.below(2) == 0 {
            let first = entries[0].1.clone();
            entries[1].1 = first;
        }
        {
            let mut owner = owner.borrow_mut();
            let op = owner.as_operation_mut();
            op.set_property(sym("signature"), sig.attr).unwrap();
            for (name, built) in &entries {
                op.set_attribute(*name, built.attr);
            }
        }
        let owner = owner.borrow();
        let op = owner.as_operation();

        // The model: properties in declared order, then the dictionary in its own order.
        let mut expected = Vec::new();
        let mut seen = FxHashSet::default();
        let mut type_owner_seen = FxHashSet::default();
        let mut expected_types = Vec::new();
        for property in op.properties() {
            if property.name == sym("signature") {
                assert!(AttributeRef::ptr_eq(&property.value, &sig.attr));
                preorder(&sig, &mut expected, &mut seen);
                direct_types(&sig, &mut expected_types, &mut type_owner_seen);
            } else {
                // the other properties of a function hold nothing here
                let built = Built {
                    attr: property.value,
                    held: Vec::new(),
                };
                assert!(
                    elements(&property.value).is_empty(),
                    "property {} holds {:?}",
                    property.name,
                    elements(&property.value)
                );
                preorder(&built, &mut expected, &mut seen);
            }
        }
        for entry in op.attributes() {
            let entry = entry.as_named_attribute();
            let built = &entries.iter().find(|(name, _)| *name == entry.name).unwrap().1;
            preorder(built, &mut expected, &mut seen);
            direct_types(built, &mut expected_types, &mut type_owner_seen);
        }

        let mut got = Vec::new();
        let result = op.walk_attributes(&mut |_, attr| {
            got.push(*attr);
            WalkResult::Continue(())
        });
        assert!(matches!(result, WalkResult::Continue(())));
        assert_eq!(got.len(), expected.len(), "seed {seed:#x}: number of attributes");
        for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
            assert!(
                AttributeRef::ptr_eq(got, expected),
                "seed {seed:#x}: attribute {index}: {:?} != {:?}",
                got.borrow(),
                expected.borrow()
            );
        }

        // types: each once; exactly those reachable from the entry block's arguments (none), the
        // results (none) and the types the attributes hold directly
        let mut types = Vec::new();
        let result = op.walk_types(&mut |_, ty| {
            types.push(ty);
            WalkResult::Continue(())
        });
        assert!(matches!(result, WalkResult::Continue(())));
        let distinct: FxHashSet<Type> = types.iter().copied().collect();
        assert_eq!(distinct.len(), types.len(), "seed {seed:#x}: a type was walked twice");
        let expected_set = reachable(expected_types.iter().map(|(_, ty)| *ty));
        assert_eq!(distinct, expected_set, "seed {seed:#x}: the types walked");
        // the first occurrences come in the order the model lists the direct types
        let mut first = Vec::new();
        let mut seen_types = FxHashSet::default();
        for (_, ty) in &expected_types {
            if seen_types.insert(*ty) {
                first.push(*ty);
                // everything this type reaches is walked before the next direct type
                seen_types.extend(reachable([*ty]));
            }
        }
        let positions: Vec<usize> =
            first.iter().map(|ty| types.iter().position(|t| t == ty).unwrap()).collect();
        assert!(positions.is_sorted(), "seed {seed:#x}: order of types {positions:?}");
    }
}

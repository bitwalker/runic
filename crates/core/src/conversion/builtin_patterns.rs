//! Conversion patterns, provided by core, that convert the builtin operations by type: what a
//! [TypeConverter] answers for each type is what they do, and nothing else.
//!
//! [populate_builtin_conversion_patterns] adds them to a pattern set, all driven by one converter.
//! The converter must answer for builtin function types itself
//! ([TypeConverter::with_function_types]): a function type converts to the function type of its
//! converted parameters and results, which is the function type of the converted signature
//! ([convert_signature]). It must have no value callbacks ([TypeConverter::has_value_conversions]):
//! a signature is converted by type, and a callback that answered differently for a value would
//! make a call and the entry block of its callee disagree.
//!
//! The patterns work under a target that judges their operations by type,
//! [add_builtin_legality_by_type]. That is a requirement, not a convenience: the legalization
//! graph is built from operation names, and a pattern that builds operations of its own root's
//! kind (a load of a local that becomes several loads) is dropped when that kind is statically
//! illegal.
//!
//! # The function pattern
//!
//! A `builtin.function`, a definition or a declaration, is converted as one: its signature
//! ([convert_signature]), its declared effects (remapped with the signature's ranges), its
//! `locals`, its entry block (through a [SignatureConversion] built from the converted signature,
//! not from the block, so the two cannot disagree), the arguments of its other blocks
//! ([ConversionPatternRewriter::convert_region_types]), and every reference to a local in its
//! body. The terminators that branch to a converted block are not converted with it: the branch
//! pattern (below) converts them.
//!
//! Local `i` of type `T` becomes the locals `[s_i, s_i + n_i)`, of the types the converter gives
//! for `T` (`n_i` of them, possibly none), where `s_i` is the sum of `n_k` for `k < i`: contiguous
//! and in order. Every [LocalVariableAttr] held, at any depth of attribute, by an operation in the
//! body outside nested callables gets the index `s_i` at once, keeping its type; every local slot
//! of a debug expression ([ExpressionOp::LocalSlot], [FrameBase::LocalSlot]) does too, so a slot
//! that names a local which became several names its first part. These are changes to
//! attributes, which need no cast. An operation that names a local is afterwards unconverted
//! exactly when the local's type is not legal; converting it is the job of the local patterns (for
//! `builtin.load_local` and `builtin.store_local`) or of its own dialect's patterns. Until it is
//! converted it fails the function's verifier, as no cast can stand for a local.
//!
//! A local that becomes no local leaves nothing to name. A `di.debug_declare` whose expression is
//! exactly that local's slot is erased, and a `di.debug_value` whose expression is exactly that
//! slot loses it (its expression becomes empty), as `Local2Reg` treats the debug records of a
//! local it removes; `Local2Reg` also records each stored value, which does not carry over here,
//! since the stored values become no value. Any other debug expression that names such a local
//! names `s_i` afterwards, the local that follows it, when the function has a local there, and is
//! an error otherwise.
//!
//! Last, each type that another attribute of the function (a property other than the three, or
//! an entry of its dictionary) holds directly, at any depth of attribute, and that converts to one
//! other type is replaced by it (a debug-info subprogram holds the function's type), so that the
//! function is legal by [TypeConverter::is_legal_op] afterwards and the pattern runs once. A type
//! held there that converts to several types or none is left as it is: the function stays
//! illegal, and the conversion reports it.
//!
//! # The local patterns
//!
//! `builtin.load_local s : T`, with `T` not legal, becomes one `builtin.load_local (s + k) : T_k`
//! per type `T_k` the converter gives for `T`, and its result is replaced by their results (by no
//! value when there are none). `builtin.store_local s, %v` becomes one store per value `%v` became,
//! and is erased when there are none. Both apply only once the function the operation is in was
//! converted: its `locals[s .. s + n]` has the converted types and its local `s` is no longer of
//! type `T`. So a partial conversion that leaves the function alone does not write loads or stores
//! against unconverted locals. An operation in no function does not match.
//!
//! # Calls, function references and returns
//!
//! A `builtin.call` becomes a call of the same callee, by the path it was written with, with the
//! signature [convert_signature] gives for its own: the function the callee's pattern uses, so the
//! two signatures stay equal (through an alias too, which holds no signature and needs no
//! pattern). Its arguments are the values its operands became, flattened in order, and old result
//! `j` is replaced by the new results [ConvertedSignature::results] gives for it. A
//! `builtin.call_indirect` is converted the same way; its callee, a function value, becomes the
//! one value of the converted function type. A `builtin.function_ref` becomes a reference to the
//! same symbol of the converted function type. A `builtin.ret` becomes a return of the values its
//! operands became, flattened. Each keeps its attribute dictionary. A call whose signature does not
//! change and whose operands are themselves, a reference whose type does not change, and a return
//! whose operands are themselves do not match. `builtin.ret_imm` is not converted here.
//!
//! # Branches
//!
//! The branch pattern is rooted on `BranchOpInterface`, so it applies to every operation that
//! implements it and is registered when the pattern set is frozen. It changes the branch in
//! place: for each successor, the forwarded operands are replaced by the values they became,
//! flattened, which is what the block arguments of a converted successor take. The arguments a
//! branch produces itself, its other operands and its successors' keys are not touched. A branch
//! none of whose forwarded operands changes does not match.

use alloc::{format, rc::Rc, vec::Vec};
use core::any::TypeId;

use smallvec::SmallVec;

use super::{
    ConversionPattern, ConversionPatternRewriter, ConversionPatternSet, ConversionTarget,
    ConvertedOperands, ConvertedSignature, DynamicLegalityResult, SignatureConversion,
    TypeConverter, convert_signature,
};
use crate::{
    AttributeRef, CallableOpInterface, Context, Op, OpRegistration, Operation, OperationRef,
    Report, Spanned, SuccessorOperands, Symbol, Type, ValueRef, WalkResult,
    attributes::ElementReplacer,
    dialects::{
        builtin::{
            Call, CallIndirect, CopiedAttributes, Function, FunctionReference, LoadLocal, Ret,
            RetImm, StoreLocal,
            attributes::{
                DeclaredEffects, DeclaredEffectsAttr, LocalVariable, LocalVariableAttr,
                SignatureAttr, TypeArrayAttr, remap_declared_effects,
            },
        },
        debuginfo::{
            DebugDeclare, DebugValue,
            attributes::{Expression, ExpressionAttr, ExpressionOp, FrameBase},
        },
    },
    interner,
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind},
    traits::BranchOpInterface,
};

/// The properties of a function that the function pattern computes itself.
const SIGNATURE: &str = "signature";
const MEMORY_EFFECTS: &str = "memory_effects";
const LOCALS: &str = "locals";

/// Adds the patterns of this part to `patterns`, all driven by `converter`. Fails when the converter
/// does not convert function types.
///
/// It also fails for a converter with value callbacks. Either error is raised before anything is
/// added. The patterns added are the function pattern, the patterns of `builtin.load_local` and
/// `builtin.store_local`, of `builtin.call`, `builtin.call_indirect`, `builtin.function_ref` and
/// `builtin.ret`, and the branch pattern, for every operation that implements `BranchOpInterface`
/// and is registered when the pattern set is frozen (see the module documentation).
///
/// `builtin.ret_imm` gets no pattern: a constant of a type that is not legal belongs to the
/// dialect that owns constants. Under [add_builtin_legality_by_type] a `builtin.ret_imm` of a type
/// that is not legal stays illegal, and a conversion reports it as having no pattern unless
/// another pattern set converts it.
pub fn populate_builtin_conversion_patterns(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
) -> Result<(), Report> {
    if !converter.converts_function_types() {
        return Err(Report::msg(
            "the builtin conversion patterns need a type converter that converts function types \
             by their parts (TypeConverter::with_function_types)",
        ));
    }
    if converter.has_value_conversions() {
        return Err(Report::msg(
            "the builtin conversion patterns convert signatures by type, and cannot be driven by \
             a type converter with value callbacks",
        ));
    }
    let context = patterns.context();
    patterns.push(FunctionPattern {
        info: pattern_info::<Function>(&context, "builtin-function-by-type", []),
        converter: converter.clone(),
    });
    patterns.push(LoadLocalPattern {
        info: pattern_info::<LoadLocal>(
            &context,
            "builtin-load-local-by-type",
            [registered::<LoadLocal>(&context)],
        ),
        converter: converter.clone(),
    });
    patterns.push(StoreLocalPattern {
        info: pattern_info::<StoreLocal>(
            &context,
            "builtin-store-local-by-type",
            [registered::<StoreLocal>(&context)],
        ),
        converter: converter.clone(),
    });
    patterns.push(CallPattern {
        info: pattern_info::<Call>(
            &context,
            "builtin-call-by-type",
            [registered::<Call>(&context)],
        ),
        converter: converter.clone(),
    });
    patterns.push(CallIndirectPattern {
        info: pattern_info::<CallIndirect>(
            &context,
            "builtin-call-indirect-by-type",
            [registered::<CallIndirect>(&context)],
        ),
        converter: converter.clone(),
    });
    patterns.push(FunctionReferencePattern {
        info: pattern_info::<FunctionReference>(
            &context,
            "builtin-function-ref-by-type",
            [registered::<FunctionReference>(&context)],
        ),
        converter: converter.clone(),
    });
    patterns.push(RetPattern {
        info: pattern_info::<Ret>(&context, "builtin-ret-by-type", [registered::<Ret>(&context)]),
        converter: converter.clone(),
    });
    let mut info = PatternInfo::new(
        context.clone(),
        "branch-by-type",
        PatternKind::Trait(TypeId::of::<dyn BranchOpInterface>()),
        PatternBenefit::new(1),
    );
    info.with_generated_ops([]);
    patterns.push(BranchPattern {
        info,
        converter: converter.clone(),
    });
    Ok(())
}

/// Marks `builtin.function`, the call operations, `builtin.function_ref`, the returns, the local
/// operations, and every operation that implements `BranchOpInterface`, legal exactly when
/// `converter.is_legal_op` says so. (A rule for an operation or its dialect wins over the interface
/// rule: a target that calls a branch operation or its dialect legal outright never offers it.)
pub fn add_builtin_legality_by_type(target: &mut ConversionTarget, converter: &TypeConverter) {
    let converter = Rc::new(converter.clone());
    let by_type = move || {
        let converter = Rc::clone(&converter);
        move |op: &Operation| DynamicLegalityResult::legal_if(converter.is_legal_op(op))
    };
    target
        .add_dynamically_legal_op::<Function, _>(by_type())
        .add_dynamically_legal_op::<Call, _>(by_type())
        .add_dynamically_legal_op::<CallIndirect, _>(by_type())
        .add_dynamically_legal_op::<FunctionReference, _>(by_type())
        .add_dynamically_legal_op::<Ret, _>(by_type())
        .add_dynamically_legal_op::<RetImm, _>(by_type())
        .add_dynamically_legal_op::<LoadLocal, _>(by_type())
        .add_dynamically_legal_op::<StoreLocal, _>(by_type())
        .add_dynamically_legal_op_interface::<dyn BranchOpInterface, _>(by_type());
}

fn registered<T: OpRegistration>(context: &Rc<Context>) -> crate::OperationName {
    context
        .get_or_register_dialect::<<T as OpRegistration>::Dialect>()
        .expect_registered_name::<T>()
}

fn pattern_info<Root: OpRegistration>(
    context: &Rc<Context>,
    name: &'static str,
    generated: impl IntoIterator<Item = crate::OperationName>,
) -> PatternInfo {
    let mut info = PatternInfo::new(
        context.clone(),
        name,
        PatternKind::Operation(registered::<Root>(context)),
        PatternBenefit::new(1),
    );
    info.with_generated_ops(generated);
    info
}

/// Converts a `builtin.function`'s signature, entry block, declared effects and locals together.
struct FunctionPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for FunctionPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

/// Where the locals of a function go: local `i` becomes `[starts[i], starts[i] + counts[i])`.
struct LocalMap {
    starts: Vec<u32>,
    counts: Vec<u32>,
    /// The number of new locals.
    total: u32,
}

impl LocalMap {
    /// The new index of local `index`, `s_i`; `None` for an index the function has no local of.
    fn start(&self, index: u32) -> Option<u32> {
        self.starts.get(index as usize).copied()
    }

    /// Whether local `index` became no local.
    fn vanished(&self, index: u32) -> bool {
        self.counts.get(index as usize) == Some(&0)
    }

    fn is_identity(&self) -> bool {
        self.counts.iter().all(|count| *count == 1)
    }
}

/// A change to an operation in a function's body that renumbers its locals.
enum BodyEdit {
    /// Erase this operation (a debug declaration of a local that became no local).
    Erase(OperationRef),
    /// Install these properties and dictionary entries on the operation.
    Set {
        op: OperationRef,
        attributes: Vec<(interner::Symbol, bool, AttributeRef)>,
    },
}

impl ConversionPattern for FunctionPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        _operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let context = op.borrow().context_rc();
        let converter = &self.converter;

        // (1) Compute everything before changing anything.
        let (signature, effects, locals, body) = {
            let op = op.borrow();
            let function = op.downcast_ref::<Function>().expect("rooted at builtin.function");
            let body = (!function.is_declaration()).then(|| function.body().as_region_ref());
            (
                function.get_signature().clone(),
                function.get_memory_effects().clone(),
                function.get_locals().iter().copied().collect::<Vec<_>>(),
                body,
            )
        };
        let converted = convert_signature(&context, &signature, converter);
        check_effects_within_signature(&op.borrow(), &effects, &converted)?;
        let new_effects = remap_declared_effects(&effects, &converted.params, &converted.results);
        let mut map = LocalMap {
            starts: Vec::with_capacity(locals.len()),
            counts: Vec::with_capacity(locals.len()),
            total: 0,
        };
        let mut new_locals = Vec::with_capacity(locals.len());
        for ty in &locals {
            let parts = converter.convert_type(&context, *ty);
            map.starts.push(map.total);
            map.counts.push(local_index(parts.len()));
            map.total += local_index(parts.len());
            new_locals.extend(parts);
        }
        let body_edits = match body {
            Some(body) if !map.is_identity() => renumber_locals(&context, body, &map)?,
            _ => Vec::new(),
        };
        let other_attributes = replace_held_types(&context, &op.borrow(), converter);
        let properties_change =
            converted.signature != signature || new_effects != effects || new_locals != locals;
        if !properties_change
            && body_edits.is_empty()
            && other_attributes.is_empty()
            && converter.is_legal_op(&op.borrow())
        {
            return Ok(false);
        }

        let mut changed = false;
        // (2) The function's own properties and attributes, in one change in place.
        if properties_change || !other_attributes.is_empty() {
            let signature_attr = context
                .create_attribute::<SignatureAttr, _>(converted.signature.clone())
                .as_attribute_ref();
            let effects_attr = context
                .create_attribute::<DeclaredEffectsAttr, _>(new_effects)
                .as_attribute_ref();
            let locals_attr =
                context.create_attribute::<TypeArrayAttr, _>(new_locals).as_attribute_ref();
            rewriter.modify_op_in_place(op, |function| -> Result<(), Report> {
                if properties_change {
                    function.set_property(SIGNATURE, signature_attr)?;
                    function.set_property(MEMORY_EFFECTS, effects_attr)?;
                    function.set_property(LOCALS, locals_attr)?;
                }
                install(function, other_attributes)
            })??;
            changed = true;
        }

        // (3) The blocks of a definition: the entry block from the converted signature, then the
        // others by their arguments' types.
        if let Some(body) = body {
            let entry = body.borrow().entry_block_ref().expect("a definition has an entry block");
            let mut conversion = SignatureConversion::new(converted.params.len());
            for (index, range) in converted.params.iter().enumerate() {
                let types = converted.signature.params()[range.start as usize..range.end as usize]
                    .iter()
                    .map(|param| param.ty)
                    .collect::<SmallVec<[Type; 2]>>();
                if types.is_empty() {
                    conversion.remap_input(index, &[])?;
                } else {
                    conversion.add_inputs(index, &types)?;
                }
            }
            if !conversion.is_identity(entry) {
                rewriter.apply_signature_conversion(entry, &conversion)?;
                changed = true;
            }
            changed |= rewriter.convert_region_types(body)?;
        }

        // (4) The references to locals in the body.
        for edit in body_edits {
            match edit {
                BodyEdit::Erase(op) => rewriter.erase_op(op)?,
                BodyEdit::Set { op, attributes } => {
                    rewriter.modify_op_in_place(op, |op| install(op, attributes))??;
                }
            }
            changed = true;
        }

        if !changed {
            rewriter.notify_match_failure(
                op,
                Report::msg(
                    "an attribute of the function holds a type that converts to several types or \
                     none, which the function pattern cannot replace",
                ),
            );
        }
        Ok(changed)
    }
}

/// An error, naming the function and the effect, when a declared effect of `function` names a
/// parameter or result its signature does not have (a function made without verification can hold
/// one): such an effect cannot be remapped.
fn check_effects_within_signature(
    function: &Operation,
    effects: &DeclaredEffects,
    converted: &ConvertedSignature,
) -> Result<(), Report> {
    let Some(effects) = effects.known() else {
        return Ok(());
    };
    for effect in effects {
        let outside = match (effect.argument, effect.result) {
            (Some(index), _) => (index as usize >= converted.params.len()).then(|| {
                format!(
                    "argument {index}, but its signature has {} parameter(s)",
                    converted.params.len()
                )
            }),
            (None, Some(index)) => (index as usize >= converted.results.len()).then(|| {
                format!(
                    "result {index}, but its signature has {} result(s)",
                    converted.results.len()
                )
            }),
            (None, None) => None,
        };
        if let Some(outside) = outside {
            let name = function
                .as_trait::<dyn Symbol>()
                .map(|symbol| format!("@{}", symbol.name()))
                .unwrap_or_else(|| format!("{}", function.name()));
            return Err(Report::msg(format!(
                "the declared effect `{effect}` of function {name} names {outside}: it cannot be \
                 remapped to the converted signature"
            )));
        }
    }
    Ok(())
}

/// `count` as a number of locals.
fn local_index(count: usize) -> u32 {
    u32::try_from(count)
        .expect("system limit: unable to allocate more than u32::MAX locals per function")
}

/// Installs `attributes` on `op`: each `(name, is_property, value)` as a property or a dictionary
/// entry. A symbol reference that a rebuilt attribute copied gets a use of its own.
fn install(
    op: &mut Operation,
    attributes: Vec<(interner::Symbol, bool, AttributeRef)>,
) -> Result<(), Report> {
    for (name, is_property, value) in attributes {
        crate::ir::give_copied_references_their_own_uses(op, &value);
        if is_property {
            op.set_property(name, value)?;
        } else {
            op.set_attribute(name, value);
        }
    }
    Ok(())
}

/// Replaces what an attribute holds at any depth of attribute: `leaf` is asked about each
/// attribute first (an answer replaces it, and what it holds is not looked at), and `ty` about each
/// type held directly by an attribute it does not answer for.
struct DeepReplacer<'a> {
    leaf: &'a mut dyn FnMut(&AttributeRef) -> Option<AttributeRef>,
    ty: &'a mut dyn FnMut(Type) -> Type,
}

impl DeepReplacer<'_> {
    /// The attribute to hold instead of `attr`, or `None` to keep it.
    fn replace(&mut self, attr: &AttributeRef) -> Option<AttributeRef> {
        if let Some(new) = (self.leaf)(attr) {
            return Some(new);
        }
        attr.borrow().replace_elements(self)
    }

    /// The properties and dictionary entries of `op` that change: `(name, is_property, new)`.
    fn replacements(&mut self, op: &Operation) -> Vec<(interner::Symbol, bool, AttributeRef)> {
        let mut changes = Vec::new();
        for property in op.properties() {
            if let Some(new) = self.replace(&property.value) {
                changes.push((property.name, true, new));
            }
        }
        for entry in op.attributes() {
            let entry = entry.as_named_attribute();
            if let Some(new) = self.replace(&entry.value) {
                changes.push((entry.name, false, new));
            }
        }
        changes
    }
}

impl ElementReplacer for DeepReplacer<'_> {
    fn replace_type(&mut self, ty: Type) -> Type {
        (self.ty)(ty)
    }

    fn replace_attribute(&mut self, attr: &AttributeRef) -> Option<AttributeRef> {
        self.replace(attr)
    }
}

/// The changes to the function's attributes other than its signature, declared effects and
/// locals: each type held directly, at any depth of attribute, that converts to one other type is
/// replaced by it. A type that converts to several or none is kept.
fn replace_held_types(
    context: &Context,
    function: &Operation,
    converter: &TypeConverter,
) -> Vec<(interner::Symbol, bool, AttributeRef)> {
    let mut leaf = |_: &AttributeRef| None;
    let mut ty = |ty: Type| match converter.convert_type(context, ty).as_slice() {
        [one] => *one,
        _ => ty,
    };
    let mut replacer = DeepReplacer {
        leaf: &mut leaf,
        ty: &mut ty,
    };
    replacer
        .replacements(function)
        .into_iter()
        .filter(|(name, is_property, _)| {
            !*is_property || ![SIGNATURE, MEMORY_EFFECTS, LOCALS].contains(&name.as_str())
        })
        .collect()
}

/// The edits that give every reference to a local in `body`, outside nested callables, its new
/// index under `map`. An error, before anything is changed, for a debug expression that names a
/// local which became no local where nothing follows it.
fn renumber_locals(
    context: &Rc<Context>,
    body: crate::RegionRef,
    map: &LocalMap,
) -> Result<Vec<BodyEdit>, Report> {
    let mut ops = Vec::new();
    let _ = body.borrow().prewalk(|op: &Operation| {
        if op.implements::<dyn CallableOpInterface>() {
            return WalkResult::<Report>::Skip;
        }
        ops.push(op.as_operation_ref());
        WalkResult::Continue(())
    });

    let mut error = None;
    let mut edits = Vec::new();
    for op in ops {
        // A debug record that is exactly the slot of a local that became no local.
        let vanished_slot = {
            let op = op.borrow();
            let expression = op
                .downcast_ref::<DebugDeclare>()
                .map(|declare| declare.expression().as_value().clone())
                .or_else(|| {
                    op.downcast_ref::<DebugValue>()
                        .map(|value| value.expression().as_value().clone())
                });
            match expression.as_ref().map(|expression| expression.operations.as_slice()) {
                Some([ExpressionOp::LocalSlot(index)]) if map.vanished(*index) => {
                    Some(op.is::<DebugDeclare>())
                }
                _ => None,
            }
        };
        match vanished_slot {
            Some(true) => {
                edits.push(BodyEdit::Erase(op));
                continue;
            }
            Some(false) => {
                let empty = context
                    .create_attribute::<ExpressionAttr, _>(Expression::new())
                    .as_attribute_ref();
                edits.push(BodyEdit::Set {
                    op,
                    attributes: alloc::vec![(interner::Symbol::intern("expression"), true, empty)],
                });
                continue;
            }
            None => (),
        }

        let mut leaf = |attr: &AttributeRef| -> Option<AttributeRef> {
            let attr = attr.borrow();
            if let Some(local) = attr.downcast_ref::<LocalVariableAttr>() {
                let local = *local.as_value();
                let index = map.start(local.index()).filter(|index| *index != local.index())?;
                return Some(
                    context
                        .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(
                            index,
                            local.ty(),
                        ))
                        .as_attribute_ref(),
                );
            }
            let expression = attr.downcast_ref::<ExpressionAttr>()?;
            let mut slot = |index: &mut u32| {
                if map.vanished(*index) && map.start(*index) == Some(map.total) && error.is_none() {
                    error = Some(Report::msg(format!(
                        "a debug expression of '{}' names local {index}, which becomes no local, \
                         and no local follows it",
                        op.borrow().name()
                    )));
                }
                if let Some(start) = map.start(*index) {
                    *index = start;
                }
            };
            let mut new = expression.as_value().clone();
            for operation in &mut new.operations {
                match operation {
                    ExpressionOp::LocalSlot(index)
                    | ExpressionOp::FrameBase {
                        base: FrameBase::LocalSlot(index),
                        ..
                    } => slot(index),
                    _ => (),
                }
            }
            (new != *expression.as_value())
                .then(|| context.create_attribute::<ExpressionAttr, _>(new).as_attribute_ref())
        };
        let mut ty = |ty: Type| ty;
        let attributes = DeepReplacer {
            leaf: &mut leaf,
            ty: &mut ty,
        }
        .replacements(&op.borrow());
        if !attributes.is_empty() {
            edits.push(BodyEdit::Set { op, attributes });
        }
    }
    match error {
        Some(error) => Err(error),
        None => Ok(edits),
    }
}

/// The local `op` names and the types the converter gives its type, when that type is not legal
/// and the function `op` is in was converted: its locals `[s, s + n)` have those types and its
/// local `s` is no longer of the old type. `None` otherwise, with a reason recorded.
fn converted_local(
    op: OperationRef,
    local: LocalVariable,
    converter: &TypeConverter,
    rewriter: &mut ConversionPatternRewriter,
) -> Option<SmallVec<[Type; 2]>> {
    let context = op.borrow().context_rc();
    if converter.is_legal_type(&context, local.ty()) {
        return None;
    }
    let parts = converter.convert_type(&context, local.ty());
    let Some(function) = op.borrow().nearest_parent_op::<Function>() else {
        rewriter.notify_match_failure(op, Report::msg("the local is in no function"));
        return None;
    };
    let function = function.borrow();
    let locals = function.get_locals();
    let start = local.index() as usize;
    let converted = locals.iter().skip(start).take(parts.len()).eq(parts.iter())
        && locals.len() >= start + parts.len()
        && locals.iter().nth(start) != Some(&local.ty());
    drop(locals);
    drop(function);
    if !converted {
        rewriter.notify_match_failure(
            op,
            Report::msg(format!(
                "local {} of type '{}' is not converted in its function yet",
                local.index(),
                local.ty()
            )),
        );
        return None;
    }
    Some(parts)
}

/// Splits a `builtin.load_local` of a local whose type is not legal into one load per part.
struct LoadLocalPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for LoadLocalPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for LoadLocalPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        _operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let local = *op.borrow().downcast_ref::<LoadLocal>().expect("rooted at load").get_local();
        let Some(parts) = converted_local(op, local, &self.converter, rewriter) else {
            return Ok(false);
        };
        let span = op.borrow().span();
        rewriter.set_insertion_point_before(op);
        let mut values = SmallVec::<[ValueRef; 2]>::new();
        for (offset, ty) in (0u32..).zip(parts) {
            let part = LocalVariable::new(local.index() + offset, ty);
            let load = rewriter.create_op::<LoadLocal, _>(span, (part,))?;
            values.push(load.borrow().result().as_value_ref());
        }
        rewriter.replace_op_with_multiple(op, &[&values])?;
        Ok(true)
    }
}

/// Splits a `builtin.store_local` to a local whose type is not legal into one store per part.
struct StoreLocalPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for StoreLocalPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for StoreLocalPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let local = *op.borrow().downcast_ref::<StoreLocal>().expect("rooted at store").get_local();
        let Some(parts) = converted_local(op, local, &self.converter, rewriter) else {
            return Ok(false);
        };
        let values = operands.get(0);
        if values.len() != parts.len() {
            return Ok(false);
        }
        let span = op.borrow().span();
        rewriter.set_insertion_point_before(op);
        for ((offset, ty), value) in (0u32..).zip(parts).zip(values) {
            let part = LocalVariable::new(local.index() + offset, ty);
            rewriter.create_op::<StoreLocal, _>(span, (part, *value))?;
        }
        rewriter.erase_op(op)?;
        Ok(true)
    }
}

/// Whether each range in `ranges` is exactly the operand of `op` it stands for.
fn operands_are_themselves<'a>(
    op: &Operation,
    ranges: impl IntoIterator<Item = &'a [ValueRef]>,
) -> bool {
    op.operands()
        .iter()
        .zip(ranges)
        .all(|(operand, range)| *range == [operand.borrow().as_value_ref()])
}

/// The results of `op` (a new call), grouped by the ranges of the old call's results.
fn grouped_results(op: OperationRef, ranges: &[core::ops::Range<u32>]) -> Vec<Vec<ValueRef>> {
    let op = op.borrow();
    let results = op
        .results()
        .all()
        .iter()
        .map(|result| result.borrow().as_value_ref())
        .collect::<Vec<_>>();
    ranges
        .iter()
        .map(|range| results[range.start as usize..range.end as usize].to_vec())
        .collect()
}

/// Replaces `old` by `new`'s results grouped by `ranges`, after giving `new` `old`'s attribute
/// dictionary.
fn replace_with_grouped_results(
    rewriter: &mut ConversionPatternRewriter,
    old: OperationRef,
    mut new: OperationRef,
    attributes: CopiedAttributes,
    ranges: &[core::ops::Range<u32>],
) -> Result<(), Report> {
    attributes.install(&mut new.borrow_mut());
    let groups = grouped_results(new, ranges);
    let groups = groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
    rewriter.replace_op_with_multiple(old, &groups)
}

/// The symbol a direct reference names, and the path it names it by; `None`, with a reason
/// recorded, when it does not resolve to a callable.
fn named_callee(
    op: OperationRef,
    callee: &crate::dialects::builtin::attributes::SymbolRefAttr,
    rewriter: &mut ConversionPatternRewriter,
) -> Option<(crate::CallableSymbolRef, crate::SymbolPath)> {
    match callee.resolve_callable() {
        Ok(resolved) => Some((resolved.named_symbol(), callee.path().clone())),
        Err(err) => {
            rewriter.notify_match_failure(
                op,
                Report::msg(format!("the callee does not resolve to a callable: {err}")),
            );
            None
        }
    }
}

/// Converts a `builtin.call`: a call of the same callee, by the same path, with the converted
/// signature and the operands' ranges flattened.
struct CallPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for CallPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for CallPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let context = op.borrow().context_rc();
        let signature = op
            .borrow()
            .downcast_ref::<Call>()
            .expect("rooted at builtin.call")
            .get_signature()
            .clone();
        let converted = convert_signature(&context, &signature, &self.converter);
        if converted.signature == signature
            && operands_are_themselves(&op.borrow(), operands.iter())
        {
            return Ok(false);
        }
        let (callee, attributes, span) = {
            let op_ref = op.borrow();
            let call = op_ref.downcast_ref::<Call>().expect("rooted at builtin.call");
            let callee = named_callee(op, &call.callee(), rewriter);
            (callee, CopiedAttributes::of(&op_ref), op_ref.span())
        };
        let Some((callee, path)) = callee else {
            return Ok(false);
        };
        let arguments = operands.group_flattened(0);
        rewriter.set_insertion_point_before(op);
        let mut call = rewriter
            .create_op::<Call, _>(span, (callee, converted.signature.clone(), arguments))?;
        call.borrow_mut().callee_mut().set_path(path);
        let call = call.as_operation_ref();
        replace_with_grouped_results(rewriter, op, call, attributes, &converted.results)?;
        Ok(true)
    }
}

/// Converts a `builtin.call_indirect`: a call of the callee's one converted value, with the
/// converted signature and the arguments' ranges flattened.
struct CallIndirectPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for CallIndirectPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for CallIndirectPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let context = op.borrow().context_rc();
        let (signature, attributes, span) = {
            let op_ref = op.borrow();
            let call = op_ref.downcast_ref::<CallIndirect>().expect("rooted at call_indirect");
            (call.get_signature().clone(), CopiedAttributes::of(&op_ref), op_ref.span())
        };
        let converted = convert_signature(&context, &signature, &self.converter);
        if converted.signature == signature
            && operands_are_themselves(&op.borrow(), operands.iter())
        {
            return Ok(false);
        }
        let Some(callee) = operands.one(0) else {
            rewriter.notify_match_failure(
                op,
                Report::msg("the callee, a function value, did not become exactly one value"),
            );
            return Ok(false);
        };
        let arguments = operands.group_flattened(1);
        rewriter.set_insertion_point_before(op);
        let call = rewriter
            .create_op::<CallIndirect, _>(span, (callee, arguments, converted.signature.clone()))?;
        let call = call.as_operation_ref();
        replace_with_grouped_results(rewriter, op, call, attributes, &converted.results)?;
        Ok(true)
    }
}

/// Converts a `builtin.function_ref`: a reference to the same symbol, by the same path, whose
/// type is the converted function type.
struct FunctionReferencePattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for FunctionReferencePattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for FunctionReferencePattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        _operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let context = op.borrow().context_rc();
        let ty = *op
            .borrow()
            .downcast_ref::<FunctionReference>()
            .expect("rooted at function_ref")
            .get_ty();
        // A converter with function types gives a function type exactly one type.
        let new_ty = self.converter.convert_type_1_to_1(&context, ty)?;
        if new_ty == ty {
            return Ok(false);
        }
        let (callee, attributes, span) = {
            let op_ref = op.borrow();
            let reference =
                op_ref.downcast_ref::<FunctionReference>().expect("rooted at function_ref");
            let callee = named_callee(op, &reference.callee(), rewriter);
            (callee, CopiedAttributes::of(&op_ref), op_ref.span())
        };
        let Some((callee, path)) = callee else {
            return Ok(false);
        };
        rewriter.set_insertion_point_before(op);
        let mut reference = rewriter.create_op::<FunctionReference, _>(span, (callee, new_ty))?;
        reference.borrow_mut().callee_mut().set_path(path);
        attributes.install(reference.borrow_mut().as_operation_mut());
        let value = reference.borrow().function().as_value_ref();
        rewriter.replace_op(op, &[value])?;
        Ok(true)
    }
}

/// Converts a `builtin.ret`: a return of the operands' ranges flattened.
struct RetPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for RetPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for RetPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        if operands_are_themselves(&op.borrow(), operands.iter()) {
            return Ok(false);
        }
        let (attributes, span) = {
            let op_ref = op.borrow();
            (CopiedAttributes::of(&op_ref), op_ref.span())
        };
        let values = operands.group_flattened(0);
        rewriter.set_insertion_point_before(op);
        let mut ret = rewriter.create_op::<Ret, _>(span, (values,))?;
        attributes.install(ret.borrow_mut().as_operation_mut());
        rewriter.erase_op(op)?;
        Ok(true)
    }
}

/// Converts the forwarded operands of a branch, in place: for each successor, they are replaced
/// by their ranges flattened. Produced operands and successor keys are not touched.
struct BranchPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for BranchPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for BranchPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        // Per successor, the values its forwarded operands became, flattened.
        let mut forwarded = SmallVec::<[SmallVec<[ValueRef; 4]>; 2]>::new();
        let mut changed = false;
        {
            let op = op.borrow();
            let branch =
                op.as_trait::<dyn BranchOpInterface>().expect("rooted at BranchOpInterface");
            for index in 0..op.num_successors() {
                let successor = branch.get_successor_operands(index);
                let mut values = SmallVec::new();
                for operand in successor.forwarded().iter() {
                    let operand = operand.borrow();
                    let range = operands.get(operand.index as usize);
                    changed |= *range != [operand.as_value_ref()];
                    values.extend_from_slice(range);
                }
                forwarded.push(values);
            }
        }
        if !changed {
            return Ok(false);
        }
        rewriter.modify_op_in_place(op, |op| {
            let owner = op.as_operation_ref();
            let context = op.context_rc();
            let branch =
                op.as_trait_mut::<dyn BranchOpInterface>().expect("rooted at BranchOpInterface");
            for (index, values) in forwarded.into_iter().enumerate() {
                branch
                    .get_successor_operands_mut(index)
                    .forwarded_mut()
                    .set_operands(values, owner, &context);
            }
        })?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests;

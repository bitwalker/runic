//! The model of what a converted function must be, from what the generator knows of it, and the
//! check that every symbol use of a module belongs to it.
//!
//! The model is independent of the patterns: an `i128` is two `i64`s, the marker nothing, a
//! function type the function type of its parts; a declared effect on a parameter or result is
//! on each of its parts, on no value when there is none, and kept once.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{
    Context, OperationRef, Symbol, Type,
    dialects::builtin::{Function, FunctionAlias},
};

use super::xgen::{Model, On, Sig, Ty};
use crate::tests::conversion::all_operations;

/// The shape of a type the checks tell apart.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Class {
    I1,
    I64,
    Ptr,
    Fn(Vec<Class>, Vec<Class>),
    Other(String),
}

fn classify(context: &Context, ty: Type) -> Class {
    if ty == context.i64() {
        Class::I64
    } else if ty == context.i1() {
        Class::I1
    } else if let Some(function) = ty.as_function() {
        Class::Fn(
            function.params().iter().map(|ty| classify(context, *ty)).collect(),
            function.results().iter().map(|ty| classify(context, *ty)).collect(),
        )
    } else if ty.to_string() == "ptr<i64>" {
        Class::Ptr
    } else {
        Class::Other(ty.to_string())
    }
}

/// What `types` become, by the model.
fn expected_classes(shapes: &[Sig], types: &[Ty]) -> Vec<Class> {
    let mut classes = Vec::new();
    for ty in types {
        match ty {
            Ty::X => classes.extend([Class::I64, Class::I64]),
            Ty::N => classes.push(Class::I64),
            Ty::B => classes.push(Class::I1),
            Ty::M => {}
            Ty::P => classes.push(Class::Ptr),
            Ty::F(shape) => classes.push(Class::Fn(
                expected_classes(shapes, &shapes[*shape].params),
                expected_classes(shapes, &shapes[*shape].results),
            )),
        }
    }
    classes
}

/// The new indices each of `types` becomes.
fn ranges(types: &[Ty]) -> Vec<(u32, u32)> {
    let mut start = 0;
    types
        .iter()
        .map(|ty| {
            let count = match ty {
                Ty::X => 2,
                Ty::M => 0,
                _ => 1,
            };
            let range = (start, start + count);
            start += count;
            range
        })
        .collect()
}

/// What the declared effects of `sig` become, by the model: each effect repeated on every part,
/// on no value when there is none, kept once.
fn expected_effects(sig: &Sig) -> Option<Vec<String>> {
    let effects = sig.effects.as_ref()?;
    let (params, results) = (ranges(&sig.params), ranges(&sig.results));
    let mut expected = Vec::<String>::new();
    let mut push = |text: String| {
        if !expected.contains(&text) {
            expected.push(text);
        }
    };
    for (kind, on) in effects {
        match on {
            On::Nothing => push(String::from(*kind)),
            On::Arg(index) => {
                let (start, end) = params[*index as usize];
                if start == end {
                    push(String::from(*kind));
                }
                for new in start..end {
                    push(format!("{kind}(arg {new})"));
                }
            }
            On::Result(index) => {
                let (start, end) = results[*index as usize];
                if start == end {
                    push(String::from(*kind));
                }
                for new in start..end {
                    push(format!("{kind}(result {new})"));
                }
            }
        }
    }
    Some(expected)
}

/// Checks every function of the converted `module` against `model`.
pub(super) fn check_model(
    context: &Context,
    module: OperationRef,
    model: &Model,
) -> Result<(), String> {
    let mut seen = 0;
    for op in all_operations(module) {
        let op = op.borrow();
        let Some(function) = op.downcast_ref::<Function>() else {
            continue;
        };
        let name = function.name().as_str().to_string();
        let index: usize = name[1..].parse().map_err(|_| format!("odd function name @{name}"))?;
        let sig = &model.sigs[index];
        seen += 1;
        let signature = function.get_signature();
        let params = signature
            .params()
            .iter()
            .map(|param| classify(context, param.ty))
            .collect::<Vec<_>>();
        let results = signature
            .results()
            .iter()
            .map(|result| classify(context, result.ty))
            .collect::<Vec<_>>();
        if params != expected_classes(&model.shapes, &sig.params) {
            return Err(format!("MODEL: the parameters of @{name} are {params:?}"));
        }
        if results != expected_classes(&model.shapes, &sig.results) {
            return Err(format!("MODEL: the results of @{name} are {results:?}"));
        }
        let locals = function
            .get_locals()
            .iter()
            .map(|ty| classify(context, *ty))
            .collect::<Vec<_>>();
        if locals != expected_classes(&model.shapes, &model.locals[index]) {
            return Err(format!("MODEL: the locals of @{name} are {locals:?}"));
        }
        let effects = function.get_memory_effects();
        let effects = effects
            .known()
            .map(|effects| effects.iter().map(|effect| effect.to_string()).collect::<Vec<_>>());
        let expected = expected_effects(sig);
        if effects != expected {
            return Err(format!(
                "MODEL: the effects of @{name} are {effects:?}, expected {expected:?}"
            ));
        }
        if !function.is_declaration() {
            let entry = function.body().as_region_ref().borrow().entry_block_ref().unwrap();
            let arguments = entry
                .borrow()
                .argument_values()
                .map(|argument| classify(context, argument.borrow().ty()))
                .collect::<Vec<_>>();
            if arguments != params {
                return Err(format!("MODEL: the entry block of @{name} takes {arguments:?}"));
            }
        }
    }
    if seen != model.sigs.len() {
        return Err(format!("MODEL: {seen} functions, expected {}", model.sigs.len()));
    }
    Ok(())
}

/// Every use in the use list of a function or an alias is held by an operation of the module,
/// and every call, function reference and alias of the module is one use.
pub(super) fn check_symbol_uses(module: OperationRef) -> Result<(), String> {
    use runic_ir::dialects::builtin::{Call, FunctionReference};
    let mut uses = 0;
    let mut users = 0;
    for op in all_operations(module) {
        let op = op.borrow();
        if op.is::<Function>() || op.is::<FunctionAlias>() {
            uses += op.as_symbol_ref().unwrap().borrow().iter_uses().count();
        }
        if op.is::<Call>() || op.is::<FunctionReference>() || op.is::<FunctionAlias>() {
            users += 1;
        }
    }
    if uses != users {
        return Err(format!(
            "SYMBOLS: {uses} uses are linked to the functions and aliases, but {users} calls, \
             references and aliases name them"
        ));
    }
    Ok(())
}

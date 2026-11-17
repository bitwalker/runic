//! A `wasm.global`'s initializer as a WebAssembly constant expression.
//!
//! The initializer is a region of SSA values that ends in `builtin.yield` of the global's value;
//! a constant expression is a tree, rebuilt from the yielded value. Under the backend's feature
//! set (decision 9 of the plan) a constant expression is one instruction that takes no operand:
//! `wasm.const`, `wasm.constf`, `wasm.ref_null`, `wasm.ref_func` (a function fix-up),
//! `wasm.global_get` (a global fix-up), `wasm.data_addr` and `wasm.func_index` (value fix-ups).
//! So the tree is that one operation, and nothing is shared. `wasm.add`, `wasm.sub` and `wasm.mul`
//! need the extended-const feature, which is outside the feature set: each is an error naming the
//! operation and the global. A `wasm.global_get` of a global the module defines (rather than
//! imports) needs the constant expressions of the GC proposal, also outside it, and is an error
//! too. Any other operation is an error naming it.

use alloc::{format, string::String};

use runic_dialect_wasm as wasm;
use runic_ir::{Operation, Report, Spanned, diagnostics::report};

use super::{
    desc::{ConstExpr, ModuleDesc},
    instr::{Encoder, Entities, encode, name_of, unsupported},
};

/// The operations a constant expression may be, under the backend's feature set.
const CONSTANT: [&str; 7] = [
    "wasm.const",
    "wasm.constf",
    "wasm.ref_null",
    "wasm.ref_func",
    "wasm.global_get",
    "wasm.data_addr",
    "wasm.func_index",
];

/// The operations of the extended-const feature.
const EXTENDED: [&str; 3] = ["wasm.add", "wasm.sub", "wasm.mul"];

/// The initializer of `global`, a `wasm.global` definition, as a constant expression of `desc`,
/// with the entities its symbols name found through `entities`.
pub fn initializer(
    global: &Operation,
    desc: &mut ModuleDesc,
    entities: &mut dyn Entities,
) -> Result<ConstExpr, Report> {
    let Some(definition) = global.downcast_ref::<wasm::Global>() else {
        return Err(unsupported(global, "it is not a `wasm.global`"));
    };
    let name = definition.get_name().as_symbol();
    let value = definition
        .initializer_value()
        .ok_or_else(|| unsupported(global, "its initializer yields no value of its type"))?;
    let op = value.borrow().get_defining_op().ok_or_else(|| {
        unsupported(global, "its initializer yields a value no operation defines")
    })?;
    let op = op.borrow();
    let op_name = name_of(&op);
    if EXTENDED.contains(&op_name.as_str()) {
        return Err(outside_baseline(
            &op,
            global,
            format!(
                "`{op_name}` in the initializer of `@{name}` needs the extended-const feature, \
                 which is outside the backend's feature set"
            ),
        ));
    }
    if !CONSTANT.contains(&op_name.as_str()) {
        return Err(outside_baseline(
            &op,
            global,
            format!("`{op_name}` is not a constant expression of WebAssembly"),
        ));
    }
    if let Some(get) = op.downcast_ref::<wasm::GlobalGet>() {
        let read = entities.global(&op, &get.global())?;
        if !desc.globals()[read.index() as usize].entity.is_import() {
            return Err(outside_baseline(
                &op,
                global,
                format!(
                    "in the initializer of `@{name}`, `wasm.global_get` reads a global the module \
                     defines, which needs the constant expressions of the GC feature, outside the \
                     backend's feature set: only an imported global can be read"
                ),
            ));
        }
    }
    let mut expr = ConstExpr::default();
    {
        let mut writer = expr.writer();
        let mut cx = Encoder {
            desc,
            entities,
            params: 0,
        };
        encode(&op, &mut writer, &mut cx)?;
    }
    Ok(expr)
}

/// The error of `op` in the initializer of `global`, with `label` saying why it cannot be
/// emitted.
fn outside_baseline(op: &Operation, global: &Operation, label: String) -> Report {
    report!(
        message: (format!("the operation `{}` cannot be emitted as WebAssembly", name_of(op))),
        labels: [
            primary(op.span(), label),
            context(global.span(), "the global it initializes"),
        ],
    )
}

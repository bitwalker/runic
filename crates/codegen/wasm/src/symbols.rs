//! The well-known symbols lowered code names: the linear memory, the function table and the
//! stack pointer.
//!
//! Each is found in a `builtin.module` by name, or declared there. A symbol of its name and kind
//! that the module holds, a definition or a declaration, is used as it is. Otherwise a declaration
//! (an import, from the module `env`) is created at the start of the module's body (a body is
//! created for a module that has none), private, with
//! the minimum limits 0 and no maximum where it has limits, and with the attribute `link_name` set
//! to the symbol's own name: the default link name is the symbol's path (`m::__linear_memory`),
//! while the name a linker knows it by is the bare name.

use alloc::{format, rc::Rc};

use runic_dialect_wasm::{
    FuncRefType, Global, GlobalRef, Limits, Memory, MemoryRef, Mutability, Table, TableRef,
    WasmOpBuilder,
};
use runic_ir::{
    Builder, Context, Ident, LINK_NAME, Op, OpBuilder, OperationRef, Report, Spanned, SymbolTable,
    UnsafeIntrusiveEntityRef, Visibility,
    diagnostics::report,
    dialects::builtin::{ModuleRef, attributes::StringAttr},
    interner,
};

/// The name of the linear memory: `@__linear_memory`, a `wasm.memory`.
pub const LINEAR_MEMORY: &str = "__linear_memory";

/// The name of the table of the functions an indirect call reaches: `@__indirect_function_table`,
/// a `wasm.table` of `!wasm.funcref`.
pub const INDIRECT_FUNCTION_TABLE: &str = "__indirect_function_table";

/// The name of the stack pointer of the shadow stack in linear memory: `@__stack_pointer`, a
/// `wasm.global` of type `mut i32`.
pub const STACK_POINTER: &str = "__stack_pointer";

/// The linear memory of `module`, `wasm.memory @__linear_memory`: the module's own, or a
/// declaration created for it.
///
/// An error, naming both, when the module holds a symbol of that name that is not a
/// `wasm.memory`.
pub fn linear_memory(module: ModuleRef) -> Result<MemoryRef, Report> {
    find_or_declare::<Memory>(
        module,
        LINEAR_MEMORY,
        "wasm.memory",
        |_, _| None,
        |builder, name| {
            builder.create_memory_declaration(name, Visibility::Private, Limits::at_least(0))
        },
    )
}

/// The table of `module` that indirect calls go through, `wasm.table @__indirect_function_table`
/// of `!wasm.funcref`: the module's own, or a declaration created for it.
///
/// An error, naming both, when the module holds a symbol of that name that is not a `wasm.table`,
/// or is one of another element type.
pub fn indirect_function_table(module: ModuleRef) -> Result<TableRef, Report> {
    find_or_declare::<Table>(
        module,
        INDIRECT_FUNCTION_TABLE,
        "wasm.table",
        |context, table: &Table| {
            let ty = *table.get_element_type();
            (ty != FuncRefType::get(context))
                .then(|| format!("a `wasm.table` of `{ty}`, not of `!wasm.funcref`"))
        },
        |builder, name| {
            let funcref = FuncRefType::get(builder.context());
            builder.create_table_declaration(
                name,
                Visibility::Private,
                Limits::at_least(0),
                funcref,
            )
        },
    )
}

/// The stack pointer of `module`, `wasm.global @__stack_pointer : mut i32`: the module's own, or a
/// declaration created for it.
///
/// An error, naming both, when the module holds a symbol of that name that is not a
/// `wasm.global`, or is one of another type or not mutable.
pub fn stack_pointer(module: ModuleRef) -> Result<GlobalRef, Report> {
    find_or_declare::<Global>(
        module,
        STACK_POINTER,
        "wasm.global",
        |context, global: &Global| {
            let ty = *global.get_ty();
            let mutable = global.get_mutability().is_mut();
            (ty != context.i32() || !mutable).then(|| {
                let mutability = if mutable { "mut " } else { "" };
                format!("a `wasm.global` of type `{mutability}{ty}`, not of type `mut i32`")
            })
        },
        |builder, name| {
            let i32 = builder.context().i32();
            builder.create_global_declaration(name, Visibility::Private, i32, Mutability::Mut)
        },
    )
}

/// The symbol `name` of `module` if it is a `T` that `mismatch` accepts; otherwise an error when
/// the module holds a symbol of that name; otherwise the declaration `declare` creates at the start
/// of the module's body, given the `link_name` of `name`.
///
/// `kind` is the name of `T` as the error spells it. `mismatch` says what is wrong with a `T` of
/// that name, or `None` when it is the one wanted.
fn find_or_declare<T: Op>(
    module: ModuleRef,
    name: &str,
    kind: &str,
    mismatch: impl FnOnce(&Context, &T) -> Option<alloc::string::String>,
    declare: impl FnOnce(&mut OpBuilder, Ident) -> Result<UnsafeIntrusiveEntityRef<T>, Report>,
) -> Result<UnsafeIntrusiveEntityRef<T>, Report> {
    let context = module.borrow().as_operation().context_rc();
    let symbol = interner::Symbol::intern(name);
    let found: Option<OperationRef> = module.borrow().symbol_manager().lookup_op(symbol);
    if let Some(found) = found {
        let found_name = found.borrow().name();
        let problem = match found.try_downcast_op::<T>() {
            Ok(entity) => match mismatch(&context, &entity.borrow()) {
                None => return Ok(entity),
                Some(problem) => problem,
            },
            Err(_) => format!("a `{found_name}`, not a `{kind}`"),
        };
        return Err(report!(
            message: (format!(
                "the module names `@{name}` the {kind} that lowered code uses, but it holds \
                 another symbol of that name"
            )),
            labels: [primary(found.borrow().span(), format!("`@{name}` is {problem}"))],
        ));
    }

    let (region, body) = {
        let module = module.borrow();
        let region = module.body();
        (region.as_region_ref(), region.entry_block_ref())
    };
    let mut builder = OpBuilder::new(Rc::clone(&context));
    // A module written with nothing in it has no block yet.
    let body = body.unwrap_or_else(|| builder.create_block(region, None, &[]));
    builder.set_insertion_point_to_start(body);
    let mut entity = declare(&mut builder, Ident::with_empty_span(symbol))?;
    let link_name = context.create_attribute::<StringAttr, _>(name).as_attribute_ref();
    entity.borrow_mut().as_operation_mut().set_attribute(LINK_NAME, link_name);
    Ok(entity)
}

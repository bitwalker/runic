//! The module-level operations, and the dictionary entries that mark imports and exports.

use alloc::{format, string::ToString, vec::Vec};

use runic_ir::{
    Builder, Ident, Op, OperationRef, SourceSpan, Symbol, SymbolTable, Visibility,
    dialects::builtin::{BuiltinOpBuilder, Function, Module, ModuleRef},
    interner,
    testing::{
        Test,
        text::{print_generic, round_trip},
    },
};

use super::*;

/// A module `@m` that holds `body`.
fn module_text(body: &str) -> String {
    format!("builtin.module public @m {{\n{body}\n}};")
}

/// `source`, a module, parsed and verified, or the error with its labels.
fn parse_text(source: &str) -> Result<(Rc<Context>, ModuleRef), String> {
    let context = context();
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "m", source)
        .into_result()
        .map_err(|report| describe(&report))?;
    module
        .borrow()
        .as_operation()
        .recursively_verify()
        .map_err(|report| format!("does not verify: {}", describe(&report)))?;
    Ok((context, module))
}

/// The module that holds `body`, parsed and verified.
fn module_of(body: &str) -> (Rc<Context>, ModuleRef) {
    parse_text(&module_text(body)).unwrap_or_else(|error| panic!("{error}"))
}

/// The error of the module that holds `body`; it must not parse or verify.
fn module_error(body: &str) -> String {
    match parse_text(&module_text(body)) {
        Ok(_) => panic!("accepted:\n{body}"),
        Err(error) => error,
    }
}

/// The text of `module`.
fn text_of(module: &ModuleRef) -> String {
    module.borrow().as_operation().to_string()
}

/// The operation named `name` of the module `module`.
fn symbol(module: &ModuleRef, name: &str) -> OperationRef {
    module
        .borrow()
        .symbol_manager()
        .lookup_op(interner::Symbol::intern(name))
        .unwrap_or_else(|| panic!("the module has no symbol `{name}`"))
}

/// One of each operation, with their imports and exports; every name is unique, so that the
/// generic form of each operation is substituted once.
const EVERY_OPERATION: &str = r#"
    wasm.memory public @mem : 1..16 attributes { wasm.export_name = "memory" };
    wasm.memory public declaration @imem : 2 attributes { wasm.import_module = "host" };
    wasm.table private @tab : 0..8 x !wasm.funcref;
    wasm.table private declaration @itab : 4 x !wasm.externref;
    wasm.global private @sp : mut i32 {
        %v = wasm.const 65536 : i32;
    };
    wasm.global private @size : i64 {
        %a = wasm.const 8 : i64;
        %b = wasm.const 4 : i64;
        %c = wasm.mul %a, %b;
    };
    wasm.global private @iglobal : f32;
    wasm.tag private @exn : (i32, i64);
    wasm.tag private declaration @itag : (f32);
    builtin.function public extern("C") @f() {
        builtin.ret;
    };
    builtin.function public extern("C") @g();
    wasm.elem private @init : @tab at 0 [@f, @g];
"#;

#[test]
fn every_module_operation_round_trips_in_the_custom_and_the_generic_form() {
    let (_context, module) = module_of(EVERY_OPERATION);
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(skipped.is_empty(), "{skipped:?}");
}

#[test]
fn the_module_operations_print_the_text_forms_the_documentation_gives() {
    let (_context, module) = module_of(EVERY_OPERATION);
    let text = text_of(&module);
    for line in [
        "wasm.memory public @mem : 1..16 attributes { wasm.export_name = \
         #builtin.string<\"memory\"> };",
        "wasm.memory public declaration @imem : 2 attributes { wasm.import_module = \
         #builtin.string<\"host\"> };",
        "wasm.table private @tab : 0..8 x !wasm.funcref;",
        "wasm.table private declaration @itab : 4 x !wasm.externref;",
        "wasm.global private @sp : mut i32 {",
        "wasm.global private @iglobal : f32;",
        "wasm.tag private @exn : (i32, i64);",
        "wasm.tag private declaration @itag : (f32);",
        "wasm.elem private @init : @tab at 0 [@f, @g];",
    ] {
        assert!(text.contains(line), "`{line}` is not in\n{text}");
    }
}

#[test]
fn the_accessors_read_the_properties_of_each_operation() {
    let (context, module) = module_of(EVERY_OPERATION);
    let memory = symbol(&module, "mem").try_downcast_op::<Memory>().unwrap();
    assert_eq!(*memory.borrow().get_limits(), Limits::between(1, 16));
    assert!(!memory.borrow().is_declaration());
    let imported = symbol(&module, "imem").try_downcast_op::<Memory>().unwrap();
    assert_eq!(*imported.borrow().get_limits(), Limits::at_least(2));
    assert!(imported.borrow().is_declaration());

    let table = symbol(&module, "tab").try_downcast_op::<Table>().unwrap();
    assert_eq!(*table.borrow().get_element_type(), FuncRefType::get(&context));
    assert_eq!(*table.borrow().get_limits(), Limits::between(0, 8));
    let imported = symbol(&module, "itab").try_downcast_op::<Table>().unwrap();
    assert!(imported.borrow().is_declaration());
    assert_eq!(*imported.borrow().get_element_type(), ExternRefType::get(&context));

    let sp = symbol(&module, "sp").try_downcast_op::<Global>().unwrap();
    assert_eq!(*sp.borrow().get_ty(), context.i32());
    assert!(sp.borrow().get_mutability().is_mut());
    assert!(!sp.borrow().is_declaration());
    let size = symbol(&module, "size").try_downcast_op::<Global>().unwrap();
    assert_eq!(*size.borrow().get_mutability(), Mutability::Const);
    let imported = symbol(&module, "iglobal").try_downcast_op::<Global>().unwrap();
    assert!(imported.borrow().is_declaration());
    assert_eq!(*imported.borrow().get_ty(), context.f32());

    let tag = symbol(&module, "exn").try_downcast_op::<Tag>().unwrap();
    assert_eq!(tag.borrow().param_types().as_slice(), [context.i32(), context.i64()]);
    assert!(!tag.borrow().is_declaration());
    assert!(
        symbol(&module, "itag")
            .try_downcast_op::<Tag>()
            .unwrap()
            .borrow()
            .is_declaration()
    );

    let elem = symbol(&module, "init").try_downcast_op::<Elem>().unwrap();
    let elem = elem.borrow();
    assert_eq!(elem.offset_value(), 0);
    assert!(elem.table_op() == Some(table));
    let functions = elem
        .function_refs()
        .iter()
        .map(|reference| reference.borrow().path().to_string())
        .collect::<Vec<_>>();
    assert_eq!(functions, ["f", "g"]);
    assert!(!elem.is_declaration());
}

#[test]
fn an_element_offset_is_read_as_a_constant_i32_is() {
    let (_context, module) = module_of(
        "wasm.table private @tab : 8 x !wasm.funcref;
         builtin.function public extern(\"C\") @f();
         wasm.elem private @a : @tab at 0xffffffff [@f];
         wasm.elem private @b : @tab at -1 [@f];",
    );
    let a = symbol(&module, "a").try_downcast_op::<Elem>().unwrap();
    let b = symbol(&module, "b").try_downcast_op::<Elem>().unwrap();
    assert_eq!(a.borrow().offset_value(), u32::MAX);
    assert_eq!(b.borrow().offset_value(), u32::MAX);
    assert!(text_of(&module).contains("@tab at -1 [@f]"));
    let error = module_error(
        "wasm.table private @tab : 8 x !wasm.funcref;
         wasm.elem private @a : @tab at 4294967296 [];",
    );
    assert!(error.contains("out of range for i32"), "{error}");
}

#[test]
fn a_declaration_is_a_global_without_a_region_and_an_empty_pair_of_braces_is_one() {
    let (_context, module) = module_of(
        "wasm.global private @a : i32;
         wasm.global private @b : mut i64 { };",
    );
    for name in ["a", "b"] {
        let global = symbol(&module, name).try_downcast_op::<Global>().unwrap();
        assert!(global.borrow().is_declaration(), "{name}");
    }
    assert!(text_of(&module).contains("wasm.global private @b : mut i64;"));
}

#[test]
fn a_memory_declaration_and_a_definition_differ_in_the_property_declaration() {
    let (_context, module) = module_of(
        "wasm.memory public @a : 1;
         wasm.memory public declaration @b : 1;",
    );
    let generic = print_generic(&symbol(&module, "b").borrow());
    assert!(generic.contains("declaration = #builtin.bool<true>"), "{generic}");
    let generic = print_generic(&symbol(&module, "a").borrow());
    assert!(generic.contains("declaration = #builtin.bool<false>"), "{generic}");
}

#[test]
fn a_symbol_of_each_kind_shares_the_one_name_space_of_its_module() {
    let error = module_error(
        "wasm.global private @x : i32;
         builtin.function public extern(\"C\") @x();",
    );
    assert!(error.contains("redefinition of symbol"), "{error}");
    let error = module_error(
        "wasm.memory public @x : 1;
         wasm.table private @x : 1 x !wasm.funcref;",
    );
    assert!(error.contains("redefinition of symbol"), "{error}");
}

#[test]
fn every_module_operation_is_a_symbol_and_a_module_entity() {
    let (_context, module) = module_of(EVERY_OPERATION);
    for name in ["mem", "imem", "tab", "itab", "sp", "size", "iglobal", "exn", "itag", "init"] {
        let op = symbol(&module, name);
        assert!(op.borrow().implements::<dyn Symbol>(), "{name}");
        assert!(op.borrow().implements::<dyn ModuleEntity>(), "{name}");
    }
    assert!(!symbol(&module, "f").borrow().implements::<dyn ModuleEntity>());
}

/// The module that holds `body`, which must not verify, and a fragment of the first message.
#[track_caller]
fn rejected(body: &str, message: &str) {
    let error = module_error(body);
    // A message with a quotation mark is shown escaped inside the labels
    let escaped = message.replace('"', "\\\"");
    assert!(
        error.contains(message) || error.contains(&escaped),
        "expected `{message}` in\n{error}"
    );
}

#[test]
fn a_memory_has_limits_within_65536_pages_and_ordered() {
    rejected(
        "wasm.memory public @mem : 65537;",
        "`wasm.memory` `@mem` has a minimum of 65537, but a 32-bit memory has at most 65536",
    );
    rejected(
        "wasm.memory public @mem : 1..65537;",
        "`wasm.memory` `@mem` has a maximum of 65537, but a 32-bit memory has at most 65536",
    );
    rejected(
        "wasm.memory public @mem : 8..4;",
        "limits `8..4` with a maximum below the minimum, but Wasm limits satisfy min <= max",
    );
    module_of("wasm.memory public @a : 65536; wasm.memory public @b : 0..65536;");
}

#[test]
fn a_table_has_limits_within_32_bits_and_a_reference_element_type() {
    rejected(
        "wasm.table private @tab : 0..4294967296 x !wasm.funcref;",
        "`wasm.table` `@tab` has a maximum of 4294967296, but a Wasm table has at most 4294967295",
    );
    rejected(
        "wasm.table private @tab : 4294967296 x !wasm.funcref;",
        "`wasm.table` `@tab` has a minimum of 4294967296, but a Wasm table has at most 4294967295",
    );
    rejected(
        "wasm.table private @tab : 5..1 x !wasm.funcref;",
        "limits `5..1` with a maximum below the minimum",
    );
    for ty in ["i32", "f64", "i8", "ptr<i32>"] {
        rejected(
            &format!("wasm.table private @tab : 0 x {ty};"),
            &format!(
                "`wasm.table` `@tab` has the element type `{ty}`, but a Wasm table holds \
                 references: `funcref` or `externref`"
            ),
        );
    }
    module_of(
        "wasm.table private @a : 4294967295 x !wasm.funcref;
         wasm.table private @b : 0..4294967295 x !wasm.externref;",
    );
}

#[test]
fn a_global_has_a_value_type() {
    for ty in ["i8", "i1", "isize", "ptr<i32>"] {
        rejected(
            &format!("wasm.global private @g : {ty};"),
            &format!(
                "`wasm.global` `@g` has the type `{ty}`, but a Wasm global holds a value: `i32`, \
                 `i64`, `f32`, `f64`, `funcref` or `externref`"
            ),
        );
    }
    for ty in ["i32", "i64", "f32", "f64", "!wasm.funcref", "!wasm.externref"] {
        module_of(&format!("wasm.global private @g : {ty};"));
        module_of(&format!("wasm.global private @m : mut {ty};"));
    }
}

#[test]
fn the_initializer_of_a_global_is_a_constant_expression_of_the_globals_type() {
    rejected(
        "wasm.global private @g : i32 {
            %v = wasm.const 1 : i64;
         };",
        "the initializer of `wasm.global` `@g` has the value type `i64`, but the global has the \
         type `i32`",
    );
    rejected(
        "wasm.global private @g : f32 {
            %v = wasm.const 1 : i32;
         };",
        "has the value type `i32`, but the global has the type `f32`",
    );
    rejected(
        "wasm.global private @g : i32 {
            %a = wasm.const 1 : i32;
            %b = wasm.div_s %a, %a;
         };",
        "the initializer of `wasm.global` `@g` holds `wasm.div_s`, but a Wasm constant expression \
         holds only constant instructions",
    );
    rejected(
        "wasm.global private @g : f32 {
            %a = wasm.constf 1.0 : f32;
            %b = wasm.add %a, %a;
         };",
        "the initializer of `wasm.global` `@g` holds `wasm.add`, but a Wasm constant expression \
         holds only constant instructions",
    );
    // The first operation is checked as the last one is
    rejected(
        "wasm.global private @g : i32 {
            %a = wasm.const 1 : i32;
            %b = wasm.eqz %a;
            %c = wasm.const 2 : i32;
         };",
        "holds `wasm.eqz`",
    );
    module_of(
        "wasm.global private @g : i32 {
            %a = wasm.const 1 : i32;
            %b = wasm.const 2 : i32;
            %c = wasm.add %a, %b;
            %d = wasm.sub %c, %a;
            %e = wasm.mul %d, %d;
         };
         wasm.global private @h : f64 {
            %a = wasm.constf 2.5 : f64;
         };",
    );
}

#[test]
fn the_initializer_of_a_global_uses_only_values_of_its_own_region() {
    let error = module_error(
        "builtin.function public extern(\"C\") @f(%x: i32) {
            builtin.ret;
         };
         wasm.global private @g : i32 {
            %v = wasm.add %x, %x;
         };",
    );
    assert!(error.contains("%x") || error.contains("use of undeclared"), "{error}");
}

#[test]
fn a_global_whose_initializer_has_a_block_and_no_operation_is_rejected() {
    let mut test = Test::named("f").in_module("m");
    let context = test.context_rc();
    context.get_or_register_dialect::<WasmDialect>();
    let module = test.module();
    let body = module.borrow().body().entry_block_ref().unwrap();
    let builder = test.builder_mut();
    builder.set_insertion_point_to_end(body);
    let global = builder
        .create_global_declaration(
            Ident::with_empty_span("g".into()),
            Visibility::Private,
            context.i32(),
            Mutability::Const,
        )
        .unwrap();
    let region = global.borrow().initializer().as_region_ref();
    builder.create_block(region, None, &[]);
    let error = module.borrow().as_operation().recursively_verify().unwrap_err();
    let error = describe(&error);
    assert!(
        error.contains(
            "`wasm.global` `@g` has an initializer with no operation, but the initializer of a \
             definition computes the value of the global (a declaration has no region)"
        ),
        "{error}"
    );
}

#[test]
fn an_element_segment_names_a_funcref_table_an_i32_offset_and_functions() {
    rejected(
        "wasm.memory public @mem : 1;
         builtin.function public extern(\"C\") @f();
         wasm.elem private @e : @mem at 0 [@f];",
        "`wasm.elem` names `mem`, which is a `wasm.memory`, not a `wasm.table`",
    );
    rejected(
        "builtin.function public extern(\"C\") @f();
         wasm.elem private @e : @f at 0 [@f];",
        "`wasm.elem` names `f`, which is a `builtin.function`, not a `wasm.table`",
    );
    rejected(
        "wasm.table private @tab : 1 x !wasm.externref;
         builtin.function public extern(\"C\") @f();
         wasm.elem private @e : @tab at 0 [@f];",
        "`wasm.elem` `@e` fills the table `tab` of `externref`, but a segment of functions fills \
         a `funcref` table",
    );
    rejected(
        "wasm.table private @tab : 1 x !wasm.funcref;
         wasm.memory public @mem : 1;
         wasm.elem private @e : @tab at 0 [@mem];",
        "`wasm.elem` `@e` lists `mem`, which is a `wasm.memory`, but every element of a segment \
         of functions is a function",
    );
    rejected(
        "wasm.table private @tab : 1 x !wasm.funcref;
         wasm.global private @g : i32;
         wasm.elem private @e : @tab at 0 [@g];",
        "lists `g`, which is a `wasm.global`",
    );
    rejected(
        "wasm.table private @tab : 1 x !wasm.funcref;
         builtin.function public extern(\"C\") @f();
         \"wasm.elem\"() <{ name = #builtin.ident<@e>, visibility = #builtin.visibility<private>, \
         table = #builtin.symbol<@tab>, offset = #builtin.int<0 : i64>, functions = \
         [#builtin.symbol<@f>] }> : () -> ();",
        "`wasm.elem` `@e` has an offset of type `i64`, but the offset of a segment is an `i32` \
         constant",
    );
    // A symbol that names nothing is reported by the check every symbol use gets
    let error = module_error(
        "wasm.table private @tab : 1 x !wasm.funcref;
         wasm.elem private @e : @tab at 0 [@missing];",
    );
    assert!(error.contains("missing"), "{error}");
    let error = module_error("wasm.elem private @e : @missing at 0 [];");
    assert!(error.contains("missing"), "{error}");
    // An alias is a callable
    module_of(
        "wasm.table private @tab : 1 x !wasm.funcref;
         builtin.function public extern(\"C\") @f();
         builtin.function_alias public @a -> @f;
         wasm.elem private @e : @tab at 0 [@a, @f];",
    );
}

#[test]
fn a_tag_takes_value_types() {
    rejected(
        "wasm.tag private @t : (i32, i1);",
        "`wasm.tag` `@t` has parameter 1 of type `i1`, but a Wasm tag takes values: `i32`, `i64`, \
         `f32`, `f64`, `funcref` or `externref`",
    );
    module_of("wasm.tag private @t : (i32, i64, f32, f64, !wasm.funcref, !wasm.externref);");
}

#[test]
fn malformed_text_of_a_module_operation_is_an_error_and_not_a_panic() {
    for body in [
        "wasm.memory public @m : ;",
        "wasm.memory public @m : 1..;",
        "wasm.memory public @m : a;",
        "wasm.memory @m : 1;",
        "wasm.memory public @m 1;",
        "wasm.memory public declaration declaration @m : 1;",
        "wasm.table private @t : 1 !wasm.funcref;",
        "wasm.table private @t : 1 x;",
        "wasm.table private @t : 1 y !wasm.funcref;",
        "wasm.global private @g : mut;",
        "wasm.global private @g : mut mut i32;",
        "wasm.global private @g : ;",
        "wasm.global private @g : i32 = 1;",
        "wasm.elem private @e : @t [];",
        "wasm.elem private @e : @t at [];",
        "wasm.elem private @e : @t at 0 @f;",
        "wasm.elem private @e : @t at 0 [@f, ];",
        "wasm.tag private @t : i32;",
        "wasm.tag private @t : (i32;",
        "wasm.tag private @t : (,);",
    ] {
        let error = module_error(body);
        assert!(!error.is_empty(), "{body}");
    }
}

#[test]
fn the_builders_make_each_operation_and_the_module_verifies() {
    let mut test = Test::named("f").in_module("m");
    let context = test.context_rc();
    context.get_or_register_dialect::<WasmDialect>();
    let module = test.module();
    let body = module.borrow().body().entry_block_ref().unwrap();
    let ident = |name: &str| Ident::with_empty_span(name.to_string().into());
    {
        let builder = test.builder_mut();
        builder.set_insertion_point_to_end(body);
        builder
            .create_memory(ident("mem"), Visibility::Public, Limits::between(1, 16))
            .unwrap();
        builder
            .create_memory_declaration(ident("imem"), Visibility::Public, Limits::at_least(1))
            .unwrap();
        let table = builder
            .create_table(
                ident("tab"),
                Visibility::Private,
                Limits::at_least(2),
                FuncRefType::get(&context),
            )
            .unwrap();
        builder
            .create_table_declaration(
                ident("itab"),
                Visibility::Private,
                Limits::at_least(2),
                ExternRefType::get(&context),
            )
            .unwrap();
        builder
            .create_global(ident("sp"), Visibility::Private, context.i32(), Mutability::Mut, |b| {
                Ok(b.i32(65536, SourceSpan::UNKNOWN))
            })
            .unwrap();
        builder
            .create_global(
                ident("sum"),
                Visibility::Private,
                context.i64(),
                Mutability::Const,
                |b| {
                    let a = b.i64(1, SourceSpan::UNKNOWN);
                    let c = b.i64(2, SourceSpan::UNKNOWN);
                    b.add(a, c, SourceSpan::UNKNOWN)
                },
            )
            .unwrap();
        builder
            .create_global_declaration(
                ident("ext"),
                Visibility::Private,
                context.f64(),
                Mutability::Mut,
            )
            .unwrap();
        builder
            .create_tag(ident("exn"), Visibility::Private, &[context.i32(), context.i64()])
            .unwrap();
        builder.create_tag_declaration(ident("itag"), Visibility::Private, &[]).unwrap();
        let f = builder
            .create_function(
                ident("f"),
                Visibility::Public,
                runic_ir::dialects::builtin::attributes::Signature::new(&context, [], []),
            )
            .unwrap();
        builder.create_elem(ident("init"), Visibility::Private, table, 3, [f]).unwrap();
    }
    module
        .borrow()
        .as_operation()
        .recursively_verify()
        .unwrap_or_else(|e| panic!("{}", describe(&e)));
    let text = text_of(&module);
    for line in [
        "wasm.memory public @mem : 1..16;",
        "wasm.memory public declaration @imem : 1;",
        "wasm.table private @tab : 2 x !wasm.funcref;",
        "wasm.table private declaration @itab : 2 x !wasm.externref;",
        "wasm.global private @sp : mut i32 {",
        "wasm.const 65536 : i32;",
        "wasm.add",
        "wasm.global private @ext : mut f64;",
        "wasm.tag private @exn : (i32, i64);",
        "wasm.tag private declaration @itag : ();",
        "wasm.elem private @init : ::@m::@tab at 3 [::@m::@f];",
    ] {
        assert!(text.contains(line), "`{line}` is not in\n{text}");
    }
    // The segment's reference is linked: the function has a use
    let f = symbol(&module, "f");
    assert_eq!(f.borrow().as_symbol().unwrap().iter_uses().count(), 1);
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn a_builder_reports_a_name_that_is_taken_and_does_not_panic() {
    let mut test = Test::named("f").in_module("m");
    let context = test.context_rc();
    context.get_or_register_dialect::<WasmDialect>();
    let module = test.module();
    let body = module.borrow().body().entry_block_ref().unwrap();
    let builder = test.builder_mut();
    builder.set_insertion_point_to_end(body);
    let name = || Ident::with_empty_span("x".into());
    builder.create_memory(name(), Visibility::Public, Limits::at_least(1)).unwrap();
    assert!(builder.create_memory(name(), Visibility::Public, Limits::at_least(1)).is_err());
    assert!(
        builder.create_tag(name(), Visibility::Public, &[]).is_err(),
        "a tag and a memory of one name collide"
    );
    assert!(builder.create_tag_declaration(name(), Visibility::Public, &[]).is_err());
    assert!(
        builder
            .create_global(name(), Visibility::Public, context.i32(), Mutability::Const, |b| Ok(
                b.i32(0, SourceSpan::UNKNOWN)
            ))
            .is_err()
    );
}

#[test]
fn referenced_entity_checks_the_kind_of_what_a_symbol_names() {
    // `wasm.elem`'s verifier is the model: the table it names is a table
    let (_context, module) = module_of(EVERY_OPERATION);
    let elem = symbol(&module, "init").try_downcast_op::<Elem>().unwrap();
    let elem = elem.borrow();
    let table = referenced_entity::<Table>(elem.as_operation(), &elem.table(), "wasm.table");
    assert!(table.is_ok());
    let error = referenced_entity::<Memory>(elem.as_operation(), &elem.table(), "wasm.memory")
        .map(|_| ())
        .unwrap_err();
    let error = describe(&error);
    assert!(
        error.contains("`wasm.elem` names `tab`, which is a `wasm.table`, not a `wasm.memory`"),
        "{error}"
    );
}

/// A function `@f` with the dictionary `attributes`, and a module that holds it and `body`.
fn with_attributes(text: &str) -> Result<(Rc<Context>, ModuleRef), String> {
    parse_text(&module_text(text))
}

#[test]
fn an_import_is_a_declaration_of_a_function_a_global_a_memory_a_table_or_a_tag() {
    let (_context, module) = with_attributes(
        "builtin.function public extern(\"C\") @f() attributes { wasm.import_module = \"other\" };
         wasm.global private @g : i32 attributes { wasm.import_module = \"env\" };
         wasm.memory public declaration @m : 1 attributes { wasm.import_module = \"env\" };
         wasm.table private declaration @t : 1 x !wasm.funcref attributes { wasm.import_module = \
         \"env\" };
         wasm.tag private declaration @e : (i32) attributes { wasm.import_module = \"env\" };",
    )
    .unwrap();
    let modules = ["f", "g", "m", "t", "e"]
        .map(|name| import_module(&symbol(&module, name).borrow()).map(|m| m.to_string()));
    assert_eq!(
        modules,
        [
            Some("other".into()),
            Some("env".into()),
            Some("env".into()),
            Some("env".into()),
            Some("env".into())
        ]
    );
}

#[test]
fn an_import_with_no_module_comes_from_env() {
    let (_context, module) = module_of(
        "builtin.function public extern(\"C\") @f();
         wasm.memory public declaration @m : 1;
         wasm.memory public @definition : 1;
         builtin.global_variable public @data : i32;",
    );
    assert_eq!(
        import_module(&symbol(&module, "f").borrow()).as_deref(),
        Some(DEFAULT_IMPORT_MODULE)
    );
    assert_eq!(import_module(&symbol(&module, "m").borrow()).as_deref(), Some("env"));
    // A definition and data in linear memory are not imports
    assert_eq!(import_module(&symbol(&module, "definition").borrow()), None);
    assert_eq!(import_module(&symbol(&module, "data").borrow()), None);
}

#[test]
fn an_export_is_a_definition_with_one_name_or_an_array_of_names() {
    let (_context, module) = module_of(
        "builtin.function public extern(\"C\") @f() {
            builtin.ret;
         } attributes { wasm.export_name = \"f\" };
         wasm.memory public @m : 1 attributes { wasm.export_name = [\"memory\", \"mem\"] };
         wasm.global private @g : i32 { %v = wasm.const 0 : i32; } attributes { wasm.export_name = \
         \"g\" };
         wasm.table private @t : 1 x !wasm.funcref attributes { wasm.export_name = \"table\" };
         wasm.tag private @e : (i32) attributes { wasm.export_name = \"tag\" };
         wasm.tag private @unexported : ();",
    );
    let names = |name: &str| {
        export_names(&symbol(&module, name).borrow())
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(names("f"), ["f"]);
    assert_eq!(names("m"), ["memory", "mem"]);
    assert_eq!(names("g"), ["g"]);
    assert_eq!(names("t"), ["table"]);
    assert_eq!(names("e"), ["tag"]);
    assert!(names("unexported").is_empty());
}

#[test]
fn an_import_module_is_a_non_empty_string() {
    let shape = "`wasm.import_module` must be a non-empty string, the name of the module an \
                 import comes from, but this is";
    rejected(
        "wasm.memory public declaration @m : 1 attributes { wasm.import_module = \"\" };",
        &format!("{shape} #builtin.string<\"\">"),
    );
    rejected(
        "wasm.memory public declaration @m : 1 attributes { wasm.import_module = #builtin.i32<1> \
         };",
        &format!("{shape} #builtin.i32<1>"),
    );
    rejected(
        "wasm.memory public declaration @m : 1 attributes { wasm.import_module = [\"a\"] };",
        shape,
    );
}

#[test]
fn an_import_module_on_a_definition_says_an_import_is_a_declaration() {
    rejected(
        "wasm.memory public @m : 1 attributes { wasm.import_module = \"env\" };",
        "`wasm.import_module` marks an import, and an import is a declaration, but `wasm.memory \
         @m` is a definition",
    );
    rejected(
        "builtin.function public extern(\"C\") @f() {
            builtin.ret;
         } attributes { wasm.import_module = \"env\" };",
        "an import is a declaration, but `builtin.function @f` is a definition",
    );
    rejected(
        "wasm.global private @g : i32 { %v = wasm.const 0 : i32; } attributes { \
         wasm.import_module = \"env\" };",
        "an import is a declaration, but `wasm.global @g` is a definition",
    );
    rejected(
        "wasm.tag private @t : () attributes { wasm.import_module = \"env\" };",
        "an import is a declaration, but `wasm.tag @t` is a definition",
    );
    rejected(
        "wasm.table private @t : 1 x !wasm.funcref attributes { wasm.import_module = \"env\" };",
        "an import is a declaration, but `wasm.table @t` is a definition",
    );
}

#[test]
fn an_import_module_on_anything_else_is_rejected() {
    rejected(
        "builtin.global_variable public @g : i32 attributes { wasm.import_module = \"env\" };",
        "`wasm.import_module` marks an import, but `builtin.global_variable @g` is not one: data \
         in linear memory is not something Wasm imports",
    );
    rejected(
        "builtin.module public @inner attributes { wasm.import_module = \"env\" } {};",
        "`wasm.import_module` marks an import, but `builtin.module @inner` is not one: Wasm \
         imports functions, globals, memories, tables and tags",
    );
    rejected(
        "wasm.table private @t : 1 x !wasm.funcref;
         builtin.function public extern(\"C\") @f();
         wasm.elem private @e : @t at 0 [@f] attributes { wasm.import_module = \"env\" };",
        "`wasm.import_module` marks an import, but `wasm.elem @e` is not one",
    );
    // An operation that is not a symbol
    let error = module_error(
        "builtin.function public extern(\"C\") @f() {
            %x = wasm.const 1 : i32 attributes { wasm.import_module = \"env\" };
            builtin.ret;
         };",
    );
    assert!(
        error.contains("`wasm.import_module` marks an import, but `wasm.const` is not one"),
        "{error}"
    );
}

#[test]
fn an_export_name_is_a_string_or_an_array_of_distinct_non_empty_strings() {
    let shape = "`wasm.export_name` must be a non-empty string or an array of distinct non-empty \
                 strings, but";
    for (value, problem) in [
        ("\"\"", "this is an empty string"),
        ("[\"a\", \"\"]", "entry 1 is an empty string"),
        ("[\"a\", \"b\", \"a\"]", "`a` is exported twice"),
        ("#builtin.i32<1>", "this is #builtin.i32<1>"),
        ("[\"a\", #builtin.i32<1>]", "this is"),
    ] {
        rejected(
            &format!("wasm.memory public @m : 1 attributes {{ wasm.export_name = {value} }};"),
            &format!("{shape} {problem}"),
        );
    }
    module_of("wasm.memory public @m : 1 attributes { wasm.export_name = [] };");
}

#[test]
fn an_export_name_on_a_declaration_or_on_anything_else_is_rejected() {
    rejected(
        "wasm.memory public declaration @m : 1 attributes { wasm.export_name = \"m\" };",
        "`wasm.export_name` marks an export, and an export is a definition, but `wasm.memory @m` \
         is a declaration (an import)",
    );
    rejected(
        "builtin.function public extern(\"C\") @f() attributes { wasm.export_name = \"f\" };",
        "an export is a definition, but `builtin.function @f` is a declaration (an import)",
    );
    rejected(
        "wasm.global private @g : i32 attributes { wasm.export_name = \"g\" };",
        "an export is a definition, but `wasm.global @g` is a declaration (an import)",
    );
    rejected(
        "builtin.global_variable public @g : i32 attributes { wasm.export_name = \"g\" };",
        "`wasm.export_name` marks an export, but `builtin.global_variable @g` is not one: Wasm \
         exports functions, globals, memories, tables and tags",
    );
    rejected(
        "builtin.module public @inner attributes { wasm.export_name = \"g\" } {};",
        "`wasm.export_name` marks an export, but `builtin.module @inner` is not one",
    );
}

#[test]
fn another_key_of_the_dialect_is_rejected_and_names_the_ones_it_knows() {
    rejected(
        "wasm.memory public @m : 1 attributes { wasm.nonsense = \"x\" };",
        "`wasm.nonsense` is not an attribute of the `wasm` dialect: it knows `wasm.import_module` \
         and `wasm.export_name`",
    );
    rejected(
        "builtin.function public extern(\"C\") @f() attributes { wasm.import = \"x\" };",
        "`wasm.import` is not an attribute of the `wasm` dialect",
    );
    // A key of another dialect, or with no dialect, is not the dialect's business
    module_of(
        "wasm.memory public @m : 1 attributes { link_name = \"m\", other.key = \"x\", \
         import_module = \"y\" };",
    );
}

#[test]
fn the_entries_are_checked_in_the_generic_form_too() {
    let memory = |declaration: bool, entry: &str| {
        module_text(&format!(
            "\"wasm.memory\"() <{{ name = #builtin.ident<@m>, visibility = \
             #builtin.visibility<public>, limits = #wasm.limits<1>, declaration = \
             #builtin.bool<{declaration}> }}> {{ {entry} }} : () -> ();"
        ))
    };
    parse_text(&memory(true, "wasm.import_module = \"env\"")).unwrap();
    parse_text(&memory(false, "wasm.export_name = \"m\"")).unwrap();
    let Err(error) = parse_text(&memory(false, "wasm.import_module = \"env\"")) else {
        panic!("a definition with an import module is accepted");
    };
    assert!(
        error.contains(
            "`wasm.import_module` marks an import, and an import is a declaration, but \
             `wasm.memory @m` is a definition"
        ),
        "{error}"
    );
    let Err(error) = parse_text(&memory(true, "wasm.export_name = \"m\"")) else {
        panic!("a declaration with an export name is accepted");
    };
    assert!(error.contains("an export is a definition"), "{error}");
}

#[test]
fn a_function_is_found_among_the_symbols_of_the_module_by_name() {
    let (_context, module) = module_of(EVERY_OPERATION);
    assert!(symbol(&module, "f").borrow().is::<Function>());
    assert!(module.borrow().as_operation().implements::<dyn SymbolTable>());
}

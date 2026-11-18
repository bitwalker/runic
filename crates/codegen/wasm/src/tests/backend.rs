//! The backend `wasm` through the compile harness, as a compiler built on `runic-compile` drives
//! it: the data layout of its target, its checkpoints and outputs, its option, its pipeline by
//! optimization level, and how a failure of the emitter fails a compilation.

use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};

use runic_compile::{
    Backend, Compilation, CompilationRequest, Compiler, DebugInfo, EmitRequest, EmitTo, Emitted,
    Input, IrFrontend, OptLevel,
};
use runic_ir::{
    BuilderExt, DataLayout, Op, SourceSpan, dialects::builtin::World, set_data_layout_spec,
    testing::text::describe,
};
use wasmparser::{Parser, Payload};

use crate::{
    WasmTarget,
    backend::{TRIPLE, WasmBackend, WasmOptions},
    emit::validate::validate,
    tests::{
        eval::{Outcome, Val},
        exec::Instance,
    },
    text::print,
};

/// A function that returns its parameter, through a slot of a frame on the shadow stack, so that
/// the module has a stack pointer.
const FRAME: &str = r#"builtin.module public @demo {
    builtin.function public extern("C") @keep(%x: i32) -> i32 {
        %p = mem.alloca i32;
        mem.store %x, %p;
        %y = mem.load %p;
        builtin.ret %y : (i32);
    } attributes { wasm.export_name = "keep" };
};"#;

/// A loop in CFG form that sums `0..n`, which `cfg-to-scf` structures with flags.
const COUNT: &str = r#"builtin.module public @demo {
    builtin.function public extern("C") @count(%n: i32) -> i32 {
        %zero = arith.constant 0 : i32;
        cf.br ^head(%zero, %zero : i32, i32);
    ^head(%i: i32, %acc: i32):
        %done = arith.cmpi uge %i, %n;
        cf.cond_br %done ^exit, ^body : (i1);
    ^body:
        %acc2 = arith.add %acc, %i;
        %one = arith.constant 1 : i32;
        %i2 = arith.add %i, %one;
        cf.br ^head(%i2, %acc2 : i32, i32);
    ^exit:
        builtin.ret %acc : (i32);
    } attributes { wasm.export_name = "count" };
};"#;

/// A function that computes one sum twice and a product it does not use.
const REPEATED: &str = r#"builtin.module public @demo {
    builtin.function public extern("C") @f(%x: i32) -> i32 {
        %k = arith.constant 7 : i32;
        %a = arith.add %x, %k;
        %b = arith.add %x, %k;
        %unused = arith.mul %x, %x;
        %r = arith.add %a, %b;
        builtin.ret %r : (i32);
    } attributes { wasm.export_name = "f" };
};"#;

/// A function with a pointer parameter, whose signature the lowering changes, beside a table
/// and an element segment: a complete module holds them, and an object cannot express the
/// segment.
const WITH_ELEM: &str = r#"builtin.module public @demo {
    wasm.table private @tab : 4 x !wasm.funcref;
    builtin.function public extern("C") @callee(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    wasm.elem private @init : @tab at 1 [@callee];
};"#;

/// Two functions exported under one name: the emitter's error.
const DUPLICATE_EXPORTS: &str = r#"builtin.module public @demo {
    builtin.function public extern("C") @a() -> i32 {
        %x = arith.constant 1 : i32;
        builtin.ret %x : (i32);
    } attributes { wasm.export_name = "f" };
    builtin.function public extern("C") @b() -> i32 {
        %x = arith.constant 2 : i32;
        builtin.ret %x : (i32);
    } attributes { wasm.export_name = "f" };
};"#;

fn compiler() -> Compiler {
    Compiler::builder()
        .frontend(IrFrontend)
        .backend(WasmBackend)
        .build()
        .expect("the IR frontend and the Wasm backend form a valid compiler")
}

/// A request to compile the IR `text` for `wasm32-unknown-unknown`, as the unit `demo`, with the
/// outputs at `checkpoints` delivered to memory. Every test asks for at least one, so that no file
/// is written.
fn request(text: &str, checkpoints: &[&str]) -> CompilationRequest {
    let mut request = CompilationRequest::new();
    request.name = Some("demo".into());
    request.inputs = vec![Input::memory("demo.runic", text)];
    request.target = Some(TRIPLE.parse().unwrap());
    for checkpoint in checkpoints {
        request.emit.push(EmitRequest {
            checkpoint: (*checkpoint).into(),
            to: EmitTo::Memory,
        });
    }
    request
}

fn compile(request: CompilationRequest) -> Compilation {
    compiler().compile(request, &mut ())
}

#[track_caller]
fn succeeded(compilation: &Compilation) {
    assert!(
        compilation.succeeded(),
        "{}",
        compilation.error().map(describe).unwrap_or_default()
    );
}

/// The bytes delivered to memory for `checkpoint`.
#[track_caller]
fn memory(compilation: &Compilation, checkpoint: &str) -> Vec<u8> {
    compilation
        .emitted()
        .iter()
        .find(|emitted| {
            matches!(emitted, Emitted::Memory { .. }) && emitted.checkpoint().as_str() == checkpoint
        })
        .unwrap_or_else(|| panic!("nothing was delivered to memory for '{checkpoint}'"))
        .bytes()
        .unwrap()
        .to_vec()
}

/// The binary module `request` compiles to, delivered at `wasm.module`.
#[track_caller]
fn module_of(request: CompilationRequest) -> Vec<u8> {
    let compilation = compile(request);
    succeeded(&compilation);
    memory(&compilation, "wasm.module")
}

/// The error of the failed `compilation`, described with its labels.
#[track_caller]
fn error_of(compilation: &Compilation) -> String {
    describe(compilation.error().expect("the compilation failed"))
}

/// The names of the custom sections of `bytes`, in order.
fn custom_sections(bytes: &[u8]) -> Vec<String> {
    Parser::new(0)
        .parse_all(bytes)
        .filter_map(|payload| match payload.unwrap() {
            Payload::CustomSection(reader) => Some(reader.name().to_string()),
            _ => None,
        })
        .collect()
}

/// The number of instructions in the body of each function of `bytes`, `end` included.
fn instruction_counts(bytes: &[u8]) -> Vec<usize> {
    Parser::new(0)
        .parse_all(bytes)
        .filter_map(|payload| match payload.unwrap() {
            Payload::CodeSectionEntry(body) => {
                let mut reader = body.get_operators_reader().unwrap();
                let mut count = 0;
                while !reader.eof() {
                    reader.read().unwrap();
                    count += 1;
                }
                Some(count)
            }
            _ => None,
        })
        .collect()
}

#[test]
fn the_data_layout_of_the_target_is_the_c_abi_of_wasm32_and_the_lowering_accepts_it() {
    let backend = WasmBackend;
    let triple = TRIPLE.parse().unwrap();
    let spec = backend.data_layout(&triple).unwrap();
    let context = crate::tests::context();
    let cx = &*context;
    let layout = DataLayout::new(cx, &spec);
    let table = [
        (cx.i8(), 1, 1),
        (cx.i16(), 2, 2),
        (cx.i32(), 4, 4),
        (cx.i64(), 8, 8),
        (cx.i128(), 16, 16),
        (cx.f32(), 4, 4),
        (cx.f64(), 8, 8),
        (cx.isize(), 4, 4),
        (cx.opaque_ptr_type(), 4, 4),
        (cx.ptr_type(cx.i64()), 4, 4),
    ];
    for (ty, size, align) in table {
        assert_eq!(layout.size_of(ty), Ok(size), "the size of {ty}");
        assert_eq!(
            layout.align_of(ty).map(|align| align.bytes()),
            Ok(align),
            "the alignment of {ty}"
        );
    }
    assert_eq!(layout.endianness(), runic_ir::Endianness::Little);

    let world = context.clone().builder().create::<World, ()>(SourceSpan::UNKNOWN)().unwrap();
    set_data_layout_spec(&mut world.as_operation_ref().borrow_mut(), spec.clone());
    let target = WasmTarget::of(world.borrow().as_operation())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    assert_eq!(target.layout().spec(), &spec);
}

#[test]
fn the_backend_supports_exactly_wasm32_unknown_unknown() {
    let backend = WasmBackend;
    assert_eq!(backend.name(), "wasm");
    let targets: Vec<String> = backend.targets().iter().map(ToString::to_string).collect();
    assert_eq!(targets, ["wasm32-unknown-unknown"]);
    assert!(backend.supports(&TRIPLE.parse().unwrap()));
    for other in ["wasm32-wasip1", "wasm32-wasip2", "wasm64-unknown-unknown", "wasm32-unknown"] {
        assert!(!backend.supports(&other.parse().unwrap()), "{other}");
    }
    let checkpoints: Vec<_> = backend
        .checkpoints()
        .iter()
        .map(|decl| (decl.id.as_str(), decl.aliases, decl.extension))
        .collect();
    assert_eq!(
        checkpoints,
        [
            ("wasm.object", &["object"][..], "o"),
            ("wasm.wat", &["wat"][..], "wat"),
            ("wasm.module", &["module"][..], "core.wasm")
        ]
    );
}

#[test]
fn a_compilation_to_the_default_stop_point_gives_a_binary_module_that_validates() {
    let compilation = compile(request(FRAME, &["module"]));
    succeeded(&compilation);
    let reached: Vec<_> = compilation.reached().iter().map(|id| id.as_str()).collect();
    assert_eq!(
        reached,
        ["ir.translated", "ir.optimized", "ir.lowered", "wasm.wat", "wasm.module"]
    );
    let [Emitted::Memory { checkpoint, bytes }] = compilation.emitted() else {
        panic!("one output, to memory: {:?}", compilation.emitted());
    };
    assert_eq!(checkpoint.as_str(), "wasm.module");
    assert_eq!(&bytes[..8], b"\0asm\x01\0\0\0");
    validate(bytes).unwrap_or_else(|report| panic!("{}", describe(&report)));
    let mut instance = Instance::new(bytes);
    assert_eq!(instance.call("keep", &[Val::i32(7)]), Outcome::Values(vec![Val::i32(7)]));
}

#[test]
fn stopping_at_wat_gives_the_text_of_the_module() {
    let mut request = request(FRAME, &["wat"]);
    request.stop_after = Some("wat".into());
    let compilation = compile(request);
    succeeded(&compilation);
    let [Emitted::Memory { checkpoint, bytes }] = compilation.emitted() else {
        panic!("one output, to memory: {:?}", compilation.emitted());
    };
    assert_eq!(checkpoint.as_str(), "wasm.wat");
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(text.starts_with("(module $demo\n"), "{text}");
    assert!(text.contains("(export \"keep\" (func $demo::keep))"), "{text}");
    assert!(text.contains("(export \"memory\" (memory $__linear_memory))"), "{text}");
}

#[test]
fn emitting_both_checkpoints_in_one_run_gives_one_module_as_text_and_as_binary() {
    let compilation = compile(request(FRAME, &["wat", "module"]));
    succeeded(&compilation);
    let text = String::from_utf8(memory(&compilation, "wasm.wat")).unwrap();
    let binary = memory(&compilation, "wasm.module");
    validate(&binary).unwrap();
    assert_eq!(text, print(&binary).unwrap());
    // A second compilation gives the same bytes: the text is that of the binary delivered.
    assert_eq!(binary, module_of(request(FRAME, &["module"])));
}

#[test]
fn the_stack_size_option_is_the_initial_value_of_the_stack_pointer() {
    let stack_pointer = |options: Option<WasmOptions>| {
        let mut request = request(FRAME, &["wat", "module"]);
        if let Some(options) = options {
            request.set_options(options);
        }
        let compilation = compile(request);
        succeeded(&compilation);
        let text = String::from_utf8(memory(&compilation, "wasm.wat")).unwrap();
        let line = text
            .lines()
            .find(|line| line.contains("(global $__stack_pointer"))
            .unwrap_or_else(|| panic!("no stack pointer: {text}"))
            .trim()
            .to_string();
        (line, memory(&compilation, "wasm.module"))
    };
    let (default, _) = stack_pointer(None);
    assert_eq!(default, "(global $__stack_pointer (;0;) (mut i32) i32.const 1048576)");
    let (small, bytes) = stack_pointer(Some(WasmOptions { stack_size: 65_536 }));
    assert_eq!(small, "(global $__stack_pointer (;0;) (mut i32) i32.const 65536)");
    let mut instance = Instance::new(&bytes);
    assert_eq!(
        instance.call("keep", &[Val::i32(-3i32 as u32)]),
        Outcome::Values(vec![Val::i32(-3i32 as u32)])
    );
}

#[test]
fn a_stack_size_that_is_not_a_multiple_of_16_is_the_emitters_error() {
    let mut request = request(FRAME, &["module"]);
    request.set_options(WasmOptions { stack_size: 100 });
    let compilation = compile(request);
    assert!(!compilation.succeeded());
    let error = error_of(&compilation);
    assert!(
        error.contains(
            "the stack size 100 (`--wasm-stack-size`) is not a multiple of 16, the alignment of \
             the stack"
        ),
        "{error}"
    );
    assert!(error.contains("the stack this stack pointer points into is laid out with that size"));
    let reached: Vec<_> = compilation.reached().iter().map(|id| id.as_str()).collect();
    assert_eq!(reached, ["ir.translated", "ir.optimized", "ir.lowered"]);
    assert!(compilation.emitted().is_empty());
}

#[test]
fn an_error_of_the_emitter_fails_the_compilation_with_its_labels() {
    let compilation = compile(request(DUPLICATE_EXPORTS, &["module"]));
    assert!(!compilation.succeeded());
    let error = error_of(&compilation);
    assert!(error.contains("two exports are named `f`"), "{error}");
    assert!(error.contains("this exports @b as `f`"), "{error}");
    assert!(error.contains("this exports @a as `f` first"), "{error}");
    assert!(compilation.emitted().is_empty());
}

#[test]
fn no_debug_information_leaves_out_the_name_section() {
    let named = module_of(request(FRAME, &["module"]));
    assert_eq!(custom_sections(&named), ["name", "producers"]);
    let mut stripped = request(FRAME, &["module"]);
    stripped.debug = DebugInfo::None;
    let stripped = module_of(stripped);
    assert_eq!(custom_sections(&stripped), ["producers"]);
    let text = print(&stripped).unwrap();
    assert!(!text.contains('$'), "{text}");
    // Line and full debug information both keep it.
    let mut full = request(FRAME, &["module"]);
    full.debug = DebugInfo::Full;
    assert_eq!(custom_sections(&module_of(full)), ["name", "producers"]);
}

#[test]
fn an_optimization_level_above_none_removes_dead_and_repeated_values_and_both_run_alike() {
    // With the optimization pipeline of the level, and with an empty one, so that the backend's
    // own `cse` and `dce` after the lowering are seen alone too.
    let compiled = |text: &str, level: OptLevel, pipeline: Option<String>| {
        let mut request = request(text, &["module"]);
        request.opt_level = level;
        request.pipeline = pipeline;
        module_of(request)
    };
    let levels = [
        OptLevel::Basic,
        OptLevel::Balanced,
        OptLevel::Max,
        OptLevel::Size,
        OptLevel::SizeMin,
    ];
    let none = compiled(REPEATED, OptLevel::None, None);
    let [count] = instruction_counts(&none)[..] else {
        panic!("one function")
    };
    for level in levels {
        for pipeline in [None, Some(String::new())] {
            let optimized = compiled(REPEATED, level, pipeline.clone());
            let [optimized_count] = instruction_counts(&optimized)[..] else {
                panic!("one function")
            };
            assert!(
                optimized_count < count,
                "-O{level}, pipeline {pipeline:?}: {optimized_count} instructions, and {count} at \
                 -O0\n{}\n{}",
                print(&optimized).unwrap(),
                print(&none).unwrap()
            );
            for x in [0, 5, u32::MAX] {
                let expected = Outcome::Values(vec![Val::i32(x.wrapping_add(7).wrapping_mul(2))]);
                assert_eq!(Instance::new(&none).call("f", &[Val::i32(x)]), expected, "-O0, {x}");
                assert_eq!(
                    Instance::new(&optimized).call("f", &[Val::i32(x)]),
                    expected,
                    "-O{level}, {x}"
                );
            }
        }
        // The loop of `cfg-to-scf` runs alike at every level.
        let none = compiled(COUNT, OptLevel::None, None);
        let optimized = compiled(COUNT, level, None);
        for n in [0, 1, 10, 100] {
            let expected = Outcome::Values(vec![Val::i32((0..n).sum())]);
            assert_eq!(Instance::new(&none).call("count", &[Val::i32(n)]), expected, "-O0, {n}");
            assert_eq!(
                Instance::new(&optimized).call("count", &[Val::i32(n)]),
                expected,
                "-O{level}, {n}"
            );
        }
    }
}

#[test]
fn the_passes_of_the_backend_depend_on_the_optimization_level() {
    let pipeline = |level: OptLevel| {
        let mut request = request(COUNT, &["ir.lowered"]);
        request.opt_level = level;
        request.pipeline = Some(String::new());
        request.ir.dump_pass_pipeline = true;
        request.stop_after = Some("ir.lowered".into());
        let compilation = compile(request);
        succeeded(&compilation);
        compilation.pass_output().to_string()
    };
    let none = pipeline(OptLevel::None);
    assert!(
        none.contains(
            "builtin.world(cfg-to-scf{exits=\"in-place\"},wasm-lower-frames,convert-to-wasm,\
             wasm-verify-form)"
        ),
        "{none}"
    );
    let balanced = pipeline(OptLevel::Balanced);
    assert!(
        balanced.contains(
            "builtin.world(cfg-to-scf{exits=\"in-place\"},wasm-lower-frames,convert-to-wasm,cse,\
             dce,wasm-verify-form)"
        ),
        "{balanced}"
    );
}

#[test]
fn a_target_the_backend_does_not_support_is_refused_by_the_harness() {
    let mut request = request(FRAME, &["module"]);
    request.target = Some("wasm32-wasip2".parse().unwrap());
    let compilation = compile(request);
    let error = compilation.error().expect("the compilation failed").to_string();
    assert_eq!(
        error,
        "no backend supports the target 'wasm32-wasip2'; the targets are 'wasm32-unknown-unknown'"
    );
}

/// The sections of `bytes`, read as a module: the names of its custom sections, and whether it
/// has a `linking` section.
#[track_caller]
fn assert_object(bytes: &[u8]) {
    assert_eq!(&bytes[..8], b"\0asm\x01\0\0\0");
    assert!(
        custom_sections(bytes).iter().any(|name| name == "linking"),
        "no linking section"
    );
}

#[test]
fn stopping_at_object_gives_a_relocatable_object_and_runs_no_further() {
    let mut request = request(FRAME, &["object"]);
    request.stop_after = Some("object".into());
    let compilation = compile(request);
    succeeded(&compilation);
    let reached: Vec<_> = compilation.reached().iter().map(|id| id.as_str()).collect();
    assert_eq!(reached, ["ir.translated", "ir.optimized", "ir.lowered", "wasm.object"]);
    let [Emitted::Memory { checkpoint, bytes }] = compilation.emitted() else {
        panic!("one output, to memory: {:?}", compilation.emitted());
    };
    assert_eq!(checkpoint.as_str(), "wasm.object");
    assert_object(bytes);
    assert!(Parser::new(0).parse_all(bytes).all(|payload| payload.is_ok()));
}

#[test]
fn the_default_stop_point_writes_no_object() {
    let compilation = compile(request(FRAME, &["module"]));
    succeeded(&compilation);
    let reached: Vec<_> = compilation.reached().iter().map(|id| id.as_str()).collect();
    assert!(!reached.contains(&"wasm.object"), "{reached:?}");
    let module = memory(&compilation, "wasm.module");
    assert!(!custom_sections(&module).iter().any(|name| name == "linking"));
}

#[test]
fn an_object_beside_the_module_leaves_the_module_as_it_is() {
    let compilation = compile(request(FRAME, &["object", "module"]));
    succeeded(&compilation);
    assert_object(&memory(&compilation, "wasm.object"));
    let module = memory(&compilation, "wasm.module");
    validate(&module).unwrap();
    assert_eq!(module, module_of(request(FRAME, &["module"])));
}

#[test]
fn the_stack_size_option_does_not_change_an_object() {
    let object = |stack_size: Option<u32>| {
        let mut request = request(FRAME, &["object"]);
        request.stop_after = Some("object".into());
        if let Some(stack_size) = stack_size {
            request.set_options(WasmOptions { stack_size });
        }
        let compilation = compile(request);
        succeeded(&compilation);
        memory(&compilation, "wasm.object")
    };
    assert_eq!(object(None), object(Some(4096)));
}

#[test]
fn a_unit_with_an_element_segment_is_a_module_and_is_refused_as_an_object() {
    let module = module_of(request(WITH_ELEM, &["module"]));
    validate(&module).unwrap();

    let compilation = compile(request(WITH_ELEM, &["module"]));
    succeeded(&compilation);

    let mut stopping = request(WITH_ELEM, &["object"]);
    stopping.stop_after = Some("object".into());
    let compilation = compile(stopping);
    assert!(!compilation.succeeded());
    let error = error_of(&compilation);
    assert!(error.contains("a relocatable object cannot express"), "{error}");
    assert!(error.contains("wasm.elem"), "{error}");
    assert!(
        error.contains("the linker ignores the element segments of an object"),
        "{error}"
    );
    assert!(compilation.emitted().is_empty());

    // Beside the module, the refusal fails the compilation too: the object was asked for.
    let compilation = compile(request(WITH_ELEM, &["object", "module"]));
    assert!(!compilation.succeeded());
}

#[test]
fn the_alias_object_names_the_checkpoint() {
    let by_alias = compile(request(FRAME, &["object"]));
    succeeded(&by_alias);
    let by_id = compile(request(FRAME, &["wasm.object"]));
    succeeded(&by_id);
    assert_object(&memory(&by_alias, "wasm.object"));
    assert_eq!(memory(&by_alias, "wasm.object"), memory(&by_id, "wasm.object"));
}

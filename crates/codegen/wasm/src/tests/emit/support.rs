//! What the tests of the emitter share: building descriptions by hand, writing, validating,
//! printing and reading modules back.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use wasmparser::{Parser, Payload};

use crate::{
    emit::{
        desc::{
            CodeWriter, Entity, FuncId, FuncType, Function, FunctionBody, Import, ModuleDesc,
            ValType,
        },
        sections::PRODUCER_VERSION,
        validate::validate,
        write::{Resolved, write_module},
    },
    text::print,
};

/// Adds to `desc` a function `name` of type `params -> results`, with the locals `locals` and the
/// body that `build` writes; `end` is appended after it.
pub fn define(
    desc: &mut ModuleDesc,
    name: &str,
    ty: (&[ValType], &[ValType]),
    locals: &[ValType],
    build: impl FnOnce(&mut CodeWriter<'_>),
) -> FuncId {
    let id = declare(desc, name, ty);
    set_body(desc, id, locals, build);
    id
}

/// Adds to `desc` a function `name` of type `params -> results` with an empty body (`end`), to be
/// replaced by [set_body].
pub fn declare(desc: &mut ModuleDesc, name: &str, ty: (&[ValType], &[ValType])) -> FuncId {
    let ty = desc.intern_type(FuncType::new(ty.0.iter().copied(), ty.1.iter().copied()));
    desc.add_function(Function {
        ty,
        name: name.to_string(),
        entity: Entity::Definition(FunctionBody {
            code: Vec::from([0x0b]),
            ..FunctionBody::default()
        }),
    })
}

/// Gives the function `id` of `desc` the locals `locals` and the body that `build` writes, then
/// `end`.
pub fn set_body(
    desc: &mut ModuleDesc,
    id: FuncId,
    locals: &[ValType],
    build: impl FnOnce(&mut CodeWriter<'_>),
) {
    let mut body = FunctionBody {
        locals: locals.to_vec(),
        ..FunctionBody::default()
    };
    let mut writer = body.writer();
    build(&mut writer);
    writer.sink().end();
    desc.function_mut(id).entity = Entity::Definition(body);
}

/// Adds to `desc` an imported function `name` of type `params -> results`, from `env`.
pub fn import(desc: &mut ModuleDesc, name: &str, ty: (&[ValType], &[ValType])) -> FuncId {
    let ty = desc.intern_type(FuncType::new(ty.0.iter().copied(), ty.1.iter().copied()));
    desc.add_function(Function {
        ty,
        name: name.to_string(),
        entity: Entity::Import(Import::new("env", name)),
    })
}

/// `desc` written with `resolved` and validated; a failure of either panics with its message.
#[track_caller]
pub fn written(desc: &ModuleDesc, resolved: &Resolved) -> Vec<u8> {
    let bytes = write_module(desc, resolved).unwrap_or_else(|error| panic!("{error}"));
    validate(&bytes).unwrap_or_else(|report| panic!("{report}"));
    bytes
}

/// The text of `desc` written with `resolved` and validated.
#[track_caller]
pub fn text(desc: &ModuleDesc, resolved: &Resolved) -> String {
    print(&written(desc, resolved)).unwrap_or_else(|report| panic!("{report}"))
}

/// The `producers` section as `wasmprinter` prints it, indented as in a module.
pub fn producers_text() -> String {
    alloc::format!("  (@producers\n    (processed-by \"runic\" \"{PRODUCER_VERSION}\")\n  )\n")
}

/// The payloads of the module `bytes`, which must parse.
#[track_caller]
pub fn payloads(bytes: &[u8]) -> Vec<Payload<'_>> {
    Parser::new(0)
        .parse_all(bytes)
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|error| panic!("{error}"))
}

/// The ids of the sections of the module `bytes`, in order, with the name of each custom section.
pub fn section_ids(bytes: &[u8]) -> Vec<(u8, String)> {
    payloads(bytes)
        .iter()
        .filter_map(|payload| {
            let (id, _) = payload.as_section()?;
            let name = match payload {
                Payload::CustomSection(custom) => custom.name().to_string(),
                _ => String::new(),
            };
            Some((id, name))
        })
        .collect()
}

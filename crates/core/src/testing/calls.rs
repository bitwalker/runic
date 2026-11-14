//! A corpus of modules that name symbols in every way the operations of the workspace can, the
//! mutations of it that leave one of those names dangling, and random declared effects: used by
//! the tests of calls and symbol references in this crate, `runic-analysis` and
//! `runic-transform`.

use alloc::{format, string::String, vec::Vec};

use crate::{
    dialects::builtin::attributes::{DeclaredEffects, MemoryEffectDescriptor},
    effects::MemoryEffect,
    testing::text::Rng,
};

/// Modules that parse and verify. Symbol names are unique across each source, so that a name
/// that is deleted or renamed cannot be found in another scope.
pub const CORPUS: [&str; 6] = [
    // direct calls, through an alias chain; a reference; an indirect call
    r#"
builtin.module public @test {
    builtin.function private extern("C") @target(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function_alias private @first -> @target;
    builtin.function_alias private @alias -> @first;
    builtin.function public extern("C") @declared(i32) -> i32 effects(read);
    builtin.function public extern("C") @caller(%a: i32) -> i32 {
        %r = builtin.call @alias(%a) : extern("C") (i32) -> i32;
        %s = builtin.call @declared(%r) : extern("C") (i32) -> i32;
        %f = builtin.function_ref @first : (i32) -> i32;
        %t = builtin.call_indirect %f(%s) : extern("C") (i32) -> i32;
        %u = builtin.call @target(%t) : extern("C") (i32) -> i32;
        builtin.ret %u : (i32);
    };
};"#,
    // a call with two possible callees
    r#"
builtin.module public @m {
    builtin.function private extern("C") @left(i32) -> i32;
    builtin.function private extern("C") @right(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function_alias private @other -> @left;
    builtin.function public extern("C") @chooser(%index: i32, %a: i32) -> i32 {
        %r = test.call_one_of %index, @left, @right(%a) : extern("C") (i32) -> i32;
        %s = test.call_one_of %index, @other, @right(%r) : extern("C") (i32) -> i32;
        builtin.ret %s : (i32);
    };
};"#,
    // a global and a reference to it, a symbol in an attribute dictionary
    r#"
builtin.module public @globals {
    builtin.global_variable public @var : i8;
    builtin.function public extern("C") @sink(i8);
    builtin.function public extern("C") @reader() -> i8 {
        %ptr = builtin.global_symbol @var : ptr<i8>;
        %v = test.load %ptr;
        builtin.call @sink(%v) : extern("C") (i8) attributes { note = #builtin.symbol<@var> };
        builtin.ret %v : (i8);
    };
};"#,
    // absolute paths across modules of a world
    r#"
builtin.world {
    builtin.module public @lib {
        builtin.function public extern("C") @exported(i32) -> i32 effects();
        builtin.function_alias public @reexported -> @exported;
    };
    builtin.module public @app {
        builtin.function public extern("C") @main(%a: i32) -> i32 {
            %r = builtin.call ::@lib::@exported(%a) : extern("C") (i32) -> i32;
            %f = builtin.function_ref ::@lib::@reexported : (i32) -> i32;
            %s = builtin.call_indirect %f(%r) : extern("C") (i32) -> i32;
            builtin.ret %s : (i32);
        };
    };
};"#,
    // calls inside regions and after branches
    r#"
builtin.module public @flow {
    builtin.function private extern("C") @pure(i32) -> i32 effects();
    builtin.function private extern("C") @effectful(i32) -> i32;
    builtin.function private extern("C") @body(%x: i32) -> i32 {
        %one = arith.constant 1 : i32;
        %y = arith.add %x, %one;
        builtin.ret %y : (i32);
    };
    builtin.function public extern("C") @entry(%c: i1, %a: i32) -> i32 {
        %k = arith.constant 42 : i32;
        %x = scf.if %c then {
            %t = builtin.call @pure(%k) : extern("C") (i32) -> i32;
            scf.yield %t : (i32);
        } else {
            %e = builtin.call @effectful(%a) : extern("C") (i32) -> i32;
            scf.yield %e : (i32);
        } : (i1) -> (i32);
        %b = builtin.call @body(%k) : extern("C") (i32) -> i32;
        cf.cond_br %c ^t, ^f : (i1);
    ^t:
        %p = builtin.call @body(%x) : extern("C") (i32) -> i32;
        builtin.ret %p : (i32);
    ^f:
        builtin.ret %b : (i32);
    };
};"#,
    // an alias of an alias of an alias, called and referenced at each level
    r#"
builtin.module public @chain {
    builtin.function private extern("C") @end(%x: i32) -> i32 effects(read) {
        builtin.ret %x : (i32);
    };
    builtin.function_alias private @one -> @end;
    builtin.function_alias private @two -> @one;
    builtin.function_alias private @three -> @two;
    builtin.function public extern("C") @user(%a: i32) -> i32 {
        %r = builtin.call @three(%a) : extern("C") (i32) -> i32;
        %s = builtin.call @two(%r) : extern("C") (i32) -> i32;
        %f = builtin.function_ref @one : (i32) -> i32;
        %g = builtin.function_ref @three : (i32) -> i32;
        %t = builtin.call_indirect %f(%s) : extern("C") (i32) -> i32;
        %u = builtin.call_indirect %g(%t) : extern("C") (i32) -> i32;
        builtin.ret %u : (i32);
    };
};"#,
];

/// The symbol name starting at the first `@` of `line` at or after `from`: `(start, end)` of
/// `@name`.
fn symbol_at(line: &str, from: usize) -> Option<(usize, usize)> {
    let start = from + line[from..].find('@')?;
    let len = line[start + 1..]
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(line.len() - start - 1);
    Some((start, start + 1 + len))
}

/// True if `line` names the symbol `name` (written with its `@`).
pub fn names_symbol(line: &str, name: &str) -> bool {
    let mut from = 0;
    while let Some((start, end)) = symbol_at(line, from) {
        if &line[start..end] == name {
            return true;
        }
        from = end;
    }
    false
}

/// Every way to leave one symbol name of `source` dangling: delete a one-line declaration, rename
/// a declaration, or rename one reference. Each is `(what, text, must fail)`; a mutation must
/// fail to verify when the name it removes is still named somewhere.
pub fn mutations(source: &str) -> Vec<(String, String, bool)> {
    const DECLARING: [&str; 5] = [
        "builtin.function ",
        "builtin.function_alias ",
        "builtin.global_variable ",
        "builtin.module ",
        "builtin.component ",
    ];
    let lines = source.lines().collect::<Vec<_>>();
    let join = |lines: &[String]| lines.join("\n");
    let mut out = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let declares = DECLARING.iter().any(|keyword| line.trim_start().starts_with(keyword));
        let mut from = 0;
        let mut first = true;
        while let Some((start, end)) = symbol_at(line, from) {
            let name = &line[start..end];
            let renamed = format!("{}{name}_gone{}", &line[..start], &line[end..]);
            let mut mutated = lines.iter().map(|l| String::from(*l)).collect::<Vec<_>>();
            mutated[index] = renamed;
            if declares && first {
                let referenced = lines.iter().enumerate().any(|(other, l)| {
                    (other != index && names_symbol(l, name))
                        || (other == index && names_symbol(&l[end..], name))
                });
                out.push((format!("rename the declaration of {name}"), join(&mutated), referenced));
                if line.trim_end().ends_with(';') && !line.contains('{') {
                    let mut deleted = lines.iter().map(|l| String::from(*l)).collect::<Vec<_>>();
                    deleted.remove(index);
                    out.push((
                        format!("delete the declaration of {name}"),
                        join(&deleted),
                        referenced,
                    ));
                }
            } else {
                out.push((
                    format!("rename a reference to {name} on line {index}"),
                    join(&mutated),
                    true,
                ));
            }
            first = false;
            from = end;
        }
    }
    out
}

/// Random declared effects for a callable with `params` parameters and `results` results: unknown,
/// none, or one to four effects, each on no value, an argument or a result.
pub fn random_declared_effects(rng: &mut Rng, params: usize, results: usize) -> DeclaredEffects {
    const EFFECTS: [MemoryEffect; 4] = [
        MemoryEffect::Read,
        MemoryEffect::Write,
        MemoryEffect::Allocate,
        MemoryEffect::Free,
    ];
    match rng.below(4) {
        0 => DeclaredEffects::Unknown,
        1 => DeclaredEffects::none(),
        _ => (0..1 + rng.below(4))
            .map(|_| {
                let effect = rng.pick(&EFFECTS);
                match rng.below(3) {
                    0 if params > 0 => {
                        MemoryEffectDescriptor::on_argument(effect, rng.below(params as u64) as u32)
                    }
                    1 if results > 0 => {
                        MemoryEffectDescriptor::on_result(effect, rng.below(results as u64) as u32)
                    }
                    _ => MemoryEffectDescriptor::new(effect),
                }
            })
            .collect(),
    }
}

/// `effects` as a function writes them: nothing when unknown, else ` effects(..)`.
pub fn declared_effects_text(effects: &DeclaredEffects) -> String {
    match effects {
        DeclaredEffects::Unknown => String::new(),
        known => format!(" effects({known})"),
    }
}

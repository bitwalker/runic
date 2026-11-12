//! The names the IR printer gives recursion-group members.

use alloc::vec::Vec;
use core::{cell::RefCell, fmt};

use compact_str::CompactString;

use super::{RecGroup, Type, TypeFormatter};
use crate::{FxHashMap, FxHashSet, interner::Symbol};

/// The aliases chosen for recursion-group members during one print.
///
/// The first time a member is named, every member of its group is named, and the group is
/// queued. The names are purely textual: the context does not remember them.
///
/// A member is named after the name its body suggests (see
/// [TypeAliasName](super::TypeAliasName)) if that is a bare identifier
/// (`[A-Za-z_][A-Za-z0-9_$]*`), and `t` otherwise. A suggested name is used as it is if no other
/// member has it yet, and otherwise gets the smallest numeric suffix `_<k>` that is free; `t`
/// always gets a suffix, without the underscore: `t0`, `t1`, ...
#[derive(Default)]
pub struct TypeAliases {
    /// The alias of each named member.
    names: FxHashMap<Type, CompactString>,
    /// Every alias handed out so far.
    taken: FxHashSet<CompactString>,
    /// The named groups, in the order they were first named.
    groups: Vec<RecGroup>,
    /// For each base a suffix is added to (`None` for `t`), the smallest suffix that may still be
    /// free: every smaller one is taken.
    next_suffix: FxHashMap<Option<Symbol>, u64>,
}

impl TypeAliases {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The alias of the recursion-group member `member`; the first time a member of its group is
    /// named, every member of the group is named, in index order, and the group is queued.
    ///
    /// Panics if `member` is not a member of a recursion group.
    pub fn alias_of(&mut self, member: Type) -> &str {
        if self.names.contains_key(&member) {
            return &self.names[&member];
        }
        let group = member
            .rec_group()
            .unwrap_or_else(|| panic!("`{member}` is not a member of a recursion group"));
        for member in group.members() {
            let alias = self.fresh_alias(*member);
            self.taken.insert(alias.clone());
            self.names.insert(*member, alias);
        }
        self.groups.push(group);
        &self.names[&member]
    }

    /// The groups named so far, in the order they were first named.
    pub fn groups(&self) -> &[RecGroup] {
        &self.groups
    }

    /// An alias for `member` that no other member has.
    ///
    /// A suffixed alias is the one with the smallest free suffix. Suffixes are tried from the
    /// base's [Self::next_suffix], not from 0, so that naming `n` members costs `O(n)` and not
    /// `O(n²)`; the result is the same, because names are never released.
    fn fresh_alias(&mut self, member: Type) -> CompactString {
        let suggested = member
            .name()
            .alias_name_fn()
            .and_then(|alias_name| alias_name(member))
            .filter(|name| is_alias_identifier(name.as_str()));
        if let Some(name) = suggested
            && !self.taken.contains(name.as_str())
        {
            return CompactString::from(name.as_str());
        }
        let next = self.next_suffix.entry(suggested).or_insert(0);
        let (suffix, alias) = (*next..)
            .map(|k| match suggested {
                Some(name) => (k, compact_str::format_compact!("{name}_{k}")),
                None => (k, compact_str::format_compact!("t{k}")),
            })
            .find(|(_, alias)| !self.taken.contains(alias.as_str()))
            .expect("some suffix is free");
        *next = suffix + 1;
        alias
    }
}

/// Returns true if `text` can be used as an alias name: an ASCII letter or `_`, then ASCII
/// letters, digits, `_` and `$`.
fn is_alias_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$'))
}

/// Writes a type with [TypeFormatter::ty], or its body with [TypeFormatter::definition], with
/// recursion-group members written as their aliases in `aliases`.
pub(crate) struct Aliased<'a> {
    pub ty: Type,
    pub aliases: &'a RefCell<TypeAliases>,
    pub definition: bool,
}

impl fmt::Display for Aliased<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printer = TypeFormatter::with_aliases(f, self.aliases);
        if self.definition {
            printer.definition(self.ty)
        } else {
            printer.ty(self.ty)
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, string::String};
    use core::cell::RefCell;

    use super::*;
    use crate::{Context, StructType};

    fn sym(s: &str) -> Symbol {
        Symbol::intern(s)
    }

    fn show(ty: Type, aliases: &RefCell<TypeAliases>) -> String {
        format!(
            "{}",
            Aliased {
                ty,
                aliases,
                definition: false
            }
        )
    }

    fn define(ty: Type, aliases: &RefCell<TypeAliases>) -> String {
        format!(
            "{}",
            Aliased {
                ty,
                aliases,
                definition: true
            }
        )
    }

    /// A group of one struct holding `ptr<rec.0>` and `extra`, named `name` if given.
    fn singleton(context: &Context, name: Option<&str>, extra: Type) -> Type {
        let fields = [context.ptr_type(context.rec_var(0)).into(), extra.into()];
        let body = match name {
            Some(name) => context.struct_type(StructType::named(sym(name), fields)),
            None => context.struct_type(StructType::new(fields)),
        };
        context.rec_group(&[body]).unwrap()[0]
    }

    /// A group of two structs pointing at each other, named `names` where given.
    fn pair(context: &Context, names: [Option<&str>; 2]) -> [Type; 2] {
        let bodies = [1, 0].map(|other| {
            let fields = [context.ptr_type(context.rec_var(other)).into()];
            match names[1 - other as usize] {
                Some(name) => context.struct_type(StructType::named(sym(name), fields)),
                None => context.struct_type(StructType::new(fields)),
            }
        });
        let members = context.rec_group(&bodies).unwrap();
        [members[0], members[1]]
    }

    #[test]
    fn a_clashing_name_gets_a_suffix() {
        let context = Context::default();
        let first = singleton(&context, Some("list"), context.i32());
        let second = singleton(&context, Some("list"), context.i64());
        let aliases = RefCell::new(TypeAliases::new());
        assert_eq!(show(first, &aliases), "!list");
        assert_eq!(show(context.ptr_type(second), &aliases), "ptr<!list_0>");
        assert_eq!(define(first, &aliases), "struct list<ptr<!list>, i32>");
        assert_eq!(define(second, &aliases), "struct list<ptr<!list_0>, i64>");
        let groups = [first, second].map(|ty| ty.rec_group().unwrap());
        assert_eq!(aliases.borrow().groups(), groups);
    }

    #[test]
    fn unnamed_members_are_numbered_in_index_order() {
        let context = Context::default();
        let [first, second] = pair(&context, [None, None]);
        let aliases = RefCell::new(TypeAliases::new());
        assert_eq!(show(second, &aliases), "!t1");
        assert_eq!(show(first, &aliases), "!t0");
        assert_eq!(define(second, &aliases), "struct<ptr<!t0>>");
    }

    #[test]
    fn a_name_that_is_not_an_identifier_is_not_used() {
        let context = Context::default();
        let quoted = singleton(&context, Some("my type"), context.i32());
        let aliases = RefCell::new(TypeAliases::new());
        assert_eq!(show(quoted, &aliases), "!t0");
        assert_eq!(define(quoted, &aliases), "struct \"my type\"<ptr<!t0>, i32>");
    }

    #[test]
    fn a_name_like_a_numbered_alias_does_not_clash() {
        let context = Context::default();
        let [named, unnamed] = pair(&context, [Some("t0"), None]);
        let aliases = RefCell::new(TypeAliases::new());
        assert_eq!(show(unnamed, &aliases), "!t1");
        assert_eq!(show(named, &aliases), "!t0");

        // The other way round: the unnamed member takes `t0` first.
        let context = Context::default();
        let unnamed = singleton(&context, None, context.i32());
        let named = singleton(&context, Some("t0"), context.i32());
        let aliases = RefCell::new(TypeAliases::new());
        assert_eq!(show(unnamed, &aliases), "!t0");
        assert_eq!(show(named, &aliases), "!t0_0");
    }

    #[test]
    fn numbered_aliases_skip_names_taken_in_between() {
        let context = Context::default();
        let names = [None, Some("t1"), None, Some("t3"), None];
        let bodies = names.map(|name| {
            let fields = [context.ptr_type(context.rec_var(0)).into()];
            match name {
                Some(name) => context.struct_type(StructType::named(sym(name), fields)),
                None => context.struct_type(StructType::new(fields)),
            }
        });
        let members = context.rec_group(&bodies).unwrap();
        let aliases = RefCell::new(TypeAliases::new());
        let shown = members.iter().map(|member| show(*member, &aliases)).collect::<Vec<_>>();
        assert_eq!(shown, ["!t0", "!t1", "!t2", "!t3", "!t4"]);
        assert_eq!(show(singleton(&context, None, context.i32()), &aliases), "!t5");
        assert_eq!(show(singleton(&context, Some("t2"), context.i32()), &aliases), "!t2_0");
    }

    #[test]
    fn naming_is_stable() {
        let context = Context::default();
        let [list, node] = pair(&context, [Some("list"), Some("node")]);
        let aliases = RefCell::new(TypeAliases::new());
        let ty = context.function_type([list], [context.ptr_type(node)]);
        let first = show(ty, &aliases);
        assert_eq!(first, "(!list) -> ptr<!node>");
        assert_eq!(show(ty, &aliases), first);
        assert_eq!(aliases.borrow().groups(), [list.rec_group().unwrap()]);
        assert_eq!(aliases.borrow_mut().alias_of(node), "node");
    }

    #[test]
    fn aliases_are_not_interned() {
        let context = Context::default();
        let first = singleton(&context, Some("alias_probe"), context.i32());
        let second = singleton(&context, Some("alias_probe"), context.i64());
        let unnamed = singleton(&context, None, context.i8());
        let aliases = RefCell::new(TypeAliases::new());
        let shown = [first, second, unnamed].map(|ty| show(ty, &aliases));
        assert_eq!(shown, ["!alias_probe", "!alias_probe_0", "!t0"]);
        assert_eq!(Symbol::lookup("alias_probe_0"), None);
    }

    #[test]
    #[should_panic(expected = "`i32` is not a member of a recursion group")]
    fn only_members_have_aliases() {
        let context = Context::default();
        TypeAliases::new().alias_of(context.i32());
    }
}

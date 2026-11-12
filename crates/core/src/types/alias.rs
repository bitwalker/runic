//! The names the IR printer gives recursion-group members.

use alloc::{format, string::String, vec::Vec};

use super::{RecGroup, Type};
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
    names: FxHashMap<Type, Symbol>,
    /// Every alias handed out so far.
    taken: FxHashSet<String>,
    /// The named groups, in the order they were first named.
    groups: Vec<RecGroup>,
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
    pub fn alias_of(&mut self, member: Type) -> Symbol {
        if let Some(alias) = self.names.get(&member) {
            return *alias;
        }
        let group = member
            .rec_group()
            .unwrap_or_else(|| panic!("`{member}` is not a member of a recursion group"));
        for member in group.members() {
            let alias = self.fresh_alias(*member);
            self.taken.insert(String::from(alias.as_str()));
            self.names.insert(*member, alias);
        }
        self.groups.push(group);
        self.names[&member]
    }

    /// The groups named so far, in the order they were first named.
    pub fn groups(&self) -> &[RecGroup] {
        &self.groups
    }

    /// An alias for `member` that no other member has.
    fn fresh_alias(&self, member: Type) -> Symbol {
        let suggested = member
            .name()
            .alias_name_fn()
            .and_then(|alias_name| alias_name(member))
            .filter(|name| is_alias_identifier(name.as_str()));
        let is_free = |alias: &str| !self.taken.contains(alias);
        match suggested {
            Some(name) if is_free(name.as_str()) => name,
            Some(name) => (0u64..)
                .map(|k| format!("{name}_{k}"))
                .find(|alias| is_free(alias))
                .map(|alias| Symbol::intern(&alias))
                .expect("some suffix is free"),
            None => (0u64..)
                .map(|k| format!("t{k}"))
                .find(|alias| is_free(alias))
                .map(|alias| Symbol::intern(&alias))
                .expect("some suffix is free"),
        }
    }
}

/// Returns true if `text` can be used as an alias name: an ASCII letter or `_`, then ASCII
/// letters, digits, `_` and `$`.
fn is_alias_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$'))
}

#[cfg(test)]
mod tests {
    use alloc::{format, string::String};
    use core::{cell::RefCell, fmt};

    use super::*;
    use crate::{Context, StructType, types::TypeFormatter};

    fn sym(s: &str) -> Symbol {
        Symbol::intern(s)
    }

    /// Writes a type with [TypeFormatter::ty], or its body with [TypeFormatter::definition], using
    /// an alias table.
    struct Aliased<'a> {
        ty: Type,
        aliases: &'a RefCell<TypeAliases>,
        definition: bool,
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
    fn naming_is_stable() {
        let context = Context::default();
        let [list, node] = pair(&context, [Some("list"), Some("node")]);
        let aliases = RefCell::new(TypeAliases::new());
        let ty = context.function_type([list], [context.ptr_type(node)]);
        let first = show(ty, &aliases);
        assert_eq!(first, "(!list) -> ptr<!node>");
        assert_eq!(show(ty, &aliases), first);
        assert_eq!(aliases.borrow().groups(), [list.rec_group().unwrap()]);
        assert_eq!(aliases.borrow_mut().alias_of(node), sym("node"));
    }

    #[test]
    #[should_panic(expected = "`i32` is not a member of a recursion group")]
    fn only_members_have_aliases() {
        let context = Context::default();
        TypeAliases::new().alias_of(context.i32());
    }
}

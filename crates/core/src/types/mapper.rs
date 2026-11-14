//! A deep, recursion-group-aware type map.

use alloc::{boxed::Box, vec::Vec};

use smallvec::SmallVec;

use super::{RecGroup, RecGroupError, Type, TypeInvariants};
use crate::{Context, FxHashMap, Report, dialects::builtin::types::RecVarType};

/// Maps every type reachable from a type, bottom-up: a type's held types are mapped first, the
/// type is rebuilt from them, and `map` is applied to the rebuilt type.
///
/// The rebuilt type is the type's value with each held type replaced by its result
/// ([Context::map_subtypes]); the result of `map` on it is final and is not mapped again. Each
/// type is mapped once per mapper: results are remembered by handle, so a type held in several
/// places, or reached again by a later call to [Self::map], is passed to `map` once. A rebuilt type
/// whose held types changed is checked with its kind's [TypeInvariants], if it has them.
///
/// A member of a recursion group is seen as its body. The group's bodies ([RecGroup::bodies]),
/// open types in which a reference to a member is a recursion variable, are mapped like any other
/// type, `map` included, and [Context::rec_group] builds a group from the mapped bodies. The result
/// for member `i` of the old group is member `i` of the new group; `map` is not applied to it. So a
/// recursive type stays recursive, and its references to itself point at the new member, even when
/// `map` changes the body's kind, name or fields. A group none of whose bodies changed is the same
/// group, and the groups a group's bodies refer to are mapped before it. `map` is never passed a
/// group member or a bare recursion variable (a variable's result is itself); it is passed bodies
/// and the open types inside them, and must handle open types. If the group of one body is built
/// from a mapped body that holds no recursion variable, [Context::rec_group] gives back that body,
/// an ordinary type, as the result.
///
/// Because `map` sees a recursive type as its open body, a map that matches a particular member by
/// handle never matches: it must match on the body's shape or name instead, or be keyed by the
/// member's body ([Type::rec_body]). Results are remembered per body, so two members with the same
/// body cannot be mapped differently.
///
/// An enum whose discriminant type changes keeps its variants' raw bits: the values of a widened
/// discriminant are zero-extended, and a narrowing that loses a set bit of a variant's value is
/// [TypeMapError::Invalid]. A caller that also rewrites integer constants compared with an enum's
/// tag must carry them the same way.
///
/// A map that changes the fields of an aggregate does not change what operations tie to the
/// aggregate's shape: the index paths of `mem.gep` and `mem.offsetof` keep their numbers. Such a
/// map needs a pattern for those operations.
///
/// For a given `map`, the result for a type depends only on the type: not on the order in which
/// types are mapped, nor on whether they are mapped by the same mapper (the context returns the
/// same group for the same bodies). `map` must therefore be a function of its argument.
///
/// The walk keeps its own stack, so the depth of a type or of a chain of groups does not consume
/// the call stack.
pub struct TypeMapper<'a> {
    context: &'a Context,
    map: BoxedMap<'a>,
    /// The result for each type mapped so far: ordinary types, bodies, and members of rebuilt
    /// groups.
    memo: FxHashMap<Type, Type>,
}

/// The function a [TypeMapper] applies.
type BoxedMap<'a> = Box<dyn FnMut(&Context, Type) -> Type + 'a>;

/// Why a type could not be mapped.
#[derive(Debug, thiserror::Error)]
pub enum TypeMapError {
    /// The mapped bodies of `group` do not define a recursion group, as `error` says: for example,
    /// `map` turned a body into a scalar in a group of several members, or returned, for a type
    /// inside a body, a type that refers to a recursion variable the group does not have.
    #[error("the mapped bodies of recursion group {} do not form a group: {error}", group.id())]
    Group {
        group: RecGroup,
        #[source]
        error: RecGroupError,
    },
    /// `ty`, rebuilt from the results of the types it holds, breaks an invariant of its kind
    /// ([TypeInvariants]), as `error` says. `ty` is the type as it was before the map.
    #[error("type '{ty}' is not valid once the types it holds are mapped: {error}")]
    Invalid {
        ty: Type,
        #[source]
        error: Report,
    },
    /// `map` was given the closed type `from` and answered the open type `to`.
    #[error("the map answered the open type '{to}' for the closed type '{from}'")]
    OpenResult { from: Type, to: Type },
}

/// A step of the walk.
enum Step {
    /// Map this type: schedule what it holds, then [Step::Rebuild] it.
    Visit(Type),
    /// Rebuild this ordinary type from the results of the types it holds, and apply `map`.
    Rebuild(Type),
    /// Build this group again from the results of its bodies.
    RebuildGroup(RecGroup),
}

impl<'a> TypeMapper<'a> {
    /// A mapper that applies `map` to the types of `context`.
    ///
    /// `map` is given a type of `context` and answers a type of `context`. For a closed argument
    /// it must answer a closed type: an open answer is [TypeMapError::OpenResult]. For an open
    /// argument (a body, or a type inside one) it may answer an open type, whose recursion
    /// variables refer to the members of the group being rebuilt; a variable the group does not
    /// have, or a body the group cannot have (a bare variable, a member of another group, a scalar
    /// in a group of several members), is [TypeMapError::Group].
    pub fn new(context: &'a Context, map: impl FnMut(&Context, Type) -> Type + 'a) -> Self {
        Self {
            context,
            map: Box::new(map),
            memo: FxHashMap::default(),
        }
    }

    /// The result of mapping `ty` and every type reachable from it; see [TypeMapper].
    ///
    /// Fails if the mapped bodies of a group do not define a group, if a rebuilt type breaks its
    /// kind's invariants, or if `map` answers an open type for a closed one. The results found
    /// before the failure are kept, and those of the failing type and of the types that reach it
    /// are not.
    pub fn map(&mut self, ty: Type) -> Result<Type, TypeMapError> {
        self.context.debug_assert_owns_type(ty, "the type being mapped");
        let mut stack = alloc::vec![Step::Visit(ty)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Visit(ty) => self.visit(ty, &mut stack),
                Step::Rebuild(ty) => self.rebuild(ty)?,
                Step::RebuildGroup(group) => self.rebuild_group(group)?,
            }
        }
        Ok(self.result(ty))
    }

    /// Schedules `ty`'s rebuild and, above it, what it holds, unless `ty` already has a result.
    ///
    /// Everything scheduled above a rebuild runs before it, so a type is rebuilt twice only if it
    /// is visited again while its rebuild is pending, that is, if it is reachable from what it
    /// holds. Outside recursion groups type storage is acyclic, and a group's bodies refer to its
    /// members through variables, never by handle, so this does not happen: a later visit always
    /// finds the result.
    fn visit(&self, ty: Type, stack: &mut Vec<Step>) {
        if self.memo.contains_key(&ty) || ty.is::<RecVarType>() {
            return;
        }
        match ty.rec_group() {
            Some(group) => {
                stack.push(Step::RebuildGroup(group));
                for body in group.bodies() {
                    stack.push(Step::Visit(*body));
                }
            }
            None => {
                stack.push(Step::Rebuild(ty));
                ty.visit_subtypes(&mut |held| stack.push(Step::Visit(held)));
            }
        }
    }

    /// The result for `ty`, which has one unless it is a recursion variable, which is its own.
    fn result(&self, ty: Type) -> Type {
        if ty.is::<RecVarType>() {
            return ty;
        }
        *self.memo.get(&ty).expect("a type held by a rebuilt type is mapped before it")
    }

    fn rebuild(&mut self, ty: Type) -> Result<(), TypeMapError> {
        debug_assert!(!self.memo.contains_key(&ty), "`{ty}` is rebuilt once (see `visit`)");
        let rebuilt = self.context.map_subtypes(ty, &mut |held| self.result(held));
        if rebuilt != ty
            && let Some(invariants) = rebuilt.as_trait::<dyn TypeInvariants>()
        {
            invariants
                .check_invariants(self.context)
                .map_err(|error| TypeMapError::Invalid { ty, error })?;
        }
        let result = (self.map)(self.context, rebuilt);
        self.context.debug_assert_owns_type(result, "the result of a type map");
        if !rebuilt.is_open() && result.is_open() {
            return Err(TypeMapError::OpenResult {
                from: rebuilt,
                to: result,
            });
        }
        self.memo.insert(ty, result);
        Ok(())
    }

    fn rebuild_group(&mut self, group: RecGroup) -> Result<(), TypeMapError> {
        let members = group.members();
        debug_assert!(
            !self.memo.contains_key(&members[0]),
            "group {} is rebuilt once (see `visit`)",
            group.id()
        );
        let bodies: SmallVec<[Type; 4]> =
            group.bodies().iter().map(|body| self.result(*body)).collect();
        let new_members = self
            .context
            .rec_group(&bodies)
            .map_err(|error| TypeMapError::Group { group, error })?;
        for (member, new_member) in members.iter().zip(new_members) {
            self.memo.insert(*member, new_member);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec::Vec};
    use core::{cell::RefCell, num::NonZeroU16};

    use super::*;
    use crate::{
        EnumType, EnumVariant, FxHashSet, StructField, StructRepr, StructType, interner::Symbol,
    };

    fn sym(s: &str) -> Symbol {
        Symbol::intern(s)
    }

    /// Maps `isize` to `i32`.
    fn isize_to_i32(context: &Context, ty: Type) -> Type {
        if ty == context.isize() {
            context.i32()
        } else {
            ty
        }
    }

    /// Maps `i32` to `isize`.
    fn i32_to_isize(context: &Context, ty: Type) -> Type {
        if ty == context.i32() {
            context.isize()
        } else {
            ty
        }
    }

    /// Every type reachable from `ty`, `ty` included; safe on cyclic storage.
    fn reachable(ty: Type) -> FxHashSet<Type> {
        let mut seen = FxHashSet::default();
        let mut stack = alloc::vec![ty];
        while let Some(ty) = stack.pop() {
            if seen.insert(ty) {
                ty.visit_subtypes(&mut |held| stack.push(held));
            }
        }
        seen
    }

    /// The type of field `index` of the struct `ty`.
    fn field(ty: Type, index: usize) -> Type {
        ty.as_struct().unwrap_or_else(|| panic!("`{ty}` is a struct")).fields()[index].ty
    }

    /// `rec { type !list = struct list<elem, ptr<!list>>; }`.
    fn list(context: &Context, elem: Type) -> Type {
        let body = context.struct_type(StructType::named(
            sym("list"),
            [elem.into(), context.ptr_type(context.rec_var(0)).into()],
        ));
        context.rec_group(&[body]).unwrap()[0]
    }

    /// The members `a` and `b` of
    /// `rec { type !a = struct a<isize, ptr<!b>>; type !b = struct b<ptr<!a>, isize>; }`.
    fn mutual(context: &Context) -> (Type, Type) {
        let isize = context.isize();
        let a = context.struct_type(StructType::named(
            sym("a"),
            [isize.into(), context.ptr_type(context.rec_var(1)).into()],
        ));
        let b = context.struct_type(StructType::named(
            sym("b"),
            [context.ptr_type(context.rec_var(0)).into(), isize.into()],
        ));
        let members = context.rec_group(&[a, b]).unwrap();
        (members[0], members[1])
    }

    /// A type of every kind, `isize` among the leaves of most.
    fn table(context: &Context) -> Vec<Type> {
        let (i8, isize) = (context.i8(), context.isize());
        let align = |n| NonZeroU16::new(n).unwrap();
        let point = context.struct_type(
            StructType::named(
                sym("point"),
                [
                    StructField::new(isize).named(sym("x")),
                    StructField::new(context.ptr_type(i8)).named(sym("y")).aligned(align(4)),
                ],
            )
            .with_repr(StructRepr::Align(align(8))),
        );
        let option = context.enum_type(
            EnumType::new(
                Some(sym("option")),
                i8,
                [EnumVariant::new(sym("none")), EnumVariant::new(sym("some")).with_payload(isize)],
            )
            .unwrap(),
        );
        let list = list(context, isize);
        let (a, b) = mutual(context);
        alloc::vec![
            context.i1(),
            i8,
            context.int_type(24),
            isize,
            context.f32(),
            context.f64(),
            context.unknown_type(),
            context.never_type(),
            context.opaque_ptr_type(),
            context.ptr_type(isize),
            context.ptr_type_in(Some(i8), 3),
            context.array_type(isize, 4),
            point,
            context.struct_type(StructType::new([isize.into(), i8.into()])),
            option,
            context.function_type([isize, i8], [point]),
            context.intern_type(crate::dialects::test::TestVectorType::new(isize, 4)),
            list,
            a,
            b,
            context.ptr_type(list),
            context.struct_type(StructType::new([context.ptr_type(b).into()])),
        ]
    }

    #[test]
    fn the_identity_maps_every_type_to_itself() {
        let context = Context::default();
        let mut mapper = TypeMapper::new(&context, |_, ty| ty);
        for ty in table(&context) {
            assert_eq!(mapper.map(ty).unwrap(), ty, "{ty}");
        }
        // A fresh mapper, with nothing remembered, gives the same answer.
        for ty in table(&context).into_iter().rev() {
            assert_eq!(TypeMapper::new(&context, |_, ty| ty).map(ty).unwrap(), ty, "{ty}");
        }
    }

    #[test]
    fn a_map_reaches_every_level() {
        let context = Context::default();
        let (i8, i32, isize) = (context.i8(), context.i32(), context.isize());
        let mut mapper = TypeMapper::new(&context, isize_to_i32);

        let nested = |int: Type| {
            context.ptr_type(
                context
                    .struct_type(StructType::new([int.into(), context.array_type(int, 4).into()])),
            )
        };
        assert_eq!(mapper.map(nested(isize)).unwrap(), nested(i32));
        assert_eq!(format!("{}", nested(i32)), "ptr<struct<i32, array<i32; 4>>>");

        let function = |int: Type| {
            context.function_type([int, context.ptr_type(int)], [context.array_type(int, 2)])
        };
        assert_eq!(mapper.map(function(isize)).unwrap(), function(i32));

        let option = |int: Type| {
            context.enum_type(
                EnumType::new(
                    Some(sym("option")),
                    i8,
                    [
                        EnumVariant::new(sym("none")),
                        EnumVariant::new(sym("some")).with_payload(int),
                        EnumVariant::new(sym("both")).with_payload(
                            context.struct_type(StructType::new([int.into(), int.into()])),
                        ),
                    ],
                )
                .unwrap(),
            )
        };
        assert_eq!(mapper.map(option(isize)).unwrap(), option(i32));
        assert_eq!(mapper.map(isize).unwrap(), i32);
    }

    #[test]
    fn a_recursive_type_stays_recursive_and_refers_to_its_new_group() {
        let context = Context::default();
        let (i32, isize) = (context.i32(), context.isize());
        let old = list(&context, isize);
        let new = TypeMapper::new(&context, isize_to_i32).map(old).unwrap();

        assert_eq!(new, list(&context, i32), "the member of the group built from the mapped body");
        let group = new.rec_group().expect("a recursion-group member");
        assert_ne!(Some(group), old.rec_group());
        assert_eq!(new.rec_index(), Some(0));
        assert_eq!(field(new, 0), i32);
        assert_eq!(field(new, 1).pointee(), Some(new), "the new member points at itself");
        assert_eq!(format!("{new}"), format!("{}", list(&context, i32)));
        let reached = reachable(new);
        assert!(!reached.contains(&old), "the old member is not reachable from the new one");
        assert!(!reached.contains(&isize));
    }

    #[test]
    fn a_two_member_group_is_rebuilt_as_one_group() {
        let context = Context::default();
        let (a, b) = mutual(&context);
        let mut mapper = TypeMapper::new(&context, isize_to_i32);
        let (new_a, new_b) = (mapper.map(a).unwrap(), mapper.map(b).unwrap());
        let group = new_a.rec_group().expect("a member");
        assert_eq!(new_b.rec_group(), Some(group));
        assert_ne!(Some(group), a.rec_group());
        assert_eq!(group.members(), [new_a, new_b]);
        assert_eq!(field(new_a, 0), context.i32());
        assert_eq!(field(new_a, 1).pointee(), Some(new_b));
        assert_eq!(field(new_b, 0).pointee(), Some(new_a));
        assert_eq!(field(new_b, 1), context.i32());
        for old in [a, b] {
            assert!(!reachable(new_a).contains(&old) && !reachable(new_b).contains(&old));
        }
    }

    #[test]
    fn the_result_does_not_depend_on_the_order_of_mapping() {
        let context = Context::default();
        let (a, b) = mutual(&context);
        // `rec { type !c = struct c<isize, ptr<!c>, ptr<a>>; }`, after `a`'s group.
        let body = context.struct_type(StructType::named(
            sym("c"),
            [
                context.isize().into(),
                context.ptr_type(context.rec_var(0)).into(),
                context.ptr_type(a).into(),
            ],
        ));
        let c = context.rec_group(&[body]).unwrap()[0];
        let through = context.ptr_type(c);
        let types = [a, b, c, through];

        let orders: [[usize; 4]; 4] = [[2, 0, 1, 3], [0, 3, 1, 2], [1, 2, 3, 0], [3, 2, 1, 0]];
        let results: Vec<[Type; 4]> = orders
            .iter()
            .map(|order| {
                let mut mapper = TypeMapper::new(&context, isize_to_i32);
                let mut results = [context.i1(); 4];
                for &index in order {
                    results[index] = mapper.map(types[index]).unwrap();
                }
                results
            })
            .collect();
        for result in &results[1..] {
            assert_eq!(result, &results[0]);
        }
        // One mapper, mapping one type at a time from scratch, agrees too.
        for (ty, expected) in types.iter().zip(results[0]) {
            assert_eq!(TypeMapper::new(&context, isize_to_i32).map(*ty).unwrap(), expected);
        }
        let [new_a, new_b, new_c, new_through] = results[0];
        assert_eq!(new_through.pointee(), Some(new_c));
        assert_eq!(field(new_c, 1).pointee(), Some(new_c));
        assert_eq!(field(new_c, 2).pointee(), Some(new_a), "the later group refers to the new `a`");
        assert_eq!(field(new_a, 1).pointee(), Some(new_b));
        for old in [a, b, c] {
            assert!(!reachable(new_through).contains(&old), "{old}");
        }
    }

    #[test]
    fn a_map_that_changes_nothing_in_a_group_keeps_the_group() {
        let context = Context::default();
        let list = list(&context, context.isize());
        let (a, b) = mutual(&context);
        let f64_to_f32 = |context: &Context, ty: Type| {
            if ty == context.f64() {
                context.f32()
            } else {
                ty
            }
        };
        let mut mapper = TypeMapper::new(&context, f64_to_f32);
        for member in [list, a, b] {
            assert_eq!(mapper.map(member).unwrap(), member);
        }
        let holder = |float: Type| {
            context.struct_type(StructType::new([float.into(), context.ptr_type(list).into()]))
        };
        assert_eq!(mapper.map(holder(context.f64())).unwrap(), holder(context.f32()));
    }

    #[test]
    fn a_map_that_reshapes_a_recursive_type_reshapes_its_references_to_itself() {
        // Gives every struct named `list` a leading `i8` field. `map` sees the list as its body,
        // `struct list<isize, ptr<rec.0>>`, so the field is added where the list refers to itself
        // too.
        let context = Context::default();
        let (i8, isize) = (context.i8(), context.isize());
        let old = list(&context, isize);
        let prepend_i8 = |context: &Context, ty: Type| match ty.as_struct() {
            Some(value) if value.name() == Some(sym("list")) => {
                context.struct_type(StructType::named(
                    sym("list"),
                    core::iter::once(StructField::new(context.i8()))
                        .chain(value.fields().iter().cloned()),
                ))
            }
            _ => ty,
        };
        let new = TypeMapper::new(&context, prepend_i8).map(old).unwrap();

        let body = context.struct_type(StructType::named(
            sym("list"),
            [i8.into(), isize.into(), context.ptr_type(context.rec_var(0)).into()],
        ));
        assert_eq!(new, context.rec_group(&[body]).unwrap()[0]);
        assert_eq!(new.rec_index(), Some(0));
        assert_eq!(field(new, 0), i8);
        assert_eq!(field(new, 1), isize);
        assert_eq!(field(new, 2).pointee(), Some(new), "the member reached is the result itself");
        assert!(!reachable(new).contains(&old));
    }

    #[test]
    fn a_map_that_turns_a_body_into_a_scalar_is_an_error() {
        // Every struct, the bodies of the group included, becomes `i32`: a group of two members
        // cannot have scalar members.
        let context = Context::default();
        let (a, b) = mutual(&context);
        let group = a.rec_group().unwrap();
        let structs_to_i32 =
            |context: &Context, ty: Type| if ty.is_struct() { context.i32() } else { ty };
        let mut mapper = TypeMapper::new(&context, structs_to_i32);
        for ty in [a, context.ptr_type(b)] {
            match mapper.map(ty) {
                Err(TypeMapError::Group {
                    group: failed,
                    error: RecGroupError::ScalarMember { member: 0 },
                }) => assert_eq!(failed, group),
                other => panic!("`{ty}`: {other:?}"),
            }
        }
    }

    #[test]
    fn a_one_body_group_whose_mapped_body_is_closed_is_an_ordinary_type() {
        // `ptr<rec.0>` becomes `ptr<i8>`: the list no longer refers to itself.
        let context = Context::default();
        let (i8, isize) = (context.i8(), context.isize());
        let open = context.ptr_type(context.rec_var(0));
        let closed = context.ptr_type(i8);
        let mut mapper =
            TypeMapper::new(&context, move |_, ty| if ty == open { closed } else { ty });
        let new = mapper.map(list(&context, isize)).unwrap();
        let expected =
            context.struct_type(StructType::named(sym("list"), [isize.into(), closed.into()]));
        assert_eq!(new, expected);
        assert_eq!(new.rec_group(), None);
    }

    /// Asserts that `result` is [TypeMapError::Group] for `group` with `error`.
    fn assert_group_error(
        result: Result<Type, TypeMapError>,
        group: RecGroup,
        error: RecGroupError,
    ) {
        match result {
            Err(TypeMapError::Group {
                group: failed,
                error: found,
            }) => assert_eq!((failed, found), (group, error)),
            other => panic!("expected a group error, got {other:?}"),
        }
    }

    #[test]
    fn a_map_that_breaks_a_group_is_an_error() {
        let context = Context::default();
        let (a, b) = mutual(&context);
        let group = a.rec_group().unwrap();
        // Turns `ptr<rec.1>`, inside `a`'s body, into a pointer to a variable the group has not.
        let out_of_range = context.ptr_type(context.rec_var(5));
        let target = context.ptr_type(context.rec_var(1));
        let mut mapper =
            TypeMapper::new(&context, move |_, ty| if ty == target { out_of_range } else { ty });
        let error = RecGroupError::VariableOutOfRange {
            member: 0,
            index: 5,
            len: 2,
        };
        assert_group_error(mapper.map(context.ptr_type(b)), group, error.clone());
        // The failing group is not remembered.
        let failure = mapper.map(a).unwrap_err();
        let message = format!("{failure}");
        assert!(message.starts_with("the mapped bodies of recursion group"), "{message}");
        assert_group_error(Err(failure), group, error);
    }

    #[test]
    fn a_closed_type_must_not_map_to_an_open_one() {
        let context = Context::default();
        let (isize, var) = (context.isize(), context.rec_var(0));
        let isize_to_var = move |_: &Context, ty: Type| if ty == isize { var } else { ty };
        for root in [context.ptr_type(isize), list(&context, isize)] {
            // Inside the list's body too: `isize` is closed wherever it is.
            match TypeMapper::new(&context, isize_to_var).map(root) {
                Err(TypeMapError::OpenResult { from, to }) => assert_eq!((from, to), (isize, var)),
                other => panic!("`{root}`: {other:?}"),
            }
        }
        let message = format!(
            "{}",
            TypeMapError::OpenResult {
                from: isize,
                to: var
            }
        );
        assert_eq!(message, "the map answered the open type 'rec.0' for the closed type 'isize'");
    }

    /// `enum<discriminant; variants>`, without a name and with no payloads.
    fn plain_enum(context: &Context, discriminant: Type, variants: &[(&str, u128)]) -> Type {
        let variants = variants
            .iter()
            .map(|(name, value)| EnumVariant::new(sym(name)).with_value(*value));
        context.enum_type(EnumType::new(None, discriminant, variants).unwrap())
    }

    /// Asserts that `result` is [TypeMapError::Invalid] for `ty`.
    fn assert_invalid(result: Result<Type, TypeMapError>, ty: Type) {
        match result {
            Err(TypeMapError::Invalid { ty: invalid, error }) => {
                assert_eq!(invalid, ty, "{error}");
                let message = format!("{}", TypeMapError::Invalid { ty, error });
                assert!(message.starts_with(&format!("type '{ty}' is not valid")), "{message}");
            }
            other => panic!("expected `{ty}` to be invalid, got {other:?}"),
        }
    }

    #[test]
    fn a_rebuilt_type_that_breaks_its_kinds_invariants_is_an_error() {
        let context = Context::default();
        let (i8, i32, isize) = (context.i8(), context.i32(), context.isize());

        // A value that does not fit the new discriminant.
        let wide = plain_enum(&context, isize, &[("a", 1 << 40)]);
        assert_invalid(TypeMapper::new(&context, isize_to_i32).map(wide), wide);
        assert_invalid(TypeMapper::new(&context, isize_to_i32).map(context.ptr_type(wide)), wide);
        // Values that do.
        let small = |int| plain_enum(&context, int, &[("a", 1), ("b", 2)]);
        let narrowed = TypeMapper::new(&context, isize_to_i32).map(small(isize)).unwrap();
        assert_eq!(narrowed, small(i32));
        assert_eq!(format!("{narrowed}"), "enum<i32; a = 1, b = 2>");

        // A discriminant that is no longer an integer.
        let i8_to_struct = |context: &Context, ty: Type| {
            if ty == context.i8() {
                context.struct_type(StructType::new([context.i1().into()]))
            } else {
                ty
            }
        };
        let flag = plain_enum(&context, i8, &[("off", 0), ("on", 1)]);
        assert_invalid(TypeMapper::new(&context, i8_to_struct).map(flag), flag);

        // The same in the body of a recursion group:
        // `rec { type !e = enum e<i8; nil, cons: ptr<!e>>; }`.
        let body = context.enum_type(
            EnumType::new(
                Some(sym("e")),
                i8,
                [
                    EnumVariant::new(sym("nil")),
                    EnumVariant::new(sym("cons"))
                        .with_payload(context.ptr_type(context.rec_var(0))),
                ],
            )
            .unwrap(),
        );
        let member = context.rec_group(&[body]).unwrap()[0];
        assert_invalid(TypeMapper::new(&context, i8_to_struct).map(member), body);
    }

    #[test]
    fn the_map_is_called_once_per_distinct_type() {
        let context = Context::default();
        let isize = context.isize();
        let ptr = context.ptr_type(isize);
        let array = context.array_type(ptr, 2);
        let list = list(&context, isize);
        let list_ptr = context.ptr_type(list);
        let open_ptr = context.ptr_type(context.rec_var(0));
        let root = context.struct_type(StructType::new([
            ptr.into(),
            ptr.into(),
            array.into(),
            list_ptr.into(),
            list_ptr.into(),
            isize.into(),
        ]));

        let calls = RefCell::new(Vec::new());
        let mut mapper = TypeMapper::new(&context, |context, ty| {
            calls.borrow_mut().push(ty);
            isize_to_i32(context, ty)
        });
        let first = mapper.map(root).unwrap();
        assert_eq!(mapper.map(root).unwrap(), first);
        assert_eq!(mapper.map(list).unwrap(), mapper.map(list_ptr).unwrap().pointee().unwrap());
        drop(mapper);

        let calls = calls.into_inner();
        let distinct: FxHashSet<Type> = calls.iter().copied().collect();
        // `map` is passed rebuilt types, and two different types rebuild to the same one when
        // they differ only by `isize` and `i32` (`ptr<isize>` and `ptr<i32>` both rebuild to
        // `ptr<i32>`). The root holds no `i32`, so here each rebuilt type is passed once.
        assert_eq!(distinct.len(), calls.len(), "no type is passed twice: {calls:?}");
        let i32 = context.i32();
        let new_list = first.as_struct().unwrap().fields()[3].ty.pointee().unwrap();
        // The rebuilt forms of `isize`, `ptr<isize>`, `array<ptr<isize>; 2>`, `ptr<rec.0>` and the
        // body `struct list<isize, ptr<rec.0>>`, the pointer to the list, and the root.
        let new_body =
            context.struct_type(StructType::named(sym("list"), [i32.into(), open_ptr.into()]));
        let expected: FxHashSet<Type> = [
            isize,
            context.ptr_type(i32),
            context.array_type(context.ptr_type(i32), 2),
            open_ptr,
            new_body,
            context.ptr_type(new_list),
            context.struct_type(StructType::new([
                context.ptr_type(i32).into(),
                context.ptr_type(i32).into(),
                context.array_type(context.ptr_type(i32), 2).into(),
                context.ptr_type(new_list).into(),
                context.ptr_type(new_list).into(),
                i32.into(),
            ])),
        ]
        .into_iter()
        .collect();
        assert_eq!(distinct, expected);
        assert_eq!(new_list.rec_group().unwrap().bodies(), [new_body]);
        assert!(!calls.contains(&context.rec_var(0)), "a recursion variable is not passed");
        assert!(!calls.contains(&list) && !calls.contains(&new_list), "a member is not passed");
    }

    #[test]
    fn a_deep_type_does_not_exhaust_the_call_stack() {
        let context = Context::default();
        let (mut before, mut after) = (context.isize(), context.i32());
        for _ in 0..20_000 {
            before = context.ptr_type(before);
            after = context.ptr_type(after);
        }
        assert_eq!(TypeMapper::new(&context, isize_to_i32).map(before).unwrap(), after);
    }

    /// A splitmix64 generator: the randomized test is deterministic for a seed.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn pick(&mut self, pool: &[Type]) -> Type {
            pool[self.below(pool.len())]
        }
    }

    /// The seeds the randomized test runs: `TYPE_MAP_TEST_SEED` if set, else three fixed ones.
    fn seeds() -> Vec<u64> {
        if let Ok(seed) = std::env::var("TYPE_MAP_TEST_SEED") {
            return alloc::vec![seed.parse().expect("TYPE_MAP_TEST_SEED is a u64")];
        }
        alloc::vec![0x7e9a_0001, 0x7e9a_0002, 0x7e9a_0003]
    }

    /// A random type over the types in `pool` and, if `vars` is not zero, the variables below it.
    fn random_type(context: &Context, rng: &mut Rng, pool: &[Type], vars: u32, depth: u32) -> Type {
        let leaf = depth == 0 || rng.below(3) == 0;
        if leaf {
            return if vars > 0 && rng.below(2) == 0 {
                context.rec_var(rng.below(vars as usize) as u32)
            } else {
                rng.pick(pool)
            };
        }
        let child = |rng: &mut Rng| random_type(context, rng, pool, vars, depth - 1);
        match rng.below(5) {
            0 => {
                let pointee = child(rng);
                context.ptr_type_in(Some(pointee), rng.below(2) as u32)
            }
            1 => {
                let element = child(rng);
                context.array_type(element, 1 + rng.below(4) as u64)
            }
            2 => random_struct(context, rng, pool, vars, depth),
            3 => {
                let params: Vec<Type> = (0..rng.below(3)).map(|_| child(rng)).collect();
                let results: Vec<Type> = (0..rng.below(2)).map(|_| child(rng)).collect();
                context.function_type(params, results)
            }
            _ => {
                let variants: Vec<EnumVariant> = (0..1 + rng.below(3))
                    .map(|index| {
                        let variant = EnumVariant::new(sym(&format!("v{index}")));
                        if rng.below(2) == 0 {
                            variant.with_payload(child(rng))
                        } else {
                            variant
                        }
                    })
                    .collect();
                context.enum_type(EnumType::new(None, context.i8(), variants).unwrap())
            }
        }
    }

    /// A random struct, sometimes named and aligned, over `pool` and the variables below `vars`.
    fn random_struct(
        context: &Context,
        rng: &mut Rng,
        pool: &[Type],
        vars: u32,
        depth: u32,
    ) -> Type {
        let fields: Vec<StructField> = (0..1 + rng.below(3))
            .map(|index| {
                let field = StructField::new(random_type(
                    context,
                    rng,
                    pool,
                    vars,
                    depth.saturating_sub(1),
                ));
                if rng.below(2) == 0 {
                    field.named(sym(&format!("f{index}")))
                } else {
                    field
                }
            })
            .collect();
        let mut ty = match rng.below(3) {
            0 => StructType::new(fields),
            _ => StructType::named(sym(&format!("s{}", rng.below(4))), fields),
        };
        if rng.below(4) == 0 {
            ty = ty.with_repr(StructRepr::Align(NonZeroU16::new(16).unwrap()));
        }
        context.struct_type(ty)
    }

    #[test]
    fn mapping_isize_to_i32_and_back_returns_the_original() {
        for seed in seeds() {
            let context = Context::default();
            let mut rng = Rng(seed);
            let mut pool = alloc::vec![
                context.isize(),
                context.i1(),
                context.i8(),
                context.i64(),
                context.f32(),
                context.opaque_ptr_type(),
            ];
            for _ in 0..60 {
                if rng.below(3) == 0 {
                    let len = 1 + rng.below(3);
                    let bodies: Vec<Type> = (0..len)
                        .map(|_| random_struct(&context, &mut rng, &pool, len as u32, 3))
                        .collect();
                    pool.extend(context.rec_group(&bodies).unwrap());
                } else {
                    let ty = random_type(&context, &mut rng, &pool, 0, 3);
                    pool.push(ty);
                }
            }

            let mut there = TypeMapper::new(&context, isize_to_i32);
            let mut back = TypeMapper::new(&context, i32_to_isize);
            // Members of groups of several members whose mapping changes them: the test is not
            // trivial.
            let mut changed_members = 0;
            for (index, &ty) in pool.iter().enumerate().rev() {
                assert!(!reachable(ty).contains(&context.i32()));
                let mapped = there.map(ty).unwrap_or_else(|err| {
                    panic!("seed {seed:#x}: type {index} `{ty}` does not map: {err}")
                });
                assert!(!reachable(mapped).contains(&context.isize()), "seed {seed:#x}: {mapped}");
                if mapped != ty && ty.rec_group().is_some_and(|group| group.members().len() > 1) {
                    changed_members += 1;
                }
                assert_eq!(
                    back.map(mapped).ok(),
                    Some(ty),
                    "seed {seed:#x} (set TYPE_MAP_TEST_SEED to rerun): type {index} `{ty}` maps \
                     to `{mapped}`"
                );
            }
            assert!(changed_members > 0, "seed {seed:#x}: no member of a larger group changed");
        }
    }
}

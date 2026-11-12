//! Recursion groups: Wasm-style iso-recursive type definitions.
//!
//! A group is an ordered list of bodies, written with ordinary type constructors over recursion
//! variables (`rec.N`, "member `N` of the group being defined"). A type that contains a variable
//! is open; [Context::rec_group](crate::Context::rec_group) turns a list of open bodies into the
//! group's members, each of which is a type with its body's name and its body's value, with
//! every variable replaced by the member it denotes. Members therefore behave like ordinary
//! types, and type storage may be cyclic.

use alloc::{boxed::Box, vec::Vec};
use core::{
    fmt,
    hash::{Hash, Hasher},
    ptr::NonNull,
};

use super::Type;
use crate::{
    FxHashSet,
    dialects::builtin::types::{
        FloatType, IntegerType, IsizeType, NeverType, RecVarType, UnknownType,
    },
};

/// The record of one recursion group, allocated in its context's arena.
pub(crate) struct RecGroupData {
    /// The group's creation order within its context, from 0.
    id: u32,
    /// Fx over the number of bodies and each body's stored hash.
    hash: u64,
    /// The open definitions: the group's identity.
    bodies: Box<[Type]>,
    /// The closed member handles; empty until every member has been filled in, and written once,
    /// before the group is published.
    members: Box<[Type]>,
}

impl RecGroupData {
    /// The record of the group `id`, whose bodies are `bodies` and whose hash is `hash`, with no
    /// members yet.
    pub(crate) fn new(id: u32, hash: u64, bodies: &[Type]) -> Self {
        Self {
            id,
            hash,
            bodies: bodies.into(),
            members: Box::default(),
        }
    }

    /// Records `members` as the group's members.
    ///
    /// # Safety
    ///
    /// `group` points at a `RecGroupData` in the arena whose pointer carries write provenance,
    /// the group has not been published (no [RecGroup] for it has been handed out, and it is not
    /// in the group table), and no reference to it exists. Member headers may point at it.
    pub(crate) unsafe fn set_members(group: NonNull<Self>, members: &[Type]) {
        // SAFETY: by the contract, nothing else accesses the record while it is written.
        unsafe { (*group.as_ptr()).members = members.into() };
    }
}

/// A handle to a recursion group; copies are equal, and two groups are equal exactly when they
/// are the same group.
#[derive(Copy, Clone)]
pub struct RecGroup(pub(crate) NonNull<RecGroupData>);

impl RecGroup {
    fn data(&self) -> &RecGroupData {
        // SAFETY: a `RecGroup` points at a record that `Context::rec_group` put in the context's
        // arena, which keeps it unmoved for as long as the context's types may be used. Before
        // the group is published the record is only read through its members' headers
        // (`Type::rec_group` on an unpublished member), and `rec_group` writes the member list
        // only after any such reference has ended; once published the record is never written
        // again.
        unsafe { self.0.as_ref() }
    }

    /// The group's members, in order.
    pub fn members(&self) -> &[Type] {
        &self.data().members
    }

    /// The group's creation order within its context, from 0.
    pub fn id(&self) -> u32 {
        self.data().id
    }

    /// The open definitions the group was created from.
    pub(crate) fn bodies(&self) -> &[Type] {
        &self.data().bodies
    }

    /// The hash the group is stored under in the group table.
    pub(crate) fn hash(&self) -> u64 {
        self.data().hash
    }
}

impl PartialEq for RecGroup {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for RecGroup {}

impl Hash for RecGroup {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Debug for RecGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecGroup")
            .field("id", &self.id())
            .field("members", &self.members())
            .finish()
    }
}

/// Why a list of bodies does not define a recursion group.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecGroupError {
    #[error("a recursion group has at least one member")]
    Empty,
    #[error("member {member} of a recursion group is a bare recursion variable")]
    BareVariable { member: usize },
    #[error("member {member} of a recursion group is a scalar type, which cannot be recursive")]
    ScalarMember { member: usize },
    #[error(
        "member {member} of a recursion group refers to variable rec.{index}, but the group has \
         {len} members"
    )]
    VariableOutOfRange {
        member: usize,
        index: u32,
        len: usize,
    },
}

/// Checks that `bodies` define a recursion group, as `Context::rec_group` describes.
pub(crate) fn validate(bodies: &[Type]) -> Result<(), RecGroupError> {
    if bodies.is_empty() {
        return Err(RecGroupError::Empty);
    }
    let one_closed_body = bodies.len() == 1 && !bodies[0].is_open();
    let mut visited = FxHashSet::default();
    let mut stack = Vec::new();
    for (member, body) in bodies.iter().enumerate() {
        if body.is::<RecVarType>() {
            return Err(RecGroupError::BareVariable { member });
        }
        if let Some(index) = variable_out_of_range(*body, bodies.len(), &mut visited, &mut stack) {
            return Err(RecGroupError::VariableOutOfRange {
                member,
                index,
                len: bodies.len(),
            });
        }
        if !one_closed_body && is_scalar(*body) {
            return Err(RecGroupError::ScalarMember { member });
        }
    }
    Ok(())
}

/// The index of a variable reachable from `ty` through open types that is not below `len`, if
/// there is one. `visited` holds open types already checked, by this or an earlier call, and
/// `stack` is scratch space.
fn variable_out_of_range(
    ty: Type,
    len: usize,
    visited: &mut FxHashSet<Type>,
    stack: &mut Vec<Type>,
) -> Option<u32> {
    if !ty.is_open() || !visited.insert(ty) {
        return None;
    }
    stack.clear();
    stack.push(ty);
    while let Some(ty) = stack.pop() {
        if let Some(var) = ty.downcast_ref::<RecVarType>() {
            if var.index() as usize >= len {
                return Some(var.index());
            }
            continue;
        }
        ty.visit_subtypes(&mut |subtype| {
            if subtype.is_open() && visited.insert(subtype) {
                stack.push(subtype);
            }
        });
    }
    None
}

/// Returns true if `ty` is a builtin scalar, which cannot be a member of a group.
fn is_scalar(ty: Type) -> bool {
    ty.is::<IntegerType>()
        || ty.is::<IsizeType>()
        || ty.is::<FloatType>()
        || ty.is::<UnknownType>()
        || ty.is::<NeverType>()
}

/// The hash of the group whose bodies are `bodies`: Fx over their number and stored hashes.
pub(crate) fn group_hash(bodies: &[Type]) -> u64 {
    let mut hasher = crate::FxHasher::default();
    hasher.write_usize(bodies.len());
    for body in bodies {
        hasher.write_u64(body.stored_hash());
    }
    hasher.finish()
}

/// Distinguishes a member's hash from the hash of an ordinary type's value.
const MEMBER_TAG: u64 = 0x7265_635f_6d65_6d62;

/// The stored hash of member `index` of the group whose hash is `group_hash`: structural, so the
/// same in every context that builds the same group.
pub(crate) fn member_hash(group_hash: u64, index: u32) -> u64 {
    let mut hasher = crate::FxHasher::default();
    hasher.write_u64(MEMBER_TAG);
    hasher.write_u64(group_hash);
    hasher.write_u32(index);
    hasher.finish()
}

/// The recursion groups of a context, keyed by their bodies.
///
/// It holds only `Copy` handles, so dropping it never reads the group records.
#[derive(Default)]
pub(crate) struct RecGroupTable {
    table: hashbrown::HashTable<RecGroup>,
    next_id: u32,
}

impl RecGroupTable {
    /// The group whose bodies are `bodies`, whose group hash is `hash`, if there is one.
    pub(crate) fn find(&self, hash: u64, bodies: &[Type]) -> Option<RecGroup> {
        self.table.find(hash, |group| group.bodies() == bodies).copied()
    }

    /// The id of the next group to be created; each call returns a new one.
    pub(crate) fn next_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = id.checked_add(1).expect("recursion group ids exhausted");
        id
    }

    /// Adds `group`, which must not be in the table yet.
    pub(crate) fn insert(&mut self, group: RecGroup) {
        self.table.insert_unique(group.hash(), group, |group| group.hash());
    }
}

/// Every recursion group reachable from `roots`, each exactly once, and each after every group
/// that its members' bodies refer to.
///
/// A group is reachable from a type that holds one of its members, directly or through other
/// types; a reachable group is followed through the types its members hold in turn. The order is
/// a depth-first post-order over groups: roots are visited in the order given, and the types a
/// type holds in the order of [Type::visit_subtypes]. A group's dependencies are the groups of
/// the members reached from its members' held types, not counting the group itself.
///
/// The walk uses explicit stacks, so the depth of the types or of a chain of groups does not
/// consume the call stack.
pub fn collect_rec_groups(roots: impl IntoIterator<Item = Type>) -> Vec<RecGroup> {
    /// A group being visited, and the groups it depends on that are still to be visited.
    struct Frame {
        group: RecGroup,
        deps: alloc::vec::IntoIter<RecGroup>,
    }

    let mut order = Vec::new();
    let mut entered = FxHashSet::default();
    let mut frames: Vec<Frame> = Vec::new();
    let enter = |group: RecGroup, entered: &mut FxHashSet<RecGroup>, frames: &mut Vec<Frame>| {
        if entered.insert(group) {
            let deps = group_dependencies(group).into_iter();
            frames.push(Frame { group, deps });
        }
    };
    for root in roots {
        for group in groups_held_by([root], None) {
            enter(group, &mut entered, &mut frames);
            while let Some(frame) = frames.last_mut() {
                match frame.deps.next() {
                    Some(dep) => enter(dep, &mut entered, &mut frames),
                    None => {
                        order.push(frame.group);
                        frames.pop();
                    }
                }
            }
        }
    }
    order
}

/// The groups `group`'s members refer to, other than `group` itself, in first-reached order.
fn group_dependencies(group: RecGroup) -> Vec<RecGroup> {
    let mut held = Vec::new();
    for member in group.members() {
        member.visit_subtypes(&mut |subtype| held.push(subtype));
    }
    groups_held_by(held, Some(group))
}

/// The groups of the members reachable from `types` without passing through a member, in
/// first-reached order, leaving out `skip`. A member is not entered: its own group's members are
/// walked when that group is visited.
fn groups_held_by(types: impl IntoIterator<Item = Type>, skip: Option<RecGroup>) -> Vec<RecGroup> {
    let mut groups = Vec::new();
    let mut seen_groups = FxHashSet::default();
    let mut visited = FxHashSet::default();
    let mut stack: Vec<Type> = types.into_iter().collect();
    stack.reverse();
    let mut children = Vec::new();
    while let Some(ty) = stack.pop() {
        if !visited.insert(ty) {
            continue;
        }
        if let Some(group) = ty.rec_group() {
            if Some(group) != skip && seen_groups.insert(group) {
                groups.push(group);
            }
            continue;
        }
        children.clear();
        ty.visit_subtypes(&mut |subtype| children.push(subtype));
        stack.extend(children.iter().rev());
    }
    groups
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec::Vec};
    use core::{
        fmt::Write,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use smallvec::SmallVec;

    use super::*;
    use crate::{
        Context, StructField, StructType,
        dialects::test::{TestDialect, TestVectorType},
        interner::Symbol,
        parse::{ParseResult, Parser},
        types::{TypeFormatter, TypeParser, TypePrinter},
    };

    fn sym(s: &str) -> Symbol {
        Symbol::intern(s)
    }

    /// The pointee of field `index` of the struct `ty`.
    fn field_pointee(ty: Type, index: usize) -> Option<Type> {
        ty.as_struct().expect("a struct").fields()[index].ty.pointee()
    }

    /// `struct list<i32, ptr<rec.0>>`, as a group of its own.
    fn self_recursive_list(context: &Context) -> Type {
        let body = context.struct_type(StructType::named(
            sym("list"),
            [context.i32().into(), context.ptr_type(context.rec_var(0)).into()],
        ));
        let members = context.rec_group(&[body]).expect("a valid group");
        assert_eq!(members.len(), 1);
        members[0]
    }

    #[test]
    fn variables_and_the_types_holding_them_are_open() {
        let context = Context::default();
        let (var0, var1) = (context.rec_var(0), context.rec_var(1));
        assert!(var0.is_open());
        assert_eq!(format!("{var0}"), "rec.0");
        assert_eq!(var0, context.rec_var(0));
        assert_ne!(var0, var1);
        assert_eq!(var1.downcast_ref::<RecVarType>().map(RecVarType::index), Some(1));
        let i32 = context.i32();
        for open in [
            context.ptr_type(var0),
            context.struct_type(StructType::new([i32.into(), context.ptr_type(var1).into()])),
            context.function_type([var0], [i32]),
        ] {
            assert!(open.is_open(), "{open}");
            assert_eq!(open.rec_group(), None);
        }
        for closed in [i32, context.ptr_type(i32)] {
            assert!(!closed.is_open(), "{closed}");
        }
    }

    #[test]
    fn a_self_recursive_struct_points_to_itself() {
        let context = Context::default();
        let member = self_recursive_list(&context);
        assert!(member.is_struct());
        let value = member.as_struct().unwrap();
        assert_eq!(value.name(), Some(sym("list")));
        assert_eq!(value.len(), 2);
        assert_eq!(value.fields()[0].ty, context.i32());
        assert_eq!(field_pointee(member, 1), Some(member));
        assert!(!member.is_open());
        let group = member.rec_group().expect("a member has a group");
        assert_eq!(group.members(), [member]);
        assert_eq!(member.rec_index(), Some(0));
        assert_eq!(group.id(), 0);
        assert_eq!(member, context.rec_group(group.bodies()).unwrap()[0]);
    }

    /// The `list`/`node` example of the design.
    fn list_and_node(context: &Context) -> SmallVec<[Type; 4]> {
        let (list, node) = (context.rec_var(0), context.rec_var(1));
        context
            .rec_group(&[
                context.struct_type(StructType::named(
                    sym("list"),
                    [context.i32().into(), context.ptr_type(node).into()],
                )),
                context.struct_type(StructType::named(
                    sym("node"),
                    [context.ptr_type(list).into(), context.ptr_type(node).into()],
                )),
            ])
            .expect("a valid group")
    }

    #[test]
    fn mutually_recursive_structs_point_to_each_other() {
        let context = Context::default();
        let members = list_and_node(&context);
        let [list, node] = [members[0], members[1]];
        assert_ne!(list, node);
        assert_eq!(list.as_struct().unwrap().name(), Some(sym("list")));
        assert_eq!(node.as_struct().unwrap().name(), Some(sym("node")));
        // list -> ptr -> node -> ptr -> list
        let node_of_list = field_pointee(list, 1).unwrap();
        assert_eq!(node_of_list, node);
        assert_eq!(field_pointee(node_of_list, 0), Some(list));
        assert_eq!(field_pointee(node, 1), Some(node));
        assert_eq!(list.rec_group(), node.rec_group());
        assert_eq!(list.rec_group().unwrap().members(), [list, node]);
        assert_eq!((list.rec_index(), node.rec_index()), (Some(0), Some(1)));
        // The pointer types in between are ordinary, closed types.
        let ptr_node = list.as_struct().unwrap().fields()[1].ty;
        assert_eq!(ptr_node, context.ptr_type(node));
        assert!(!ptr_node.is_open() && ptr_node.rec_group().is_none());
    }

    #[test]
    fn a_group_is_identified_by_its_bodies_in_order() {
        let context = Context::default();
        let first = list_and_node(&context);
        assert_eq!(first, list_and_node(&context));

        // The same bodies the other way round: variables still name positions, so this is a
        // different group with different members.
        let (list, node) = (context.rec_var(1), context.rec_var(0));
        let swapped = context
            .rec_group(&[
                context.struct_type(StructType::named(
                    sym("node"),
                    [context.ptr_type(list).into(), context.ptr_type(node).into()],
                )),
                context.struct_type(StructType::named(
                    sym("list"),
                    [context.i32().into(), context.ptr_type(node).into()],
                )),
            ])
            .unwrap();
        assert!(swapped.iter().all(|member| !first.contains(member)));

        // Two equal bodies are two members.
        let body =
            context.struct_type(StructType::new([context.ptr_type(context.rec_var(0)).into()]));
        let twins = context.rec_group(&[body, body]).unwrap();
        assert_ne!(twins[0], twins[1]);
        assert_eq!(twins[0].as_struct(), twins[1].as_struct());
    }

    #[test]
    fn a_member_is_not_the_type_of_an_equal_value() {
        let context = Context::default();
        let member = self_recursive_list(&context);
        let unrolled = || {
            context.struct_type(StructType::named(
                sym("list"),
                [context.i32().into(), context.ptr_type(member).into()],
            ))
        };
        let ordinary = unrolled();
        assert_eq!(ordinary.as_struct(), member.as_struct());
        assert_ne!(ordinary, member);
        assert_eq!(ordinary.rec_group(), None);
        assert_eq!(ordinary, unrolled());
        // Mapping a member rebuilds an ordinary type, even with the identity.
        assert_eq!(context.map_subtypes(member, &mut |ty| ty), ordinary);
    }

    #[test]
    fn a_closed_body_is_a_member_unless_it_is_alone() {
        let context = Context::default();
        let closed = context.struct_type(StructType::new([context.i32().into()]));
        let alone = context.rec_group(&[closed]).unwrap();
        assert_eq!(alone.as_slice(), [closed]);
        assert_eq!(closed.rec_group(), None);

        let other =
            context.struct_type(StructType::new([context.ptr_type(context.rec_var(0)).into()]));
        let members = context.rec_group(&[closed, other]).unwrap();
        assert!(members[0].is_struct());
        assert_eq!(members[0].as_struct(), closed.as_struct());
        assert_ne!(members[0], closed);
        assert_eq!(members[0].rec_index(), Some(0));
        assert_eq!(field_pointee(members[1], 0), Some(members[0]));
    }

    #[test]
    fn a_dialect_type_can_be_a_member() {
        let context = Context::default();
        let body =
            context.intern_type(TestVectorType::new(context.ptr_type(context.rec_var(0)), 4));
        assert!(body.is_open());
        let member = context.rec_group(&[body]).unwrap()[0];
        let vector = member.downcast_ref::<TestVectorType>().unwrap();
        assert_eq!(vector.element().pointee(), Some(member));
        assert_eq!(vector.lanes(), 4);
        assert_eq!(format!("{member}"), "!test.vector<ptr<!rec0_0> x 4>");
    }

    /// An interface implemented by [WeightedType].
    trait Weigh {
        fn weight(&self) -> u64;
        fn next(&self) -> Type;
    }

    /// Aligned beyond the header, holds a type, and implements an interface.
    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect, name = "weighted", implements(Weigh))]
    #[repr(align(16))]
    struct WeightedType {
        next: Type,
        weight: u64,
    }
    impl Weigh for WeightedType {
        fn weight(&self) -> u64 {
            self.weight
        }

        fn next(&self) -> Type {
            self.next
        }
    }
    impl TypePrinter for WeightedType {
        fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
            printer.ty(self.next)?;
            write!(printer, ", {}", self.weight)
        }
    }
    impl TypeParser for WeightedType {
        fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<Type> {
            use crate::parse::ParserExt;

            let next = parser.parse_type()?.into_inner();
            parser.parse_comma()?;
            let weight = parser.parse_decimal_integer::<u64>()?.into_inner();
            Ok(parser.context().intern_type(Self { next, weight }))
        }
    }

    #[test]
    fn a_member_answers_its_interfaces() {
        let context = Context::default();
        let body = context.intern_type(WeightedType {
            next: context.ptr_type(context.rec_var(0)),
            weight: 0xfeed_beef,
        });
        let member = context.rec_group(&[body]).unwrap()[0];
        let weighed = member.as_trait::<dyn Weigh>().expect("WeightedType implements Weigh");
        assert_eq!(weighed.weight(), 0xfeed_beef);
        assert_eq!(weighed.next().pointee(), Some(member));
        assert!((weighed as *const dyn Weigh).cast::<WeightedType>().is_aligned());
    }

    #[test]
    fn invalid_groups_are_rejected() {
        let context = Context::default();
        let (var0, var2) = (context.rec_var(0), context.rec_var(2));
        let i32 = context.i32();
        let open = context.struct_type(StructType::new([context.ptr_type(var0).into()]));
        assert_eq!(context.rec_group(&[]), Err(RecGroupError::Empty));
        assert_eq!(
            context.rec_group(&[open, var0]),
            Err(RecGroupError::BareVariable { member: 1 })
        );
        // rec.2, three levels down: struct -> ptr -> array -> rec.2.
        let deep = context
            .struct_type(StructType::new([context.ptr_type(context.array_type(var2, 2)).into()]));
        assert_eq!(
            context.rec_group(&[deep, open]),
            Err(RecGroupError::VariableOutOfRange {
                member: 0,
                index: 2,
                len: 2
            })
        );
        assert_eq!(context.rec_group(&[i32, open]), Err(RecGroupError::ScalarMember { member: 0 }));
        // A single closed scalar is returned as it is.
        assert_eq!(context.rec_group(&[i32]).unwrap().as_slice(), [i32]);
        assert_eq!(
            format!(
                "{}",
                RecGroupError::VariableOutOfRange {
                    member: 0,
                    index: 2,
                    len: 2
                }
            ),
            "member 0 of a recursion group refers to variable rec.2, but the group has 2 members"
        );
    }

    #[test]
    fn a_nested_member_prints_as_a_reference() {
        let context = Context::default();
        let unnamed = context.struct_type(StructType::new([
            context.i32().into(),
            context.ptr_type(context.rec_var(0)).into(),
        ]));
        let unnamed = context.rec_group(&[unnamed]).unwrap()[0];
        assert_eq!(format!("{unnamed}"), "struct<i32, ptr<!rec0_0>>");

        let list = self_recursive_list(&context);
        assert_eq!(format!("{list}"), "struct list<i32, ptr<!list>>");
        assert_eq!(format!("{}", context.ptr_type(list)), "ptr<!list>");
        assert_eq!(format!("{list:?}"), format!("{list}"));
        let group = list.rec_group().unwrap();
        assert_eq!(group.id(), 1);
        assert_eq!(
            format!("{group:?}"),
            "RecGroup { id: 1, members: [struct list<i32, ptr<!list>>] }"
        );

        // A name that is not a bare identifier is not used as a reference.
        let quoted = context.struct_type(StructType::named(
            sym("my list"),
            [context.ptr_type(context.rec_var(0)).into()],
        ));
        let quoted = context.rec_group(&[quoted]).unwrap()[0];
        assert_eq!(format!("{quoted}"), "struct \"my list\"<ptr<!rec2_0>>");

        let members = list_and_node(&context);
        assert_eq!(format!("{}", members[0]), "struct list<i32, ptr<!node>>");
        assert_eq!(format!("{}", members[1]), "struct node<ptr<!list>, ptr<!node>>");
    }

    #[test]
    fn member_hashes_are_structural() {
        let (first, second) = (Context::default(), Context::default());
        let (a, b) = (list_and_node(&first), list_and_node(&second));
        for (a, b) in a.iter().zip(&b) {
            assert_ne!(a, b);
            assert_eq!(a.stored_hash(), b.stored_hash());
        }
        assert_ne!(a[0].stored_hash(), a[1].stored_hash());
    }

    static DROPS: AtomicUsize = AtomicUsize::new(0);

    /// Counts its drops in [DROPS]; only `member_values_are_dropped_with_the_context` uses it.
    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect, name = "drop_counted")]
    struct DropCountedType {
        next: Type,
    }
    impl Drop for DropCountedType {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::Relaxed);
        }
    }
    impl TypePrinter for DropCountedType {
        fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
            printer.ty(self.next)
        }
    }
    impl TypeParser for DropCountedType {
        fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<Type> {
            let next = parser.parse_type()?.into_inner();
            Ok(parser.context().intern_type(Self { next }))
        }
    }

    #[test]
    fn member_values_are_dropped_with_the_context() {
        let context = Context::default();
        let bodies = [1, 0].map(|index| {
            context.intern_type(DropCountedType {
                next: context.ptr_type(context.rec_var(index)),
            })
        });
        assert_eq!(DROPS.load(Ordering::Relaxed), 0, "interning a new value drops nothing");
        let members = context.rec_group(&bodies).unwrap();
        // Each member's placeholder copy of its open body is dropped once it is filled in.
        assert_eq!(DROPS.load(Ordering::Relaxed), 2);
        assert_eq!(
            members[0].downcast_ref::<DropCountedType>().unwrap().next.pointee(),
            Some(members[1])
        );
        drop(context);
        // The two open bodies, the two placeholders and the two closed member values.
        assert_eq!(DROPS.load(Ordering::Relaxed), 6);
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_large_group_is_built_and_dropped_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                const N: u32 = 10_000;
                let context = Context::default();
                let bodies: Vec<Type> = (0..N)
                    .map(|i| {
                        let next = context.ptr_type(context.rec_var((i + 1) % N));
                        context.struct_type(StructType::new([StructField::new(next)]))
                    })
                    .collect();
                let members = context.rec_group(&bodies).unwrap();
                let group = members[0].rec_group().unwrap();
                assert_eq!(group.members().len(), N as usize);
                assert_eq!(field_pointee(group.members()[9_999], 0), Some(group.members()[0]));
                assert_eq!(field_pointee(members[0], 0), Some(members[1]));
                assert_eq!(format!("{}", members[9_999]), "struct<ptr<!rec0_0>>");
                drop(context);
            })
            .expect("spawn a thread")
            .join()
            .expect("a large group fits a 1 MB stack");
    }

    /// `struct<ptr<rec.0>>`-style singleton group, optionally also holding `held`.
    fn singleton(context: &Context, name: &str, held: Option<Type>) -> Type {
        let mut fields = SmallVec::<[StructField; 2]>::new();
        fields.push(context.ptr_type(context.rec_var(0)).into());
        if let Some(held) = held {
            fields.push(held.into());
        }
        let body = context.struct_type(StructType::named(sym(name), fields));
        context.rec_group(&[body]).expect("a valid group")[0]
    }

    #[test]
    fn types_without_members_have_no_groups() {
        let context = Context::default();
        let ty = context.ptr_type(context.struct_type(StructType::new([context.i32().into()])));
        assert!(collect_rec_groups([context.i32(), ty]).is_empty());
    }

    #[test]
    fn a_group_is_found_through_the_types_holding_its_member() {
        let context = Context::default();
        let members = list_and_node(&context);
        let group = members[0].rec_group().unwrap();
        let ty = context.ptr_type(context.array_type(members[1], 4));
        assert_eq!(collect_rec_groups([ty]), [group]);
        assert_eq!(collect_rec_groups([ty, members[0], ty]), [group]);
    }

    #[test]
    fn a_group_follows_the_groups_its_members_refer_to() {
        let context = Context::default();
        let a = singleton(&context, "a", None);
        let b = singleton(&context, "b", Some(context.ptr_type(a)));
        let (ga, gb) = (a.rec_group().unwrap(), b.rec_group().unwrap());
        assert_ne!(ga, gb);
        assert_eq!(collect_rec_groups([b]), [ga, gb]);
        assert_eq!(collect_rec_groups([b, a]), [ga, gb]);
        assert_eq!(collect_rec_groups([a, b]), [ga, gb]);
    }

    #[test]
    fn unrelated_groups_come_in_root_order() {
        let context = Context::default();
        let a = singleton(&context, "a", None);
        let b = singleton(&context, "b", None);
        let (ga, gb) = (a.rec_group().unwrap(), b.rec_group().unwrap());
        assert_eq!(collect_rec_groups([a, b]), [ga, gb]);
        assert_eq!(collect_rec_groups([b, a]), [gb, ga]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_long_chain_of_groups_is_collected_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                const N: usize = 5_000;
                let context = Context::default();
                let mut previous = None;
                let mut expected = Vec::new();
                for i in 0..N {
                    let ty = singleton(
                        &context,
                        &format!("g{i}"),
                        previous.map(|p| context.ptr_type(p)),
                    );
                    expected.push(ty.rec_group().unwrap());
                    previous = Some(ty);
                }
                assert_eq!(collect_rec_groups([previous.unwrap()]), expected);
                drop(context);
            })
            .expect("spawn a thread")
            .join()
            .expect("a long chain of groups fits a 1 MB stack");
    }

    #[cfg(feature = "std")]
    #[test]
    fn deeply_nested_types_are_collected_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                let context = Context::default();
                let member = singleton(&context, "m", None);
                let mut ty = member;
                for _ in 0..20_000 {
                    ty = context.ptr_type(ty);
                }
                assert_eq!(collect_rec_groups([ty]), [member.rec_group().unwrap()]);
                drop(context);
            })
            .expect("spawn a thread")
            .join()
            .expect("deeply nested types fit a 1 MB stack");
    }

    #[test]
    fn substituting_variables_rebuilds_the_open_types_around_them() {
        let context = Context::default();
        let open = context.ptr_type(context.array_type(context.rec_var(1), 4));
        let closed = context.substitute_rec_vars(open, &mut |index| {
            assert_eq!(index, 1);
            context.i32()
        });
        assert_eq!(closed, context.ptr_type(context.array_type(context.i32(), 4)));
        assert_eq!(
            context.substitute_rec_vars(context.i8(), &mut |_| unreachable!()),
            context.i8()
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(
        expected = "a body of a recursion group is `ptr<rec.0>`, a type of another context"
    )]
    fn a_body_of_another_context_panics() {
        let (first, second) = (Context::default(), Context::default());
        let _ = first.rec_group(&[second.ptr_type(second.rec_var(0))]);
    }
}

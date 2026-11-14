//! `TypeMapper` against an independent model.
//!
//! The model is a tree description of a type (`T`) over described recursion groups (`World`). A
//! map is applied to the *description*, and the expected type is built from scratch from the
//! mapped description; the mapper's answer for the type built from the original description must
//! be that handle.

use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
use core::num::NonZeroU16;

use crate::{
    Context, EnumType, EnumVariant, FxHashMap, FxHashSet, StructField, StructRepr, StructType,
    Type, TypeMapError, TypeMapper, dialects::builtin::types::RecVarType, interner::Symbol,
    testing::text::Rng,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Leaf {
    I1,
    I8,
    I32,
    I64,
    Isize,
    F32,
    OpaquePtr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum T {
    Leaf(Leaf),
    Ptr(Box<T>, u32),
    Array(Box<T>, u64),
    Struct {
        name: Option<u8>,
        align16: bool,
        fields: Vec<(Option<u8>, T)>,
    },
    Func(Vec<T>, Vec<T>),
    Enum {
        disc: Leaf,
        variants: Vec<Option<T>>,
    },
    /// Member `index` of group `group`: a recursion variable inside that group's own bodies.
    Member(usize, u32),
}

/// The described recursion groups, each a list of bodies, in the order they were made (a body
/// refers to members of its own group and of earlier groups only).
#[derive(Clone, Debug, Default)]
struct World {
    groups: Vec<Vec<T>>,
}

fn sym(s: &str) -> Symbol {
    Symbol::intern(s)
}

/// Builds types from descriptions, from scratch, in one context.
struct Builder<'a> {
    cx: &'a Context,
    world: &'a World,
    members: FxHashMap<usize, Result<Vec<Type>, String>>,
}

impl<'a> Builder<'a> {
    fn new(cx: &'a Context, world: &'a World) -> Self {
        Self {
            cx,
            world,
            members: FxHashMap::default(),
        }
    }

    fn leaf(&self, leaf: &Leaf) -> Type {
        let cx = self.cx;
        match leaf {
            Leaf::I1 => cx.i1(),
            Leaf::I8 => cx.i8(),
            Leaf::I32 => cx.i32(),
            Leaf::I64 => cx.i64(),
            Leaf::Isize => cx.isize(),
            Leaf::F32 => cx.f32(),
            Leaf::OpaquePtr => cx.opaque_ptr_type(),
        }
    }

    fn group(&mut self, group: usize) -> Result<Vec<Type>, String> {
        if let Some(members) = self.members.get(&group) {
            return members.clone();
        }
        let bodies: Result<Vec<Type>, String> = self.world.groups[group]
            .clone()
            .iter()
            .map(|body| self.build(body, Some(group)))
            .collect();
        let members = bodies.and_then(|bodies| {
            self.cx.rec_group(&bodies).map(|m| m.to_vec()).map_err(|err| format!("{err}"))
        });
        self.members.insert(group, members.clone());
        members
    }

    /// The type `t` describes; inside the bodies of group `current`, a reference to a member of
    /// that group is a recursion variable.
    fn build(&mut self, t: &T, current: Option<usize>) -> Result<Type, String> {
        let cx = self.cx;
        Ok(match t {
            T::Leaf(leaf) => self.leaf(leaf),
            T::Ptr(pointee, space) => {
                let pointee = self.build(pointee, current)?;
                cx.ptr_type_in(Some(pointee), *space)
            }
            T::Array(element, len) => {
                let element = self.build(element, current)?;
                cx.array_type(element, *len)
            }
            T::Struct {
                name,
                align16,
                fields,
            } => {
                let fields: Result<Vec<StructField>, String> = fields
                    .iter()
                    .map(|(name, ty)| {
                        let field = StructField::new(self.build(ty, current)?);
                        Ok(match name {
                            Some(name) => field.named(sym(&format!("f{name}"))),
                            None => field,
                        })
                    })
                    .collect();
                let fields = fields?;
                let mut ty = match name {
                    Some(name) => StructType::named(sym(&format!("s{name}")), fields),
                    None => StructType::new(fields),
                };
                if *align16 {
                    ty = ty.with_repr(StructRepr::Align(NonZeroU16::new(16).unwrap()));
                }
                cx.struct_type(ty)
            }
            T::Func(params, results) => {
                let params: Result<Vec<Type>, String> =
                    params.iter().map(|p| self.build(p, current)).collect();
                let results: Result<Vec<Type>, String> =
                    results.iter().map(|r| self.build(r, current)).collect();
                cx.function_type(params?, results?)
            }
            T::Enum { disc, variants } => {
                let disc = self.leaf(disc);
                let variants: Result<Vec<EnumVariant>, String> = variants
                    .iter()
                    .enumerate()
                    .map(|(index, payload)| {
                        let variant = EnumVariant::new(sym(&format!("v{index}")));
                        Ok(match payload {
                            Some(payload) => variant.with_payload(self.build(payload, current)?),
                            None => variant,
                        })
                    })
                    .collect();
                let ty = EnumType::new(None, disc, variants?).map_err(|err| format!("{err}"))?;
                cx.enum_type(ty)
            }
            T::Member(group, index) => {
                if Some(*group) == current {
                    cx.rec_var(*index)
                } else {
                    self.group(*group)?[*index as usize]
                }
            }
        })
    }
}

/// Applies `f` bottom-up to a description: the parts first, then the whole. A member reference
/// is not passed to `f` (the mapper never passes a member or a variable either).
fn map_t(t: &T, f: &dyn Fn(T) -> T) -> T {
    let m = |t: &T| map_t(t, f);
    let rebuilt = match t {
        T::Leaf(leaf) => T::Leaf(leaf.clone()),
        T::Ptr(pointee, space) => T::Ptr(Box::new(m(pointee)), *space),
        T::Array(element, len) => T::Array(Box::new(m(element)), *len),
        T::Struct {
            name,
            align16,
            fields,
        } => T::Struct {
            name: *name,
            align16: *align16,
            fields: fields.iter().map(|(name, ty)| (*name, m(ty))).collect(),
        },
        T::Func(params, results) => {
            T::Func(params.iter().map(m).collect(), results.iter().map(m).collect())
        }
        T::Enum { disc, variants } => {
            let disc = match f(T::Leaf(disc.clone())) {
                T::Leaf(leaf) => leaf,
                other => panic!("the model maps a discriminant to a leaf, got {other:?}"),
            };
            T::Enum {
                disc,
                variants: variants.iter().map(|payload| payload.as_ref().map(m)).collect(),
            }
        }
        T::Member(group, index) => return T::Member(*group, *index),
    };
    f(rebuilt)
}

fn map_world(world: &World, f: &dyn Fn(T) -> T) -> World {
    World {
        groups: world
            .groups
            .iter()
            .map(|bodies| bodies.iter().map(|body| map_t(body, f)).collect())
            .collect(),
    }
}

// ---- the maps, each as a function of descriptions and as a function of handles ----

fn leaf_isize_to_i32(t: T) -> T {
    if t == T::Leaf(Leaf::Isize) {
        T::Leaf(Leaf::I32)
    } else {
        t
    }
}

fn handle_isize_to_i32(cx: &Context, ty: Type) -> Type {
    if ty == cx.isize() { cx.i32() } else { ty }
}

fn handle_i32_to_isize(cx: &Context, ty: Type) -> Type {
    if ty == cx.i32() { cx.isize() } else { ty }
}

/// Every struct named `s0` gets a leading `i8` field named `f9`.
fn reshape_desc(t: T) -> T {
    match t {
        T::Struct {
            name: Some(0),
            align16,
            mut fields,
        } => {
            fields.insert(0, (Some(9), T::Leaf(Leaf::I8)));
            T::Struct {
                name: Some(0),
                align16,
                fields,
            }
        }
        other => other,
    }
}

fn reshape_handle(cx: &Context, ty: Type) -> Type {
    let Some(st) = ty.as_struct() else {
        return ty;
    };
    if st.name() != Some(sym("s0")) {
        return ty;
    }
    let mut fields = vec![StructField::new(cx.i8()).named(sym("f9"))];
    fields.extend(st.fields().iter().copied());
    cx.struct_type(StructType::named(sym("s0"), fields).with_repr(st.repr()))
}

/// Both at once: leaves change and bodies are reshaped.
fn both_desc(t: T) -> T {
    reshape_desc(leaf_isize_to_i32(t))
}

fn both_handle(cx: &Context, ty: Type) -> Type {
    reshape_handle(cx, handle_isize_to_i32(cx, ty))
}

// ---- random descriptions ----

fn random_leaf(rng: &mut Rng) -> Leaf {
    match rng.below(8) {
        0 => Leaf::I1,
        1 => Leaf::I8,
        2 => Leaf::I64,
        3 => Leaf::F32,
        4 => Leaf::OpaquePtr,
        _ => Leaf::Isize,
    }
}

/// A random description over the pool of earlier descriptions, the earlier groups, and (inside a
/// body) the members of the group being described.
fn random_t(
    rng: &mut Rng,
    pool: &[T],
    world: &World,
    current: Option<(usize, u32)>,
    depth: u32,
) -> T {
    if depth == 0 || rng.below(3) == 0 {
        return match rng.below(6) {
            0 | 1 if current.is_some() => {
                let (group, len) = current.unwrap();
                T::Member(group, rng.below(u64::from(len)) as u32)
            }
            2 if !world.groups.is_empty() => {
                let group = rng.below(world.groups.len() as u64) as usize;
                let len = world.groups[group].len() as u64;
                T::Member(group, rng.below(len) as u32)
            }
            3 if !pool.is_empty() => pool[rng.below(pool.len() as u64) as usize].clone(),
            _ => T::Leaf(random_leaf(rng)),
        };
    }
    let child = |rng: &mut Rng| random_t(rng, pool, world, current, depth - 1);
    match rng.below(5) {
        0 => T::Ptr(Box::new(child(rng)), rng.below(2) as u32),
        1 => T::Array(Box::new(child(rng)), 1 + rng.below(4)),
        2 => random_struct(rng, pool, world, current, depth),
        3 => T::Func(
            (0..rng.below(3)).map(|_| child(rng)).collect(),
            (0..rng.below(2)).map(|_| child(rng)).collect(),
        ),
        _ => T::Enum {
            disc: if rng.below(2) == 0 {
                Leaf::I8
            } else {
                Leaf::Isize
            },
            variants: (0..1 + rng.below(3))
                .map(|_| (rng.below(2) == 0).then(|| child(rng)))
                .collect(),
        },
    }
}

fn random_struct(
    rng: &mut Rng,
    pool: &[T],
    world: &World,
    current: Option<(usize, u32)>,
    depth: u32,
) -> T {
    T::Struct {
        name: (rng.below(3) != 0).then(|| rng.below(4) as u8),
        align16: rng.below(4) == 0,
        fields: (0..1 + rng.below(3))
            .map(|index| {
                let name = (rng.below(2) == 0).then_some(index as u8);
                (name, random_t(rng, pool, world, current, depth.saturating_sub(1)))
            })
            .collect(),
    }
}

/// A random world and a pool of closed descriptions over it (members of every group included).
fn random_world(rng: &mut Rng, steps: usize) -> (World, Vec<T>) {
    let mut world = World::default();
    let mut pool: Vec<T> = Vec::new();
    for _ in 0..steps {
        if rng.below(3) == 0 {
            let group = world.groups.len();
            let len = 1 + rng.below(3) as u32;
            let bodies: Vec<T> = (0..len)
                .map(|_| random_struct(rng, &pool, &world, Some((group, len)), 3))
                .collect();
            world.groups.push(bodies);
            pool.extend((0..len).map(|index| T::Member(group, index)));
        } else {
            let t = random_t(rng, &pool, &world, None, 3);
            pool.push(t);
        }
    }
    (world, pool)
}

fn reachable(ty: Type) -> FxHashSet<Type> {
    let mut seen = FxHashSet::default();
    let mut stack = vec![ty];
    while let Some(ty) = stack.pop() {
        if seen.insert(ty) {
            ty.visit_subtypes(&mut |held| stack.push(held));
        }
    }
    seen
}

type DescMap = fn(T) -> T;
type HandleMap = fn(&Context, Type) -> Type;

const MAPS: [(&str, DescMap, HandleMap); 4] = [
    ("identity", |t| t, |_, ty| ty),
    ("isize -> i32", leaf_isize_to_i32, handle_isize_to_i32),
    ("reshape s0", reshape_desc, reshape_handle),
    ("isize -> i32 and reshape s0", both_desc, both_handle),
];

/// For random types over random groups, the mapper's answer is the type built from scratch
/// from the mapped description: with one long-lived mapper in a random order, with a second
/// long-lived mapper in the reverse order, and with a fresh mapper per type. The identity gives
/// back every handle; `map` is passed no type twice by one mapper; a mapped recursive type does
/// not reach an old member whose body changed.
#[test]
fn the_mapper_agrees_with_a_model_built_from_scratch() {
    let mut recursive_changed = 0;
    for seed in crate::testing::test_seeds(60) {
        let mut rng = Rng(seed);
        let (world, pool) = random_world(&mut rng, 50);
        for (name, desc_map, handle_map) in MAPS {
            let cx = Context::default();
            let mut original = Builder::new(&cx, &world);
            let types: Vec<Type> = pool
                .iter()
                .map(|t| original.build(t, None).expect("the model builds"))
                .collect();

            let mapped_world = map_world(&world, &desc_map);
            let mut mapped = Builder::new(&cx, &mapped_world);
            let expected: Vec<Type> = pool
                .iter()
                .map(|t| {
                    mapped
                        .build(&map_t(t, &desc_map), None)
                        .unwrap_or_else(|err| panic!("seed {seed:#x} {name}: {err}"))
                })
                .collect();

            // one mapper, a random order
            let mut order: Vec<usize> = (0..types.len()).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let passed = core::cell::RefCell::new(Vec::<Type>::new());
            {
                let mut mapper = TypeMapper::new(&cx, |cx, ty| {
                    passed.borrow_mut().push(ty);
                    handle_map(cx, ty)
                });
                for &index in &order {
                    let got = mapper.map(types[index]).unwrap_or_else(|err| {
                        panic!("seed {seed:#x} {name}: `{}` does not map: {err}", types[index])
                    });
                    assert_eq!(
                        got, expected[index],
                        "seed {seed:#x} {name}: `{}` (type {index}, {:?})",
                        types[index], pool[index]
                    );
                }
            }
            let passed = passed.into_inner();
            let distinct: FxHashSet<Type> = passed.iter().copied().collect();
            assert_eq!(distinct.len(), passed.len(), "seed {seed:#x} {name}: a type passed twice");
            assert!(
                passed.iter().all(|ty| ty.rec_group().is_none() && !ty.is::<RecVarType>()),
                "seed {seed:#x} {name}: `map` was passed a member or a variable"
            );

            // a second mapper in the reverse order, and a fresh mapper per type
            let mut reverse = TypeMapper::new(&cx, handle_map);
            for &index in order.iter().rev() {
                assert_eq!(reverse.map(types[index]).unwrap(), expected[index], "{seed:#x} {name}");
            }
            for (index, ty) in types.iter().enumerate() {
                let got = TypeMapper::new(&cx, handle_map).map(*ty).unwrap();
                assert_eq!(got, expected[index], "seed {seed:#x} {name}: fresh mapper");
            }

            if name == "identity" {
                assert_eq!(types, expected, "seed {seed:#x}: the model's identity");
            }

            // a recursive type whose group changed does not reach its old members
            for (index, ty) in types.iter().enumerate() {
                let Some(group) = ty.rec_group() else {
                    continue;
                };
                let new = expected[index];
                if new == *ty {
                    continue;
                }
                recursive_changed += 1;
                assert!(new.rec_group().is_some() || !new.is_open());
                let reached = reachable(new);
                if let Some(new_group) = new.rec_group() {
                    assert_ne!(new_group.id(), group.id());
                    for member in group.members() {
                        assert!(
                            !reached.contains(member),
                            "seed {seed:#x} {name}: `{new}` reaches the old member `{member}`"
                        );
                    }
                }
            }
        }
    }
    assert!(recursive_changed > 100, "the test maps recursive types: {recursive_changed}");
}

/// There and back: on types with no `i32`, `isize -> i32 -> isize` gives back the original
/// handle, with the mapped handle equal to the model's.
#[test]
fn there_and_back_returns_the_original_handles() {
    for seed in crate::testing::test_seeds(60) {
        let mut rng = Rng(seed ^ 0xbac4);
        let (world, pool) = random_world(&mut rng, 50);
        let cx = Context::default();
        let mut original = Builder::new(&cx, &world);
        let types: Vec<Type> = pool.iter().map(|t| original.build(t, None).unwrap()).collect();
        let mut there = TypeMapper::new(&cx, handle_isize_to_i32);
        let mut back = TypeMapper::new(&cx, handle_i32_to_isize);
        for ty in types.iter().rev() {
            assert!(!reachable(*ty).contains(&cx.i32()));
            let mapped = there.map(*ty).unwrap();
            assert!(!reachable(mapped).contains(&cx.isize()), "seed {seed:#x}");
            assert_eq!(back.map(mapped).unwrap(), *ty, "seed {seed:#x}: `{ty}` via `{mapped}`");
        }
    }
}

/// Invalid maps are reported, not panicked on, and the mapper stays usable: after an error it
/// still maps an unrelated type, and gives the same error for the failing type again.
#[test]
fn invalid_maps_are_errors_and_the_mapper_stays_usable() {
    type Bad = (&'static str, fn(&Context, Type) -> Type);
    let bad: [Bad; 7] = [
        ("every struct becomes i32", |cx, ty| if ty.is_struct() { cx.i32() } else { ty }),
        (
            "isize becomes rec.0",
            |cx, ty| if ty == cx.isize() { cx.rec_var(0) } else { ty },
        ),
        (
            "isize becomes rec.7",
            |cx, ty| if ty == cx.isize() { cx.rec_var(7) } else { ty },
        ),
        ("i8 becomes f32", |cx, ty| if ty == cx.i8() { cx.f32() } else { ty }),
        ("every open struct becomes rec.0", |cx, ty| {
            if ty.is_struct() && ty.is_open() {
                cx.rec_var(0)
            } else {
                ty
            }
        }),
        ("every open type becomes a variable out of range", |cx, ty| {
            if ty.is_open() {
                cx.ptr_type(cx.rec_var(40))
            } else {
                ty
            }
        }),
        (
            "every pointer becomes isize",
            |cx, ty| if ty.is_pointer() { cx.isize() } else { ty },
        ),
    ];
    let mut errors = [0usize; 7];
    for seed in crate::testing::test_seeds(30) {
        let mut rng = Rng(seed ^ 0xbad);
        let (world, pool) = random_world(&mut rng, 40);
        for (which, (name, map)) in bad.iter().enumerate() {
            let cx = Context::default();
            let mut builder = Builder::new(&cx, &world);
            let types: Vec<Type> = pool.iter().map(|t| builder.build(t, None).unwrap()).collect();
            let mut mapper = TypeMapper::new(&cx, *map);
            let mut outcomes: Vec<Result<Type, String>> = Vec::new();
            for ty in &types {
                outcomes.push(mapper.map(*ty).map_err(|err| format!("{err}")));
            }
            errors[which] += outcomes.iter().filter(|o| o.is_err()).count();
            // the same mapper again, and a fresh mapper per type, give the same outcome
            for (index, ty) in types.iter().enumerate().rev() {
                let again = mapper.map(*ty).map_err(|err| format!("{err}"));
                assert_eq!(again, outcomes[index], "seed {seed:#x} {name}: `{ty}` again");
                let fresh = TypeMapper::new(&cx, *map).map(*ty).map_err(|err| format!("{err}"));
                assert_eq!(fresh, outcomes[index], "seed {seed:#x} {name}: `{ty}` fresh");
            }
            // an Ok answer is closed and owned
            for (ty, outcome) in types.iter().zip(&outcomes) {
                if let Ok(new) = outcome {
                    assert!(!new.is_open(), "seed {seed:#x} {name}: `{ty}` maps to open `{new}`");
                    assert!(cx.owns_type(*new));
                }
            }
        }
    }
    std::eprintln!("errors per invalid map: {errors:?}");
    assert!(
        errors.iter().take(6).all(|&n| n > 0),
        "every invalid map was exercised: {errors:?}"
    );
}

/// A map that changes an enum's discriminant keeps the variants' raw bits: widening `i8` with a
/// variant of value -1 (bits 0xff) gives the value 255, and narrowing `isize` with a variant of
/// value -1 is `Invalid`, though -1 fits `i32` as a signed number.
#[test]
fn an_enum_keeps_raw_bits_when_its_discriminant_changes() {
    let cx = Context::default();
    let small = cx.enum_type(
        EnumType::new(None, cx.i8(), [EnumVariant::new(sym("neg")).with_value(0xff)]).unwrap(),
    );
    let widened = TypeMapper::new(&cx, |cx, ty| if ty == cx.i8() { cx.i32() } else { ty })
        .map(small)
        .unwrap();
    let widened = widened.as_enum().unwrap();
    assert_eq!(widened.discriminant(), cx.i32());
    assert_eq!(widened.variants()[0].value(), 0xff, "-1 became 255");

    let wide = cx.enum_type(
        EnumType::new(
            None,
            cx.isize(),
            [EnumVariant::new(sym("neg")).with_value(u64::MAX as u128)],
        )
        .unwrap(),
    );
    let narrowed =
        TypeMapper::new(&cx, |cx, ty| if ty == cx.isize() { cx.i32() } else { ty }).map(wide);
    assert!(matches!(narrowed, Err(TypeMapError::Invalid { .. })), "{narrowed:?}");
}

/// A map written as a table from handles to handles leaves a recursive type as it was, because
/// `map` sees a member as its body; keyed by the member's body ([Type::rec_body]), it works.
#[test]
fn a_handle_keyed_map_misses_a_recursive_member() {
    let cx = Context::default();
    let body = cx.struct_type(StructType::named(
        sym("list"),
        [cx.i64().into(), cx.ptr_type(cx.rec_var(0)).into()],
    ));
    let list = cx.rec_group(&[body]).unwrap()[0];
    let holder = cx.ptr_type(list);

    // "every `list` becomes an opaque pointer", keyed by the member's handle
    let table: FxHashMap<Type, Type> = [(list, cx.opaque_ptr_type())].into_iter().collect();
    let mut by_handle = TypeMapper::new(&cx, |_, ty| table.get(&ty).copied().unwrap_or(ty));
    assert_eq!(by_handle.map(list).unwrap(), list, "the member was not matched");
    assert_eq!(by_handle.map(holder).unwrap(), holder);

    // the same table keyed by the member's body
    assert_eq!(list.rec_body(), Some(body));
    assert_eq!(body.rec_body(), None, "a body is not a member");
    let table: FxHashMap<Type, Type> = [(body, cx.opaque_ptr_type())].into_iter().collect();
    let mut by_body = TypeMapper::new(&cx, |_, ty| table.get(&ty).copied().unwrap_or(ty));
    assert_eq!(by_body.map(list).unwrap(), cx.opaque_ptr_type());
    assert_eq!(by_body.map(holder).unwrap(), cx.ptr_type(cx.opaque_ptr_type()));
}

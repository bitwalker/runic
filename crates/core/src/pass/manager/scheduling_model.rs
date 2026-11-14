//! A model of pass scheduling: random pipelines of typed and untyped passes over random nests of
//! test operations, built through the API and through pipeline text, against an independent
//! statement of which pass runs on which operation in which order.
//!
//! The statement (`Model`) knows nothing of adaptors or of merging: a nested pipeline runs, as a
//! whole stage, on the operations of its anchor directly inside the operation (or, when
//! recursive, on the outermost ones at any depth). The pass manager merges adjacent nested
//! pipelines, which interleaves the stages per operation: that changes the order of two runs
//! only when their operations are in different subtrees, so the comparison is on the order of
//! every pair of runs whose operations are the same or one inside the other.

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::{cell::RefCell, fmt};

use super::*;
use crate::{
    Builder, BuilderExt, OpBuilder, RegionKindInterface, SourceSpan,
    dialects::test::TestDialect,
    pass::{PassPipeline, registry::register_pass, write_pass_option},
    traits::{HasOnlyGraphRegion, NoTerminator},
};

macro_rules! container {
    ($ty:ident, $($traits:ident),*) => {
        #[crate::derive::operation(
            dialect = TestDialect,
            traits(NoTerminator, HasOnlyGraphRegion $(, $traits)*),
            implements(RegionKindInterface)
        )]
        pub struct $ty {
            #[region]
            body: crate::RegionRef,
        }

        impl crate::RegionKindInterface for $ty {
            fn kind(&self) -> crate::RegionKind {
                crate::RegionKind::Graph
            }
        }
    };
}

container!(SchedA, IsolatedFromAbove);
container!(SchedB, IsolatedFromAbove);
container!(SchedC, IsolatedFromAbove);
// Not isolated from above: nothing may be scheduled on it.
container!(SchedPlain,);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    A,
    B,
    C,
    Plain,
}

impl Kind {
    fn op_name(self) -> &'static str {
        match self {
            Kind::A => "test.sched_a",
            Kind::B => "test.sched_b",
            Kind::C => "test.sched_c",
            Kind::Plain => "test.sched_plain",
        }
    }

    fn isolated(self) -> bool {
        self != Kind::Plain
    }

    fn of(name: &OperationName) -> Option<Kind> {
        if name.is::<SchedA>() {
            Some(Kind::A)
        } else if name.is::<SchedB>() {
            Some(Kind::B)
        } else if name.is::<SchedC>() {
            Some(Kind::C)
        } else if name.is::<SchedPlain>() {
            Some(Kind::Plain)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The IR: a tree of containers
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Node {
    id: usize,
    kind: Kind,
    children: Vec<Node>,
}

fn build(node: &Node, builder: &mut OpBuilder, ops: &mut Vec<(OperationRef, usize)>) {
    let span = SourceSpan::UNKNOWN;
    let op = match node.kind {
        Kind::A => builder.create::<SchedA, ()>(span)().unwrap().as_operation_ref(),
        Kind::B => builder.create::<SchedB, ()>(span)().unwrap().as_operation_ref(),
        Kind::C => builder.create::<SchedC, ()>(span)().unwrap().as_operation_ref(),
        Kind::Plain => builder.create::<SchedPlain, ()>(span)().unwrap().as_operation_ref(),
    };
    ops.push((op, node.id));
    let region = op.borrow().regions().front().as_pointer().unwrap();
    let block = builder.create_block(region, None, &[]);
    for child in &node.children {
        builder.set_insertion_point_to_end(block);
        build(child, builder, ops);
    }
}

// ---------------------------------------------------------------------------------------------
// The passes: each records (its id, the operation) in a thread-local
// ---------------------------------------------------------------------------------------------

std::thread_local! {
    static EVENTS: RefCell<Vec<(usize, OperationRef)>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassKind {
    /// `Target` is the operation type of the kind; schedules on it only.
    Typed(Kind),
    /// `Target = Operation`; schedules on everything.
    Any,
    /// `Target = Operation`; schedules on `test.sched_a` only.
    OnlyA,
    /// `Target = Operation`; schedules on `test.sched_a` and `test.sched_b`.
    AOrB,
}

impl PassKind {
    fn argument(self) -> &'static str {
        match self {
            PassKind::Typed(Kind::A) => "sched-a",
            PassKind::Typed(Kind::B) => "sched-b",
            PassKind::Typed(Kind::C) => "sched-c",
            PassKind::Typed(Kind::Plain) => unreachable!(),
            PassKind::Any => "sched-any",
            PassKind::OnlyA => "sched-only-a",
            PassKind::AOrB => "sched-a-or-b",
        }
    }

    fn target(self) -> Option<Kind> {
        match self {
            PassKind::Typed(kind) => Some(kind),
            _ => None,
        }
    }

    fn can_schedule_on(self, kind: Kind) -> bool {
        match self {
            PassKind::Typed(target) => kind == target,
            PassKind::Any => true,
            PassKind::OnlyA => kind == Kind::A,
            PassKind::AOrB => matches!(kind, Kind::A | Kind::B),
        }
    }
}

macro_rules! recording_pass {
    ($ty:ident, $target:ty, $kind:expr) => {
        #[derive(Default)]
        struct $ty {
            id: usize,
        }

        impl Pass for $ty {
            type Target = $target;

            fn name(&self) -> &'static str {
                $kind.argument()
            }

            fn argument(&self) -> &'static str {
                $kind.argument()
            }

            fn description(&self) -> &'static str {
                "a pass of the scheduling model"
            }

            fn initialize_options(&mut self, options: &str) -> Result<(), Report> {
                let Some(id) = options.strip_prefix("id=") else {
                    return Err(Report::msg(format!("expected id=<n>, got '{options}'")));
                };
                self.id = id.parse().map_err(|_| Report::msg("id is not a number"))?;
                Ok(())
            }

            fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}{{", $kind.argument())?;
                write_pass_option(f, "id", Some(&self.id.to_string()))?;
                f.write_str("}")
            }

            fn can_schedule_on(&self, name: &OperationName) -> bool {
                Kind::of(name).is_some_and(|kind| $kind.can_schedule_on(kind))
            }

            fn run_on_operation(
                &mut self,
                _op: EntityMut<'_, $target>,
                state: &mut PassExecutionState,
            ) -> Result<(), Report> {
                let op = *state.current_operation();
                EVENTS.with(|events| events.borrow_mut().push((self.id, op)));
                state.set_post_pass_status(PostPassStatus::Unchanged);
                Ok(())
            }
        }
    };
}

recording_pass!(PassA, SchedA, PassKind::Typed(Kind::A));
recording_pass!(PassB, SchedB, PassKind::Typed(Kind::B));
recording_pass!(PassC, SchedC, PassKind::Typed(Kind::C));
recording_pass!(PassAny, Operation, PassKind::Any);
recording_pass!(PassOnlyA, Operation, PassKind::OnlyA);
recording_pass!(PassAOrB, Operation, PassKind::AOrB);

fn make(kind: PassKind, id: usize) -> Box<dyn OperationPass> {
    match kind {
        PassKind::Typed(Kind::A) => Box::new(PassA { id }),
        PassKind::Typed(Kind::B) => Box::new(PassB { id }),
        PassKind::Typed(Kind::C) => Box::new(PassC { id }),
        PassKind::Typed(Kind::Plain) => unreachable!(),
        PassKind::Any => Box::new(PassAny { id }),
        PassKind::OnlyA => Box::new(PassOnlyA { id }),
        PassKind::AOrB => Box::new(PassAOrB { id }),
    }
}

fn register() {
    register_pass(|| Box::<PassA>::default()).unwrap();
    register_pass(|| Box::<PassB>::default()).unwrap();
    register_pass(|| Box::<PassC>::default()).unwrap();
    register_pass(|| Box::<PassAny>::default()).unwrap();
    register_pass(|| Box::<PassOnlyA>::default()).unwrap();
    register_pass(|| Box::<PassAOrB>::default()).unwrap();
}

// ---------------------------------------------------------------------------------------------
// The pipeline, as written
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Elem {
    Pass {
        id: usize,
        kind: PassKind,
    },
    Nested {
        recursive: bool,
        anchor: Option<Kind>,
        elems: Vec<Elem>,
    },
}

fn anchor_text(anchor: Option<Kind>) -> &'static str {
    anchor.map_or("any", Kind::op_name)
}

fn write_elems(out: &mut String, elems: &[Elem]) {
    for (i, elem) in elems.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match elem {
            Elem::Pass { id, kind } => out.push_str(&format!("{}{{id=\"{id}\"}}", kind.argument())),
            Elem::Nested {
                recursive,
                anchor,
                elems,
            } => {
                if *recursive {
                    out.push_str("recursive(");
                }
                out.push_str(anchor_text(*anchor));
                out.push('(');
                write_elems(out, elems);
                out.push(')');
                if *recursive {
                    out.push(')');
                }
            }
        }
    }
}

fn pipeline_text(anchor: Option<Kind>, elems: &[Elem]) -> String {
    let mut out = String::from(anchor_text(anchor));
    out.push('(');
    write_elems(&mut out, elems);
    out.push(')');
    out
}

/// The pipeline as the pass manager holds it once every pass is added: with implicit nesting a
/// typed pass under an anchor of another type is in a nested pipeline of its own; with explicit
/// nesting that is an error.
fn resolve(anchor: Option<Kind>, nesting: Nesting, elems: &[Elem]) -> Result<Vec<Elem>, ()> {
    let mut resolved = Vec::new();
    for elem in elems {
        match elem {
            Elem::Pass { id, kind } => match (anchor, kind.target()) {
                (Some(anchor), Some(target)) if anchor != target => {
                    if nesting == Nesting::Explicit {
                        return Err(());
                    }
                    resolved.push(Elem::Nested {
                        recursive: false,
                        anchor: Some(target),
                        elems: vec![Elem::Pass {
                            id: *id,
                            kind: *kind,
                        }],
                    });
                }
                _ => resolved.push(elem.clone()),
            },
            Elem::Nested {
                recursive,
                anchor,
                elems,
            } => resolved.push(Elem::Nested {
                recursive: *recursive,
                anchor: *anchor,
                elems: resolve(*anchor, nesting, elems)?,
            }),
        }
    }
    Ok(resolved)
}

fn add(pm: &mut OpPassManager, elems: &[Elem]) -> Result<(), Report> {
    for elem in elems {
        match elem {
            Elem::Pass { id, kind } => pm.add_pass(make(*kind, *id))?,
            Elem::Nested {
                recursive,
                anchor,
                elems,
            } => {
                let mut nested = if *recursive {
                    pm.nest_recursive_with_type(anchor_text(*anchor))?
                } else {
                    pm.nest_with_type(anchor_text(*anchor))?
                };
                add(&mut nested, elems)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The statement of what runs
// ---------------------------------------------------------------------------------------------

struct Model {
    events: Vec<(usize, usize)>,
}

impl Model {
    /// A typed manager holds only passes that can be scheduled on its anchor (checked for every
    /// manager before anything runs).
    fn check(anchor: Option<Kind>, elems: &[Elem]) -> Result<(), ()> {
        for elem in elems {
            match elem {
                Elem::Pass { kind, .. } => {
                    if anchor.is_some_and(|anchor| !kind.can_schedule_on(anchor)) {
                        return Err(());
                    }
                }
                Elem::Nested { anchor, elems, .. } => Self::check(*anchor, elems)?,
            }
        }
        Ok(())
    }

    fn schedules(anchor: Option<Kind>, elems: &[Elem], kind: Kind) -> bool {
        match anchor {
            Some(anchor) => anchor == kind,
            None => {
                kind.isolated()
                    && elems.iter().all(|elem| match elem {
                        Elem::Pass { kind: pass, .. } => pass.can_schedule_on(kind),
                        Elem::Nested { .. } => true,
                    })
            }
        }
    }

    fn run(&mut self, elems: &[Elem], node: &Node) -> Result<(), ()> {
        for elem in elems {
            // Whatever is run on an operation, a pass or the walk of a nested pipeline, needs the
            // operation to be isolated from above.
            if !node.kind.isolated() {
                return Err(());
            }
            match elem {
                Elem::Pass { id, kind } => {
                    if !kind.can_schedule_on(node.kind) {
                        return Err(());
                    }
                    self.events.push((*id, node.id));
                }
                Elem::Nested {
                    recursive,
                    anchor,
                    elems,
                } => self.walk(*recursive, *anchor, elems, node)?,
            }
        }
        Ok(())
    }

    fn walk(
        &mut self,
        recursive: bool,
        anchor: Option<Kind>,
        elems: &[Elem],
        node: &Node,
    ) -> Result<(), ()> {
        for child in &node.children {
            if Self::schedules(anchor, elems, child.kind) {
                self.run(elems, child)?;
            } else if recursive {
                self.walk(recursive, anchor, elems, child)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------------------------

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

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

fn random_kind(rng: &mut Rng, plain_percent: usize) -> Kind {
    if rng.chance(plain_percent) {
        Kind::Plain
    } else {
        [Kind::A, Kind::B, Kind::C][rng.below(3)]
    }
}

fn random_tree(rng: &mut Rng, next_id: &mut usize, depth: usize, plain: usize) -> Node {
    let id = *next_id;
    *next_id += 1;
    let kind = if depth == 0 {
        random_kind(rng, 0)
    } else {
        random_kind(rng, plain)
    };
    let width = if depth >= 4 { 0 } else { rng.below(4) };
    let children = (0..width).map(|_| random_tree(rng, next_id, depth + 1, plain)).collect();
    Node { id, kind, children }
}

fn random_pass(rng: &mut Rng, anchor: Option<Kind>, nesting: Nesting, next_id: &mut usize) -> Elem {
    let id = *next_id;
    *next_id += 1;
    let kind = match rng.below(10) {
        0..=4 => {
            // With explicit nesting a typed pass of another type is an error when the pipeline
            // is built: keep that rare, so that most pipelines run.
            let target = match anchor {
                Some(anchor) if nesting == Nesting::Explicit && rng.chance(95) => anchor,
                Some(anchor) if rng.chance(40) => anchor,
                _ => [Kind::A, Kind::B, Kind::C][rng.below(3)],
            };
            if target == Kind::Plain {
                PassKind::Any
            } else {
                PassKind::Typed(target)
            }
        }
        5..=7 => PassKind::Any,
        8 => match anchor {
            // On a typed anchor other than `sched_a` this pass fails the check before the run.
            Some(Kind::A) | None => PassKind::OnlyA,
            Some(_) if rng.chance(5) => PassKind::OnlyA,
            Some(_) => PassKind::Any,
        },
        _ => match anchor {
            Some(Kind::A | Kind::B) | None => PassKind::AOrB,
            Some(_) if rng.chance(5) => PassKind::AOrB,
            Some(_) => PassKind::Any,
        },
    };
    Elem::Pass { id, kind }
}

fn random_elems(
    rng: &mut Rng,
    anchor: Option<Kind>,
    nesting: Nesting,
    depth: usize,
    next_id: &mut usize,
    plain: usize,
) -> Vec<Elem> {
    let count = rng.below(5);
    (0..count)
        .map(|_| {
            if depth < 4 && rng.chance(45) {
                let nested = if rng.chance(25) {
                    None
                } else {
                    Some(random_kind(rng, plain))
                };
                Elem::Nested {
                    recursive: rng.chance(30),
                    anchor: nested,
                    elems: random_elems(rng, nested, nesting, depth + 1, next_id, plain),
                }
            } else {
                random_pass(rng, anchor, nesting, next_id)
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------------------------

fn printed(pm: &PassManager) -> String {
    struct Printed<'a>(&'a PassManager);
    impl fmt::Display for Printed<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.print_as_textual_pipeline(f)
        }
    }
    Printed(pm).to_string()
}

fn ancestors(node: &Node, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if out.len() <= node.id {
        out.resize(node.id + 1, Vec::new());
    }
    out[node.id] = path.clone();
    path.push(node.id);
    for child in &node.children {
        ancestors(child, path, out);
    }
    path.pop();
}

#[derive(Default, Debug)]
struct Tally {
    cases: usize,
    build_errors: usize,
    run_errors: usize,
    ran: usize,
    events: usize,
    implicit_nests: usize,
}

fn one_case(rng: &mut Rng, nesting: Nesting, plain: usize, tally: &mut Tally, label: &str) {
    tally.cases += 1;
    let mut next_node = 0;
    let tree = random_tree(rng, &mut next_node, 0, plain);
    let anchor = match rng.below(10) {
        0 => None,
        1 => Some(random_kind(rng, 0)),
        _ => Some(tree.kind),
    };
    let mut next_pass = 0;
    let elems = random_elems(rng, anchor, nesting, 0, &mut next_pass, plain);
    let text = pipeline_text(anchor, &elems);
    let context = Rc::new(Context::default());
    let mut builder = OpBuilder::new(context.clone());
    let mut ops = Vec::new();
    build(&tree, &mut builder, &mut ops);
    let root = ops[0].0;
    let describe = || format!("{label}: {nesting:?} {text}\n{tree:?}");

    // Built through the API and through the text: both are the resolved pipeline, or both fail.
    let resolved = resolve(anchor, nesting, &elems);
    let mut by_api = PassManager::new(context.clone(), anchor_text(anchor), nesting).unwrap();
    let api_result = add(by_api.op_pass_manager_mut(), &elems);
    let parsed = PassPipeline::parse(&text).unwrap_or_else(|err| panic!("{}: {err}", describe()));
    assert_eq!(parsed.to_string(), text, "{}", describe());
    let by_text = parsed.load(context.clone(), nesting);
    let Ok(resolved) = resolved else {
        tally.build_errors += 1;
        assert!(api_result.is_err(), "expected add_pass to fail: {}", describe());
        assert!(by_text.is_err(), "expected load to fail: {}", describe());
        return;
    };
    api_result.unwrap_or_else(|err| panic!("{}: {err}", describe()));
    let mut by_text = by_text.unwrap_or_else(|err| panic!("{}: {err}", describe()));
    let resolved_text = pipeline_text(anchor, &resolved);
    if resolved_text != text {
        tally.implicit_nests += 1;
    }
    assert_eq!(printed(&by_api), resolved_text, "{}", describe());
    assert_eq!(printed(&by_text), resolved_text, "{}", describe());
    // The text a manager prints loads to a manager that prints it.
    let again = PassPipeline::parse(&resolved_text).unwrap().load(context.clone(), nesting);
    assert_eq!(printed(&again.unwrap()), resolved_text, "{}", describe());

    // What must run.
    let mut model = Model { events: Vec::new() };
    let expected = (|| {
        if anchor.is_some_and(|anchor| anchor != tree.kind) {
            return Err(());
        }
        Model::check(anchor, &resolved)?;
        model.run(&resolved, &tree)
    })();

    let mut paths = Vec::new();
    ancestors(&tree, &mut Vec::new(), &mut paths);
    let related = |a: usize, b: usize| a == b || paths[a].contains(&b) || paths[b].contains(&a);
    let id_of = |op: OperationRef| ops.iter().find(|(o, _)| *o == op).unwrap().1;

    let mut observed = Vec::new();
    for (how, pm) in [("api", &mut by_api), ("text", &mut by_text)] {
        pm.enable_verifier(rng.chance(50));
        EVENTS.with(|events| events.borrow_mut().clear());
        let result = pm.run(root);
        let events = EVENTS
            .with(|events| events.take())
            .into_iter()
            .map(|(pass, op)| (pass, id_of(op)))
            .collect::<Vec<_>>();
        match (&expected, &result) {
            (Err(()), Err(_)) => (),
            (Ok(()), Ok(())) => {
                let mut sorted = events.clone();
                sorted.sort();
                let mut wanted = model.events.clone();
                wanted.sort();
                assert_eq!(sorted, wanted, "{how}: {}", describe());
                let position =
                    |event: &(usize, usize)| model.events.iter().position(|e| e == event).unwrap();
                for (i, first) in events.iter().enumerate() {
                    for second in &events[i + 1..] {
                        if related(first.1, second.1) {
                            assert!(
                                position(first) < position(second),
                                "{how}: pass {} on op {} ran before pass {} on op {}, against the \
                                 order written\n{}\nobserved {events:?}\nexpected {:?}",
                                first.0,
                                first.1,
                                second.0,
                                second.1,
                                describe(),
                                model.events
                            );
                        }
                    }
                }
                // After a run the manager prints text that still loads and prints itself.
                let after = printed(pm);
                let reloaded = PassPipeline::parse(&after)
                    .unwrap_or_else(|err| panic!("{after}: {err}"))
                    .load(context.clone(), nesting)
                    .unwrap_or_else(|err| panic!("{after}: {err}"));
                assert_eq!(printed(&reloaded), after, "{}", describe());
            }
            (Ok(()), Err(err)) => {
                panic!("{how}: expected the run to succeed: {err}\n{}", describe())
            }
            (Err(()), Ok(())) => {
                panic!("{how}: expected the run to fail, ran {events:?}\n{}", describe())
            }
        }
        observed.push(events);
    }
    assert_eq!(observed[0], observed[1], "api and text ran differently: {}", describe());
    match expected {
        Ok(()) => {
            tally.ran += 1;
            tally.events += model.events.len();
        }
        Err(()) => tally.run_errors += 1,
    }
    drop(by_api);
    drop(by_text);
}

/// The cases per seed, for each nesting and each share of operations that are not isolated:
/// chosen so that the test runs in a few seconds in an unoptimized build.
const CASES: usize = 250;

/// With either nesting, through the API and through text, the passes that run, the operations
/// they run on and the order of the runs on one operation and on operations inside one another
/// are those the pipeline states. Seeds: `CORE_TEST_SEED` replaces the fixed ones.
#[test]
fn scheduling_agrees_with_the_statement_of_the_pipeline() {
    register();
    for nesting in [Nesting::Implicit, Nesting::Explicit] {
        for plain in [0, 12] {
            let mut tally = Tally::default();
            for seed in crate::testing::test_seeds(4) {
                let mut rng = Rng(seed ^ ((plain as u64) << 32));
                for case in 0..CASES {
                    let label =
                        format!("seed {seed:#x} (CORE_TEST_SEED) plain {plain} case {case}");
                    one_case(&mut rng, nesting, plain, &mut tally, &label);
                }
            }
            std::eprintln!("{nesting:?} plain={plain}: {tally:?}");
            assert!(tally.ran * 4 > tally.cases, "{tally:?}");
            assert!(tally.events > tally.cases, "{tally:?}");
        }
    }
}

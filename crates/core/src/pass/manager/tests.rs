use alloc::{boxed::Box, format, rc::Rc, string::String, vec::Vec};
use core::cell::RefCell;

use super::*;
use crate::{
    RegionKindInterface, Symbol,
    dialects::{
        builtin::{Function, Module, World},
        test::TestDialect,
    },
    parse::{ParserConfig, parse_any},
    pass::PassTarget,
    traits::{HasOnlyGraphRegion, NoTerminator},
};

type Events = Rc<RefCell<Vec<String>>>;

const TREE: &str = r#"builtin.world {
    builtin.function public extern("C") @root() { builtin.ret; };
    builtin.module public @outer {
        builtin.function public extern("C") @first() { builtin.ret; };
        builtin.module public @inner {
            builtin.function public extern("C") @deep() { builtin.ret; };
        };
        builtin.function public extern("C") @last() { builtin.ret; };
    };
};"#;

fn tree() -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    let world = parse_any(ParserConfig::new(context.clone()), "scheduling.hir", TREE)
        .expect("valid namespace tree");
    (context, world)
}

struct RecordFunctions {
    stage: &'static str,
    events: Events,
}

impl Pass for RecordFunctions {
    type Target = Function;

    fn name(&self) -> &'static str {
        self.stage
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Function>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        self.events.borrow_mut().push(format!("{}:{}", self.stage, Symbol::name(&*op)));
        state.set_post_pass_status(PostPassStatus::Unchanged);
        Ok(())
    }
}

fn record(stage: &'static str, events: &Events) -> Box<dyn OperationPass> {
    Box::new(RecordFunctions {
        stage,
        events: events.clone(),
    })
}

#[test]
fn incompatible_adjacent_adaptors_both_execute() {
    let (context, world) = tree();
    let events = Events::default();
    let mut pm = PassManager::on::<World>(context, Nesting::Explicit);
    pm.nest::<Function>().add_pass(record("typed", &events)).unwrap();
    pm.nest_any().add_pass(record("any", &events)).unwrap();
    pm.run(world).unwrap();
    assert_eq!(&*events.borrow(), &["typed:root", "any:root"]);
}

#[test]
fn trailing_adaptor_finalizes_its_nested_pipeline() {
    let (context, world) = tree();
    let events = Events::default();
    let mut pm = PassManager::on::<World>(context, Nesting::Explicit);
    {
        let mut modules = pm.nest::<Module>();
        modules.nest::<Function>().add_pass(record("a", &events)).unwrap();
        modules.nest::<Function>().add_pass(record("b", &events)).unwrap();
    }
    pm.run(world).unwrap();
    assert_eq!(&*events.borrow(), &["a:first", "b:first", "a:last", "b:last"]);
}

#[test]
fn immediate_and_implicit_nesting_stay_immediate() {
    for implicit in [false, true] {
        let (context, world) = tree();
        let events = Events::default();
        let mut pm = PassManager::on::<World>(context, Nesting::Implicit);
        if implicit {
            pm.add_pass(record("visit", &events)).unwrap();
        } else {
            pm.nest::<Function>().add_pass(record("visit", &events)).unwrap();
        }
        pm.run(world).unwrap();
        assert_eq!(&*events.borrow(), &["visit:root"]);
    }
}

#[test]
fn generic_before_typed_adaptors_keep_both_stages() {
    let (context, world) = tree();
    let events = Events::default();
    let mut pm = PassManager::on::<World>(context, Nesting::Explicit);
    pm.nest_any().add_pass(record("any", &events)).unwrap();
    pm.nest::<Function>().add_pass(record("typed", &events)).unwrap();
    pm.run(world).unwrap();
    assert_eq!(&*events.borrow(), &["any:root", "typed:root"]);
}

#[crate::derive::operation_trait]
trait ValidChildren {
    #[verifier]
    fn verify_children(op: &Operation, _context: &Context) -> Result<(), Report> {
        for region in op.regions().iter() {
            for block in region.body().iter() {
                if block.body().iter().any(|child| child.has_attribute("invalid")) {
                    return Err(Report::msg("container forbids invalid children"));
                }
            }
        }
        Ok(())
    }
}

#[crate::derive::operation(
    dialect = TestDialect,
    traits(NoTerminator, HasOnlyGraphRegion, IsolatedFromAbove, ValidChildren),
    implements(RegionKindInterface)
)]
pub struct RegionPair {
    #[region]
    first: crate::RegionRef,
    #[region]
    second: crate::RegionRef,
}

impl crate::RegionKindInterface for RegionPair {
    fn kind(&self) -> crate::RegionKind {
        crate::RegionKind::Graph
    }
}

#[crate::derive::operation(dialect = TestDialect, traits(IsolatedFromAbove))]
pub struct First {}
#[crate::derive::operation(dialect = TestDialect, traits(IsolatedFromAbove))]
pub struct Second {}

struct RecordOps(Events);
impl Pass for RecordOps {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "record-ops"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Operation>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        self.0.borrow_mut().push(op.name().to_string());
        state.set_post_pass_status(PostPassStatus::Unchanged);
        Ok(())
    }
}

#[test]
fn immediate_scheduling_visits_all_regions_in_order() {
    use crate::{Builder, BuilderExt, OpBuilder};
    let context = Rc::new(Context::default());
    let mut builder = OpBuilder::new(context.clone());
    let pair = builder.create::<RegionPair, ()>(crate::SourceSpan::UNKNOWN)().unwrap();
    let regions = pair
        .as_operation_ref()
        .borrow()
        .regions()
        .iter()
        .map(|r| r.as_region_ref())
        .collect::<Vec<_>>();
    assert_eq!(regions.len(), 2);
    builder.create_block(regions[0], None, &[]);
    builder.create::<First, ()>(crate::SourceSpan::UNKNOWN)().unwrap();
    builder.create_block(regions[1], None, &[]);
    builder.create::<Second, ()>(crate::SourceSpan::UNKNOWN)().unwrap();
    let events = Events::default();
    let mut pm = PassManager::on::<RegionPair>(context, Nesting::Explicit);
    pm.nest::<First>().add_pass(Box::new(RecordOps(events.clone()))).unwrap();
    pm.nest::<Second>().add_pass(Box::new(RecordOps(events.clone()))).unwrap();
    pm.run(pair.as_operation_ref()).unwrap();
    assert_eq!(&*events.borrow(), &["test.first", "test.second"]);
}

#[test]
fn recursive_stages_visit_each_function_once_and_keep_stage_order() {
    let (context, world) = tree();
    let events = Events::default();
    let mut pm = PassManager::on::<World>(context, Nesting::Explicit);
    pm.nest_recursive::<Function>().add_pass(record("a", &events)).unwrap();
    pm.nest::<Function>().add_pass(record("immediate", &events)).unwrap();
    pm.nest_recursive::<Function>().add_pass(record("b", &events)).unwrap();
    pm.run(world).unwrap();
    assert_eq!(
        &*events.borrow(),
        &[
            "a:root",
            "a:first",
            "a:deep",
            "a:last",
            "immediate:root",
            "b:root",
            "b:first",
            "b:deep",
            "b:last",
        ]
    );
}

#[test]
fn recursive_matches_delegate_their_subtree_to_the_nested_pipeline() {
    for recurse_inside in [false, true] {
        let (context, world) = tree();
        let events = Events::default();
        let mut pm = PassManager::on::<World>(context, Nesting::Explicit);
        {
            let mut modules = pm.nest_recursive::<Module>();
            if recurse_inside {
                modules.nest_recursive::<Function>().add_pass(record("visit", &events)).unwrap();
            } else {
                modules.nest::<Function>().add_pass(record("visit", &events)).unwrap();
            }
        }
        pm.run(world).unwrap();
        let expected = if recurse_inside {
            &["visit:first", "visit:deep", "visit:last"][..]
        } else {
            &["visit:first", "visit:last"][..]
        };
        assert_eq!(&*events.borrow(), expected);
    }
}

#[test]
fn recursive_scheduling_excludes_its_root() {
    let (context, world) = tree();
    let events = Events::default();
    let mut pm = PassManager::on::<World>(context, Nesting::Explicit);
    pm.nest_recursive::<World>()
        .add_pass(Box::new(RecordOps(events.clone())))
        .unwrap();
    pm.run(world).unwrap();
    assert!(events.borrow().is_empty());
}

#[derive(Default)]
struct OperationIdentity(Option<OperationRef>);
impl crate::pass::Analysis for OperationIdentity {
    type Target = Operation;

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }

    fn as_any_rc(self: Rc<Self>) -> Rc<dyn core::any::Any> {
        self
    }

    fn name(&self) -> &'static str {
        "operation-identity"
    }

    fn analyze(&mut self, op: &Operation, _am: AnalysisManager) -> Result<(), Report> {
        self.0 = Some(op.as_operation_ref());
        Ok(())
    }

    fn invalidate(&self, preserved: &mut crate::pass::PreservedAnalyses) -> bool {
        !preserved.is_preserved::<Self>()
    }
}

struct InspectAnalyses(Events);
impl Pass for InspectAnalyses {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "inspect-analyses"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Operation>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let root = op.as_operation_ref();
        let is_world = op.is::<World>();
        drop(op);
        let am = state.analysis_manager();
        assert_eq!(am.get_analysis::<OperationIdentity>()?.0, Some(root));
        if is_world {
            // Cache analyses at every namespace depth before the recursive function traversal.
            let result = root.raw_prewalk::<Forward, _, _>(|op: OperationRef| {
                if op.borrow().is::<Module>()
                    && let Err(err) = am.get_child_analysis::<OperationIdentity>(op)
                {
                    return WalkResult::Break(err);
                }
                WalkResult::Continue(())
            });
            if let WalkResult::Break(err) = result {
                return Err(err);
            }
        } else {
            let mut next = root.borrow().parent_op();
            let mut count = 0;
            while let Some(parent) = next {
                let cached = am
                    .get_cached_parent_analysis::<OperationIdentity>(parent)
                    .expect("ancestor analysis remains accessible through recursive scheduling");
                assert_eq!(cached.0, Some(parent));
                count += 1;
                next = parent.borrow().parent_op();
            }
            let op = root.borrow();
            let function = op.downcast_ref::<Function>().unwrap();
            self.0.borrow_mut().push(format!("{}:{count}", Symbol::name(function)));
        }
        state.preserved_analyses_mut().preserve_all();
        state.set_post_pass_status(PostPassStatus::Unchanged);
        Ok(())
    }
}

#[test]
fn recursive_scheduling_preserves_analysis_ancestry() {
    let (context, world) = tree();
    let events = Events::default();
    let mut pm = PassManager::on::<World>(context, Nesting::Explicit);
    pm.add_pass(Box::new(InspectAnalyses(events.clone()))).unwrap();
    pm.nest_recursive::<Function>()
        .add_pass(Box::new(InspectAnalyses(events.clone())))
        .unwrap();
    pm.run(world).unwrap();
    assert_eq!(&*events.borrow(), &["root:1", "first:2", "deep:3", "last:2"]);
}

struct FailOnDeep;
impl Pass for FailOnDeep {
    type Target = Function;

    fn name(&self) -> &'static str {
        "fail-on-deep"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Function>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        if Symbol::name(&*op).as_str() == "deep" {
            Err(Report::msg("deep function rejected"))
        } else {
            state.set_post_pass_status(PostPassStatus::Unchanged);
            Ok(())
        }
    }
}

struct ObserveFailure(Events);
impl PassInstrumentation for ObserveFailure {
    fn run_before_pass(&mut self, pass: &dyn OperationPass, op: &OperationRef) {
        if pass.name() == "fail-on-deep" {
            let op = op.borrow();
            self.0
                .borrow_mut()
                .push(format!("before:{}", Symbol::name(op.downcast_ref::<Function>().unwrap())));
        }
    }

    fn run_after_pass_failed(&mut self, pass: &dyn OperationPass, op: &OperationRef) {
        if pass.name() == "fail-on-deep" {
            let op = op.borrow();
            self.0
                .borrow_mut()
                .push(format!("failed:{}", Symbol::name(op.downcast_ref::<Function>().unwrap())));
        }
    }
}

#[test]
fn recursive_failure_is_instrumented_and_stops_siblings_and_later_stages() {
    let (context, world) = tree();
    let events = Events::default();
    let mut pm = PassManager::on::<World>(context, Nesting::Explicit);
    pm.add_instrumentation(Box::new(ObserveFailure(events.clone())));
    pm.nest_recursive::<Function>().add_pass(Box::new(FailOnDeep)).unwrap();
    pm.nest_recursive::<Function>().add_pass(record("later", &events)).unwrap();
    assert!(pm.run(world).unwrap_err().to_string().contains("deep function rejected"));
    assert_eq!(
        &*events.borrow(),
        &["before:root", "before:first", "before:deep", "failed:deep"]
    );
}

#[test]
fn adjacent_recursive_pipelines_do_not_coalesce_into_per_function_order() {
    let (context, world) = tree();
    let events = Events::default();
    let mut pm = PassManager::on::<World>(context, Nesting::Explicit);
    pm.nest_recursive::<Function>().add_pass(record("a", &events)).unwrap();
    pm.nest_recursive::<Function>().add_pass(record("b", &events)).unwrap();
    pm.run(world).unwrap();
    assert_eq!(
        &*events.borrow(),
        &["a:root", "a:first", "a:deep", "a:last", "b:root", "b:first", "b:deep", "b:last",]
    );
}

struct InvalidateParent;
impl Pass for InvalidateParent {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "invalidate-parent"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<First>()
    }

    fn run_on_operation(
        &mut self,
        mut op: EntityMut<'_, Operation>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let marker = op
            .context_rc()
            .create_attribute::<crate::dialects::builtin::attributes::BoolAttr, _>(true);
        op.set_attribute("invalid", marker);
        state.set_post_pass_status(PostPassStatus::Changed);
        Ok(())
    }
}

#[test]
fn recursive_scheduling_reverifies_unmatched_containers_after_child_changes() {
    use crate::{Builder, BuilderExt, OpBuilder};
    let context = Rc::new(Context::default());
    let mut builder = OpBuilder::new(context.clone());
    let root = builder.create::<RegionPair, ()>(crate::SourceSpan::UNKNOWN)().unwrap();
    let region = root.borrow().first().as_region_ref();
    builder.create_block(region, None, &[]);
    let container = builder.create::<RegionPair, ()>(crate::SourceSpan::UNKNOWN)().unwrap();
    let region = container.borrow().first().as_region_ref();
    builder.create_block(region, None, &[]);
    builder.create::<First, ()>(crate::SourceSpan::UNKNOWN)().unwrap();
    root.as_operation_ref().borrow().recursively_verify().unwrap();

    let later = Events::default();
    let mut pm = PassManager::on::<RegionPair>(context, Nesting::Explicit);
    pm.nest_recursive::<First>().add_pass(Box::new(InvalidateParent)).unwrap();
    pm.add_pass(Box::new(RecordOps(later.clone()))).unwrap();
    assert!(
        pm.run(root.as_operation_ref()).is_err(),
        "the intermediate container invariant must be checked"
    );
    assert!(later.borrow().is_empty(), "verification failure must stop later passes");
}

/// A module of two functions, `@a` and `@b`.
const MODULE: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @a() { builtin.ret; };
    builtin.function public extern("C") @b() { builtin.ret; };
};"#;

fn module() -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    let module = parse_any(ParserConfig::new(context.clone()), "module.runic", MODULE)
        .expect("valid module");
    (context, module)
}

/// The target of a pass is the name of its operation type, and nothing for a pass on any
/// operation.
#[test]
fn a_typed_pass_names_its_target() {
    let context = Context::default();
    let events = Events::default();
    assert_eq!(
        record("typed", &events).target_name(&context),
        Some(context.get_registered_name::<Function>())
    );
    let any: Box<dyn OperationPass> = Box::new(RecordOps(events));
    assert_eq!(any.target_name(&context), None);
    assert_eq!(<Operation as PassTarget>::target_name(&context), None);
    assert_eq!(<dyn crate::Op as PassTarget>::target_name(&context), None);
    assert_eq!(<dyn Symbol as PassTarget>::target_name(&context), None);
    assert_eq!(<dyn crate::SymbolTable as PassTarget>::target_name(&context), None);
    assert_eq!(
        <dyn crate::traits::BranchOpInterface as PassTarget>::target_name(&context),
        None
    );
}

/// With implicit nesting, a pass on functions added to a manager on a module is nested under a
/// manager for functions: it runs once per function, and never on the module.
#[test]
fn a_typed_pass_is_nested_implicitly_under_its_operation() {
    let (context, module) = module();
    let events = Events::default();
    let mut pm = PassManager::on::<Module>(context, Nesting::Implicit);
    pm.add_pass(record("visit", &events)).unwrap();
    pm.run(module).unwrap();
    assert_eq!(&*events.borrow(), &["visit:a", "visit:b"]);
}

/// With explicit nesting, the same is an error at `add_pass` that names the pass, its target and
/// the anchor; nested under `builtin.function` by name, the pass runs.
#[test]
fn a_typed_pass_under_another_anchor_with_explicit_nesting_is_an_error() {
    let (context, module) = module();
    let events = Events::default();
    let mut pm = PassManager::on::<Module>(context, Nesting::Explicit);
    let message = pm.add_pass(record("visit", &events)).unwrap_err().to_string();
    assert_eq!(
        message,
        "cannot add pass 'visit', which runs on 'builtin.function', to a pass manager anchored on \
         'builtin.module': nest it under 'builtin.function', or use implicit nesting"
    );
    pm.nest_with_type("builtin.function")
        .unwrap()
        .add_pass(record("visit", &events))
        .unwrap();
    pm.run(module).unwrap();
    assert_eq!(&*events.borrow(), &["visit:a", "visit:b"]);
}

/// A pass registered by name is checked by `add_pass`, with the same error; a pass whose options
/// are invalid is not added.
#[test]
fn a_registered_pass_is_added_through_add_pass() {
    #[derive(Default)]
    struct OnFunctions;
    impl Pass for OnFunctions {
        type Target = Function;

        fn name(&self) -> &'static str {
            "on-functions"
        }

        fn can_schedule_on(&self, name: &OperationName) -> bool {
            name.is::<Function>()
        }

        fn run_on_operation(
            &mut self,
            _op: EntityMut<'_, Function>,
            state: &mut PassExecutionState,
        ) -> Result<(), Report> {
            state.set_post_pass_status(PostPassStatus::Unchanged);
            Ok(())
        }
    }

    let context = Rc::new(Context::default());
    let mut explicit = OpPassManager::on::<Module>(Nesting::Explicit, context.clone());
    let message = crate::pass::registry::default_registration::<OnFunctions>(&mut explicit, "")
        .unwrap_err()
        .to_string();
    assert!(
        message.starts_with("cannot add pass 'on-functions', which runs on"),
        "{message}"
    );
    assert!(explicit.is_empty());

    let factory = crate::pass::registry::default_registration_factory(|| Box::new(OnFunctions));
    let mut implicit = OpPassManager::on::<Module>(Nesting::Implicit, context);
    let message = factory(&mut implicit, "x=1").unwrap_err().to_string();
    assert_eq!(message, "pass 'on-functions' takes no options");
    assert!(implicit.is_empty());
    factory(&mut implicit, "").unwrap();
    assert_eq!(implicit.len(), 1);
}

/// A manager for any operation runs a typed pass only on operations of its type.
#[test]
fn any_runs_a_typed_pass_only_on_its_type() {
    let (context, world) = tree();
    let events = Events::default();
    let mut pm = PassManager::on::<World>(context, Nesting::Implicit);
    pm.nest_any().add_pass(record("any", &events)).unwrap();
    {
        let mut modules = pm.nest::<Module>();
        modules.nest_any().add_pass(record("inner", &events)).unwrap();
    }
    pm.run(world).unwrap();
    assert_eq!(&*events.borrow(), &["any:root", "inner:first", "inner:last"]);
}

/// Two typed passes for the same type added in a row share one nested manager: each function
/// sees both before the next function is visited.
#[test]
fn adjacent_typed_passes_share_one_nested_manager() {
    let (context, module) = module();
    let events = Events::default();
    let mut pm = PassManager::on::<Module>(context, Nesting::Implicit);
    pm.add_pass(record("a", &events)).unwrap();
    pm.add_pass(record("b", &events)).unwrap();
    pm.run(module).unwrap();
    assert_eq!(&*events.borrow(), &["a:a", "b:a", "a:b", "b:b"]);
    let passes = pm.op_pass_manager_mut().passes();
    assert_eq!(passes.len(), 1);
    let adaptor = passes[0].as_any().downcast_ref::<OpToOpPassAdaptor>().unwrap();
    assert_eq!(adaptor.pass_managers().len(), 1);
    assert_eq!(adaptor.pass_managers()[0].len(), 2);
}

/// An anchor that is not `any` and names no registered operation is an error, for a pass manager
/// and for a nested one.
#[test]
fn an_unknown_anchor_is_an_error() {
    let context = Rc::new(Context::default());
    for (anchor, expected) in [
        (
            "function",
            "invalid pass manager anchor 'function': expected 'any' or an operation name of the \
             form '<dialect>.<opcode>'",
        ),
        (
            "builtin.funktion",
            "invalid pass manager anchor 'builtin.funktion': dialect 'builtin' has no operation \
             'funktion'",
        ),
        (
            "nowhere.function",
            "invalid pass manager anchor 'nowhere.function': no dialect 'nowhere' is registered",
        ),
    ] {
        let message = PassManager::new(context.clone(), anchor, Nesting::Implicit)
            .err()
            .expect("an unknown anchor is an error")
            .to_string();
        assert_eq!(message, expected);
        let message = OpPassManager::new(anchor, Nesting::Implicit, context.clone())
            .err()
            .expect("an unknown anchor is an error")
            .to_string();
        assert_eq!(message, expected);
        let mut pm = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
        let message = pm
            .nest_with_type(anchor)
            .err()
            .expect("an unknown anchor is an error")
            .to_string();
        assert_eq!(message, expected);
    }
    assert!(PassManager::new(context.clone(), "any", Nesting::Implicit).is_ok());
    assert!(PassManager::new(context, "builtin.module", Nesting::Implicit).is_ok());
}

/// A typed pass whose `can_schedule_on` answers true for every operation, put by hand in a
/// manager for any operation, is run on the module: an error that names the pass and both
/// operations, not a panic.
#[test]
fn a_typed_pass_run_on_another_operation_is_an_error() {
    struct Careless;
    impl Pass for Careless {
        type Target = Function;

        fn name(&self) -> &'static str {
            "careless"
        }

        fn can_schedule_on(&self, _name: &OperationName) -> bool {
            true
        }

        fn run_on_operation(
            &mut self,
            _op: EntityMut<'_, Function>,
            state: &mut PassExecutionState,
        ) -> Result<(), Report> {
            state.set_post_pass_status(PostPassStatus::Unchanged);
            Ok(())
        }
    }

    let (context, world) = tree();
    let mut pm = PassManager::on::<World>(context.clone(), Nesting::Explicit);
    let mut any = OpPassManager::any(Nesting::Explicit, context);
    any.add_pass(Box::new(Careless)).unwrap();
    pm.nest::<Module>().nest_pass_manager(any);
    let message = pm.run(world).unwrap_err().to_string();
    assert_eq!(
        message,
        "pass 'careless' cannot run on this operation: expected 'builtin.function', but found \
         'builtin.module'"
    );
}

/// A dialect that is not submitted to `inventory`, so a context does not register it until it is
/// asked for.
#[derive(Debug)]
pub struct UnloadedDialect {
    info: crate::DialectInfo,
}
impl crate::Dialect for UnloadedDialect {
    fn info(&self) -> &crate::DialectInfo {
        &self.info
    }
}
impl crate::DialectRegistration for UnloadedDialect {
    const NAMESPACE: &'static str = "unloaded";

    fn init(info: crate::DialectInfo) -> Self {
        Self { info }
    }
}

#[crate::derive::operation(dialect = UnloadedDialect, traits(IsolatedFromAbove))]
pub struct Unloaded {}

/// A pass on an operation whose dialect the context has not registered: asking for its target
/// registers the dialect, as building such an operation does, and the pass is nested under it
/// (and so never runs on a module that has none).
#[test]
fn a_typed_pass_on_an_unregistered_dialect_registers_it() {
    struct OnUnloaded(Events);
    impl Pass for OnUnloaded {
        type Target = Unloaded;

        fn name(&self) -> &'static str {
            "on-unloaded"
        }

        fn can_schedule_on(&self, name: &OperationName) -> bool {
            name.is::<Unloaded>()
        }

        fn run_on_operation(
            &mut self,
            _op: EntityMut<'_, Unloaded>,
            state: &mut PassExecutionState,
        ) -> Result<(), Report> {
            self.0.borrow_mut().push("unloaded".into());
            state.set_post_pass_status(PostPassStatus::Unchanged);
            Ok(())
        }
    }

    let (context, module) = module();
    let unloaded = crate::interner::Symbol::intern("unloaded");
    assert!(context.find_registered_dialect(unloaded).is_none());
    let events = Events::default();
    let mut pm = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
    pm.add_pass(Box::new(OnUnloaded(events.clone()))).unwrap();
    assert!(context.find_registered_dialect(unloaded).is_some());
    pm.run(module).unwrap();
    assert!(events.borrow().is_empty());

    let context = Rc::new(Context::default());
    let pm = OpPassManager::on::<Unloaded>(Nesting::Explicit, context.clone());
    assert_eq!(pm.name().map(|name| name.to_string()).as_deref(), Some("unloaded.unloaded"));
}

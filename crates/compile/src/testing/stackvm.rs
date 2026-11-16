//! The `stackvm` and `altvm` machines and their backends.
//!
//! `stackvm` is a stack machine with the instructions `load N` (parameter `N`), `push V`, `add`,
//! `sub`, `mul`, `call MODULE::NAME` and `ret`, for the targets `stackvm-unknown-unknown` and
//! `stackvm-unknown-tiny`. `altvm` shares the first of them and lists only each function and its
//! arity.

use core::fmt;
use std::{io, rc::Rc};

use runic_core::{
    Builder, BuilderExt, Context, DataLayoutSpec, Endianness, EntityMut, Op, OpBuilder, Operation,
    OperationName, OperationRef, Overflow, PointerSpec, Report, Signedness, Symbol, Usable, Value,
    ValueRef,
    attributes::IntegerLikeAttr,
    dialects::builtin::{Call, Function, Ret, WorldRef},
    matchers::{self, Matcher},
    pass::{
        OpPassManager, Pass, PassExecutionState, PassInfo, PassOptions, PostPassStatus,
        registry::RegistryEntry,
    },
};
use runic_dialect_arith::{Add, ArithOpBuilder, Constant, Mul, Sub};

use crate::{
    Artifact, Backend, CheckpointDecl, CheckpointId, Checkpoints, Flow, Render, Session, Triple,
};

const UNKNOWN: &str = "stackvm-unknown-unknown";
const TINY: &str = "stackvm-unknown-tiny";

/// The name of the lowering pass of the `stackvm` backend, as pipeline text writes it.
const LEGALIZE: &str = "stackvm-legalize";

const ASM: CheckpointId = CheckpointId::new("stackvm.asm");
const OUT: CheckpointId = CheckpointId::new("altvm.out");

const STACKVM_CHECKPOINTS: &[CheckpointDecl] = &[CheckpointDecl {
    id: ASM,
    aliases: &["asm"],
    extension: "svm",
    render: render_listing as Render,
}];

const ALTVM_CHECKPOINTS: &[CheckpointDecl] = &[CheckpointDecl {
    id: OUT,
    aliases: &[],
    extension: "alt",
    render: render_listing as Render,
}];

fn data_layout() -> Result<DataLayoutSpec, Report> {
    let pointer = PointerSpec::sized(32).ok_or_else(|| Report::msg("32 is not a pointer size"))?;
    DataLayoutSpec::new(Endianness::Little, pointer)
        .map_err(|error| Report::msg(format!("the data layout is not valid: {error}")))
}

fn render_listing(
    artifact: &Artifact,
    _session: &Session<'_>,
    out: &mut dyn io::Write,
) -> Result<(), Report> {
    let checkpoint = artifact.checkpoint();
    let Some(listing) = artifact.downcast_ref::<String>() else {
        return Err(Report::msg(format!(
            "the artifact at checkpoint '{checkpoint}' is not a listing"
        )));
    };
    out.write_all(listing.as_bytes()).map_err(|error| {
        Report::msg(format!("could not render checkpoint '{checkpoint}': {error}"))
    })
}

/// The backend for the `stackvm` machine.
///
/// It lowers the world with the pass `stackvm-legalize`, which rejects what the machine cannot run
/// and turns the subtraction of a constant into the addition of its negation, and then lists each
/// function as instructions at `stackvm.asm` (also `asm`, in files with the extension `svm`): a
/// line `MODULE::NAME:` and then one instruction per line, indented by two spaces, ending in
/// `ret`. The instructions of a function are those of the value it returns, operands first.
///
/// ```text
/// demo::inc:
///   load 0
///   push 1
///   add
///   ret
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct StackVmBackend;

impl Backend for StackVmBackend {
    fn name(&self) -> &'static str {
        "stackvm"
    }

    fn supports(&self, triple: &Triple) -> bool {
        matches!(triple.as_str(), UNKNOWN | TINY)
    }

    fn targets(&self) -> Vec<Triple> {
        [UNKNOWN, TINY]
            .into_iter()
            .map(|triple| triple.parse().expect("a triple of the stack machine is well formed"))
            .collect()
    }

    fn data_layout(&self, _triple: &Triple) -> Result<DataLayoutSpec, Report> {
        data_layout()
    }

    fn checkpoints(&self) -> &'static [CheckpointDecl] {
        STACKVM_CHECKPOINTS
    }

    fn add_passes(&self, _session: &Session<'_>, passes: &mut OpPassManager) -> Result<(), Report> {
        let info = PassInfo::lookup(LEGALIZE)
            .ok_or_else(|| Report::msg(format!("the pass '{LEGALIZE}' is not registered")))?;
        info.add_to_pipeline(passes, &PassOptions::new())
    }

    fn emit(
        &self,
        _session: &Session<'_>,
        world: WorldRef,
        checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        let mut listing = String::new();
        for function in functions(world) {
            list(&function, &mut listing)?;
        }
        Ok(match checkpoints.reach(ASM, listing)? {
            Flow::Continue(_) => Flow::Continue(()),
            Flow::Break(stopped) => Flow::Break(stopped),
        })
    }
}

/// The backend for the `altvm` machine, which exists so that one triple has two backends.
///
/// It lowers nothing and lists, at `altvm.out`, one line per function: `MODULE::NAME/ARITY`.
#[derive(Debug, Default, Clone, Copy)]
pub struct AltVmBackend;

impl Backend for AltVmBackend {
    fn name(&self) -> &'static str {
        "altvm"
    }

    fn supports(&self, triple: &Triple) -> bool {
        triple.as_str() == UNKNOWN
    }

    fn targets(&self) -> Vec<Triple> {
        vec![UNKNOWN.parse().expect("a triple of the stack machine is well formed")]
    }

    fn data_layout(&self, _triple: &Triple) -> Result<DataLayoutSpec, Report> {
        data_layout()
    }

    fn checkpoints(&self) -> &'static [CheckpointDecl] {
        ALTVM_CHECKPOINTS
    }

    fn add_passes(
        &self,
        _session: &Session<'_>,
        _passes: &mut OpPassManager,
    ) -> Result<(), Report> {
        Ok(())
    }

    fn emit(
        &self,
        _session: &Session<'_>,
        world: WorldRef,
        checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        let mut listing = String::new();
        for function in functions(world) {
            let (name, arity) = {
                let operation = function.borrow();
                let arity = operation
                    .downcast_ref::<Function>()
                    .map_or(0, |function| function.get_signature().arity());
                (qualified_name(&operation), arity)
            };
            listing.push_str(&format!("{name}/{arity}\n"));
        }
        Ok(match checkpoints.reach(OUT, listing)? {
            Flow::Continue(_) => Flow::Continue(()),
            Flow::Break(stopped) => Flow::Break(stopped),
        })
    }
}

/// The functions of `world` that have a body, in the order the world holds them.
fn functions(world: WorldRef) -> Vec<OperationRef> {
    let mut found = Vec::new();
    world.borrow().as_operation().prewalk_all(|op: &Operation| {
        if op
            .downcast_ref::<Function>()
            .is_some_and(|function| !function.body().is_empty())
        {
            found.push(op.as_operation_ref());
        }
    });
    found
}

/// `MODULE::NAME` of the symbol `op`, where `MODULE` is the name of the module that holds it.
fn qualified_name(op: &Operation) -> String {
    let name = op
        .as_symbol()
        .map_or_else(|| op.name().to_string(), |symbol| symbol.name().to_string());
    match op.parent_op().and_then(|parent| parent.borrow().as_symbol().map(Symbol::name)) {
        Some(module) => format!("{module}::{name}"),
        None => name,
    }
}

/// Appends the lines of the listing of `function`.
fn list(function: &OperationRef, listing: &mut String) -> Result<(), Report> {
    let function = function.borrow();
    let name = qualified_name(&function);
    let function = function.downcast_ref::<Function>().expect("only functions are listed");
    listing.push_str(&format!("{name}:\n"));
    let entry = function.entry_block();
    let entry = entry.borrow();
    let terminator = entry.terminator().ok_or_else(|| {
        Report::msg(format!("the function '{name}' has no terminator, so it cannot be listed"))
    })?;
    let returned = {
        let terminator = terminator.borrow();
        match (terminator.downcast_ref::<Ret>(), terminator.num_operands()) {
            (Some(_), 1) => terminator.operands()[0].borrow().as_value_ref(),
            _ => {
                return Err(Report::msg(format!(
                    "the function '{name}' does not end by returning one value, so it cannot be \
                     listed"
                )));
            }
        }
    };
    instructions(&name, returned, listing)?;
    listing.push_str("  ret\n");
    Ok(())
}

/// Appends the instructions that leave `value` on the stack, operands first.
fn instructions(function: &str, value: ValueRef, listing: &mut String) -> Result<(), Report> {
    let defining = value.borrow().get_defining_op();
    let Some(defining) = defining else {
        // A block argument is a parameter.
        let index = value
            .borrow()
            .downcast_ref::<runic_core::BlockArgument>()
            .map(|argument| argument.index())
            .ok_or_else(|| {
                Report::msg(format!(
                    "the function '{function}' uses a value that is neither a parameter nor the \
                     result of an operation"
                ))
            })?;
        listing.push_str(&format!("  load {index}\n"));
        return Ok(());
    };
    let op = defining.borrow();
    if op.is::<Constant>() {
        let value = constant_value(&op).ok_or_else(|| {
            Report::msg(format!(
                "the function '{function}' holds a constant that is not an integer"
            ))
        })?;
        listing.push_str(&format!("  push {value}\n"));
        return Ok(());
    }
    let instruction = if op.is::<Add>() {
        "add"
    } else if op.is::<Sub>() {
        "sub"
    } else if op.is::<Mul>() {
        "mul"
    } else if let Some(call) = op.downcast_ref::<Call>() {
        let callee = call.callee().resolve_callable().map_err(|error| {
            Report::msg(format!(
                "the function '{function}' calls a function that is not defined: {error}"
            ))
        })?;
        for operand in op.operands().iter() {
            instructions(function, operand.borrow().as_value_ref(), listing)?;
        }
        let callee = callee.target().as_operation_ref();
        listing.push_str(&format!("  call {}\n", qualified_name(&callee.borrow())));
        return Ok(());
    } else {
        return Err(Report::msg(format!(
            "the function '{function}' holds the operation '{}', which the machine cannot run",
            op.name()
        )));
    };
    for operand in op.operands().iter() {
        instructions(function, operand.borrow().as_value_ref(), listing)?;
    }
    listing.push_str(&format!("  {instruction}\n"));
    Ok(())
}

/// The value of the integer constant `op` as a signed number.
fn constant_value(op: &Operation) -> Option<i128> {
    let attr = matchers::constant().matches(op)?;
    let attr = attr.borrow();
    Some(attr.as_attr().as_trait::<dyn IntegerLikeAttr>()?.as_int().signed())
}

/// `stackvm-legalize`: fails on an operation in a function that the machine cannot run, and
/// rewrites a subtraction of a constant into the addition of the negated constant.
#[derive(Default)]
struct Legalize;

runic_core::inventory::submit!(PassInfo::new::<Legalize>(
    LEGALIZE,
    "Rejects what the stack machine cannot run and subtracts constants by adding their negations"
));

impl Pass for Legalize {
    type Target = Operation;

    fn name(&self) -> &'static str {
        LEGALIZE
    }

    fn argument(&self) -> &'static str {
        LEGALIZE
    }

    fn description(&self) -> &'static str {
        "Rejects what the stack machine cannot run and subtracts constants by adding their \
         negations"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Operation>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let context = op.context_rc();
        let world = op.as_operation_ref();
        drop(op);

        // Each operation in the body of a function, and the subtractions among them.
        let mut subtractions = Vec::new();
        let mut rejected = None;
        world.borrow().prewalk_all(|op: &Operation| {
            let Some(function) = op.downcast_ref::<Function>() else {
                return;
            };
            if function.body().is_empty() {
                return;
            }
            let name = qualified_name(op);
            function.body().as_region_ref().borrow().prewalk_all(|op: &Operation| {
                if op.is::<Sub>() {
                    subtractions.push(op.as_operation_ref());
                }
                let runs = op.is::<Constant>()
                    || op.is::<Add>()
                    || op.is::<Sub>()
                    || op.is::<Mul>()
                    || op.is::<Call>()
                    || op.is::<Ret>();
                if !runs && rejected.is_none() {
                    rejected = Some(format!(
                        "the function '{name}' holds the operation '{}', which the stack machine \
                         cannot run",
                        op.name()
                    ));
                }
            });
        });
        if let Some(message) = rejected {
            return Err(Report::msg(message));
        }

        let mut changed = PostPassStatus::Unchanged;
        for subtraction in subtractions {
            if negate_constant_subtraction(&context, subtraction)? {
                changed = PostPassStatus::Changed;
            }
        }
        state.set_post_pass_status(changed);
        if !changed.ir_changed() {
            state.preserved_analyses_mut().preserve_all();
        }
        Ok(())
    }
}

/// Replaces `subtraction` by the addition of the negated constant when its right operand is a
/// constant that can be negated. Returns whether it did.
///
/// Only a subtraction of `i32` values is rewritten, and only when it wraps or reads its operands
/// as signed: `x - c` and `x + (-c)` then overflow for exactly the same `x`. Read as unsigned they
/// do not (`5 - 3` fits, `5 + 4294967293` does not), so such a subtraction is left as it is.
fn negate_constant_subtraction(
    context: &Rc<Context>,
    mut subtraction: OperationRef,
) -> Result<bool, Report> {
    let (span, lhs, constant, overflow, mut old_result) = {
        let op = subtraction.borrow();
        let Some(sub) = op.downcast_ref::<Sub>() else {
            return Ok(false);
        };
        let unsigned = matches!(
            *sub.get_overflow(),
            Overflow::Checked(Signedness::Unsigned) | Overflow::Unchecked(Signedness::Unsigned)
        );
        if unsigned || sub.result().ty() != context.i32() {
            return Ok(false);
        }
        let rhs = sub.rhs().as_value_ref();
        let Some(constant) = rhs.borrow().get_defining_op() else {
            return Ok(false);
        };
        let Some(value) = constant_value(&constant.borrow()) else {
            return Ok(false);
        };
        let Some(negated) = value.checked_neg().filter(|negated| i32::try_from(*negated).is_ok())
        else {
            return Ok(false);
        };
        (
            op.span,
            sub.lhs().as_value_ref(),
            (constant, negated as i32),
            *sub.get_overflow(),
            sub.result().as_value_ref(),
        )
    };
    let (mut old_constant, negated) = constant;
    let mut builder = OpBuilder::new(context.clone());
    builder.set_insertion_point_before(subtraction);
    let negated = ArithOpBuilder::i32(&mut builder, negated, span);
    let sum = {
        let op_builder = builder.create::<Add, (_, _, _)>(span);
        op_builder(lhs, negated, overflow)?.borrow().result().as_value_ref()
    };
    old_result.borrow_mut().replace_all_uses_with(sum);
    subtraction.borrow_mut().erase();
    let unused = old_constant.borrow().results().iter().all(|result| !result.borrow().is_used());
    if unused {
        old_constant.borrow_mut().erase();
    }
    Ok(true)
}

impl fmt::Debug for Legalize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(LEGALIZE)
    }
}

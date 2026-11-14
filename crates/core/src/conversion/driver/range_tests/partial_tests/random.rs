//! Random straight-line programs over `i128` and `i64` test operations, partially converted with a
//! random subset of operation kinds given patterns and the rest left without a rule; an evaluator
//! written here gives every returned value the number it had before.
//!
//! Two families of programs:
//!
//! - **Split**: constants and adds of `i128` and `i64`, joins, lows and highs. The converter splits
//!   an `i128` into two `i64` halves; its source callback builds a `test.join` and its target
//!   callback a `test.low` and a `test.high`. A random subset of the kinds `test.constant`,
//!   `test.join`, `test.low` and `test.high` is given a pattern; adds never are (a split add needs
//!   a carry, which the test operations cannot compute). Every returned value must be equal.
//! - **Narrow**: constants and adds of `i64`. The converter narrows `i64` to `i32`; its target
//!   callback builds a `test.narrow` (the low 32 bits) and its source callback a `test.widen`
//!   (zero extension), two operations declared here. A random subset of the kinds `test.constant`
//!   and `test.add` is given a pattern that retypes the operation in place through
//!   `replace_elements_in`. The low 32 bits of every returned value must agree.
//!
//! The tests run three fixed seeds, or the one in `CONVERSION_TEST_SEED`; a failure prints the
//! seed and the program.

use super::*;
use crate::{
    ElementRewrite, OpPrinter,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    effects::MemoryEffectOpInterface,
    replace_elements_in,
    traits::{InferTypeOpInterface, Int32, Int64},
    types::TypeMapper,
};

/// The low 32 bits of an `i64`: `%r = test.narrow %value;`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Narrow {
    #[operand]
    value: Int64,
    #[result]
    result: Int32,
}

impl InferTypeOpInterface for Narrow {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

/// An `i32` zero-extended to an `i64`: `%r = test.widen %value;`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Widen {
    #[operand]
    value: Int32,
    #[result]
    result: Int64,
}

impl InferTypeOpInterface for Widen {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i64());
        Ok(())
    }
}

/// A splitmix64 generator: the tests are deterministic for a seed.
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

    fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }

    /// A value of `width` bits, often an edge: zero, one, all ones, or a single bit.
    fn value(&mut self, width: u32) -> u128 {
        let bits = match self.below(6) {
            0 => 0,
            1 => 1,
            2 => u128::MAX,
            3 => 1 << self.below(width as usize),
            _ => (self.next() as u128) << 64 | self.next() as u128,
        };
        bits & mask(width)
    }
}

/// The seeds the test runs: `CONVERSION_TEST_SEED` if set, else three fixed ones.
fn seeds() -> Vec<u64> {
    match std::env::var("CONVERSION_TEST_SEED") {
        Ok(seed) => vec![seed.parse().expect("CONVERSION_TEST_SEED is a u64")],
        Err(_) => vec![0xc0de_0001, 0xc0de_0002, 0xc0de_0003],
    }
}

/// The number of programs of each family per seed.
const PROGRAMS: usize = 150;

fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// The types of the programs.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Ty {
    I128,
    I64,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Self::I128 => "i128",
            Self::I64 => "i64",
        }
    }
}

/// One operation of a program; operands are the indices of earlier values.
#[derive(Debug, Copy, Clone)]
enum Step {
    Constant(Ty, u128),
    Add(usize, usize),
    Join(usize, usize),
    Low(usize),
    High(usize),
}

/// The operation kinds of the programs.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Kind {
    Constant,
    Add,
    Join,
    Low,
    High,
}

impl Kind {
    fn of(op: &Operation) -> Option<Self> {
        if op.is::<Constant>() {
            Some(Self::Constant)
        } else if op.is::<Add>() {
            Some(Self::Add)
        } else if op.is::<Join>() {
            Some(Self::Join)
        } else if op.is::<Low>() {
            Some(Self::Low)
        } else if op.is::<High>() {
            Some(Self::High)
        } else {
            None
        }
    }
}

/// A straight-line program: one value per step, and the values it returns.
struct Program {
    steps: Vec<Step>,
    types: Vec<Ty>,
    returned: Vec<usize>,
}

impl Program {
    /// A random program of the split family (`split`) or the narrow family.
    fn random(rng: &mut Rng, split: bool) -> Self {
        let mut program = Program {
            steps: vec![],
            types: vec![],
            returned: vec![],
        };
        let count = 1 + rng.below(12);
        while program.steps.len() < count {
            let of = |ty: Ty| {
                program
                    .types
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| **t == ty)
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>()
            };
            let (wide, narrow) = (of(Ty::I128), of(Ty::I64));
            let pick = |rng: &mut Rng, values: &[usize]| values[rng.below(values.len())];
            let kinds = if split { 7 } else { 2 };
            let (step, ty) = match rng.below(kinds) {
                0 => (Step::Constant(Ty::I64, rng.value(64)), Ty::I64),
                1 if !narrow.is_empty() => {
                    (Step::Add(pick(rng, &narrow), pick(rng, &narrow)), Ty::I64)
                }
                2 => (Step::Constant(Ty::I128, rng.value(128)), Ty::I128),
                3 if !wide.is_empty() => (Step::Add(pick(rng, &wide), pick(rng, &wide)), Ty::I128),
                4 if !narrow.is_empty() => {
                    (Step::Join(pick(rng, &narrow), pick(rng, &narrow)), Ty::I128)
                }
                5 if !wide.is_empty() => (Step::Low(pick(rng, &wide)), Ty::I64),
                6 if !wide.is_empty() => (Step::High(pick(rng, &wide)), Ty::I64),
                _ => continue,
            };
            program.steps.push(step);
            program.types.push(ty);
        }
        // Every value without a user, and up to two others.
        let mut used = vec![false; program.steps.len()];
        for step in &program.steps {
            match *step {
                Step::Constant(..) => {}
                Step::Add(a, b) | Step::Join(a, b) => {
                    used[a] = true;
                    used[b] = true;
                }
                Step::Low(a) | Step::High(a) => used[a] = true,
            }
        }
        program.returned = (0..used.len()).filter(|index| !used[*index]).collect();
        for _ in 0..rng.below(3) {
            program.returned.push(rng.below(used.len()));
        }
        program
    }

    fn signature(&self) -> String {
        let types = self.returned.iter().map(|index| self.types[*index].name()).collect::<Vec<_>>();
        if types.len() == 1 {
            format!("() -> {}", types[0])
        } else {
            format!("() -> ({})", types.join(", "))
        }
    }

    fn body(&self) -> String {
        let mut body = String::new();
        for (index, step) in self.steps.iter().enumerate() {
            let line = match *step {
                Step::Constant(ty, value) => format!("test.constant {value} : {}", ty.name()),
                Step::Add(a, b) => format!("test.add %{a}, %{b}"),
                Step::Join(a, b) => format!("test.join %{a}, %{b}"),
                Step::Low(a) => format!("test.low %{a}"),
                Step::High(a) => format!("test.high %{a}"),
            };
            body.push_str(&format!("    %{index} = {line};\n"));
        }
        let values = self.returned.iter().map(|index| format!("%{index}")).collect::<Vec<_>>();
        let types = self.returned.iter().map(|index| self.types[*index].name()).collect::<Vec<_>>();
        body.push_str(&format!("    builtin.ret {} : ({});", values.join(", "), types.join(", ")));
        body
    }

    fn count(&self, kind: Kind) -> usize {
        self.steps
            .iter()
            .filter(|step| {
                matches!(
                    (kind, step),
                    (Kind::Constant, Step::Constant(..))
                        | (Kind::Add, Step::Add(..))
                        | (Kind::Join, Step::Join(..))
                        | (Kind::Low, Step::Low(..))
                        | (Kind::High, Step::High(..))
                )
            })
            .count()
    }
}

/// The operations of the entry block of `function`, in order.
fn operations(function: OperationRef) -> Vec<OperationRef> {
    let function = function.try_downcast_op::<Function>().unwrap();
    let entry = function.borrow().entry_block();
    entry.borrow().body().iter().map(|op| op.as_operation_ref()).collect()
}

/// The numbers `function` returns, computed by running its entry block over the test operations.
fn evaluate(function: OperationRef) -> Vec<u128> {
    let mut values: Vec<(ValueRef, u128)> = vec![];
    let get = |values: &[(ValueRef, u128)], value: ValueRef| {
        values
            .iter()
            .find(|(defined, _)| *defined == value)
            .map(|(_, number)| *number)
            .unwrap_or_else(|| panic!("{} is not defined before its use", value.borrow()))
    };
    for op in operations(function) {
        let op = op.borrow();
        let operands = op
            .operands()
            .iter()
            .map(|operand| get(&values, operand.borrow().as_value_ref()))
            .collect::<Vec<_>>();
        if op.is::<Ret>() {
            return operands;
        }
        let result = op.results()[0].borrow().as_value_ref();
        let width = Int::width_for_type(result.borrow().ty()).expect("an integer result");
        let number = if let Some(constant) = op.downcast_ref::<Constant>() {
            constant.get_value().unsigned()
        } else if op.is::<Add>() {
            operands[0].wrapping_add(operands[1])
        } else if op.is::<Join>() {
            (operands[0] & mask(64)) | operands[1] << 64
        } else if op.is::<Low>() || op.is::<Narrow>() || op.is::<Widen>() {
            operands[0]
        } else if op.is::<High>() {
            operands[0] >> 64
        } else {
            panic!("the evaluator does not know '{}'", op.name());
        };
        values.push((result, number & mask(width)));
    }
    panic!("the function does not return");
}

/// Whether `op` has an operand or a result of type `ty`.
fn touches(op: &Operation, ty: Type) -> bool {
    op.operands().iter().any(|operand| operand.borrow().ty() == ty)
        || op.results().iter().any(|result| result.borrow().ty() == ty)
}

/// A split `test.join` is replaced by its two halves.
fn split_join(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    Rule::new::<Join>(
        context,
        "split-join",
        Some(converter.clone()),
        [],
        |op, operands, rewriter| {
            let Some([low, high]) = operands.all_one().map(|values| [values[0], values[1]]) else {
                return Ok(false);
            };
            rewriter.replace_op_with_multiple(op, &[&[low, high]])?;
            Ok(true)
        },
    )
}

/// A `test.low` (or, with `high`, a `test.high`) of a split value is replaced by that half.
fn split_half<Root: OpRegistration>(
    context: &Rc<Context>,
    converter: &TypeConverter,
    high: bool,
) -> Rule {
    Rule::new::<Root>(
        context,
        "split-half",
        Some(converter.clone()),
        [],
        move |op, operands, rewriter| {
            let [low_half, high_half] = operands.get(0) else {
                return Ok(false);
            };
            rewriter.replace_op(op, &[if high { *high_half } else { *low_half }])?;
            Ok(true)
        },
    )
}

/// A `test.constant` or `test.add` that uses an `i64` is retyped to `i32` in place: its operands
/// are set to the values they became, and its result type (and a constant's value, truncated) is
/// rewritten by `replace_elements_in`.
fn narrow_in_place<Root: OpRegistration>(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    Rule::new::<Root>(
        context,
        "narrow-in-place",
        Some(converter.clone()),
        [],
        |op, operands, rewriter| {
            let i64 = op.borrow().context().i64();
            if !touches(&op.borrow(), i64) {
                return Ok(false);
            }
            let Some(new) = operands.all_one() else {
                return Ok(false);
            };
            let old = op
                .borrow()
                .operands()
                .iter()
                .map(|operand| operand.borrow().as_value_ref())
                .collect::<Vec<_>>();
            let context = op.borrow().context_rc();
            let mapper = TypeMapper::new(&context, |context, ty| {
                if ty == context.i64() {
                    context.i32()
                } else {
                    ty
                }
            });
            let mut rewrite = ElementRewrite::types(mapper).with_int_constants(|value, _, new| {
                Ok(value.trunc(Int::width_for_type(new).expect("an integer type")))
            });
            rewriter.modify_op_in_place(op, |op| {
                for (old, new) in old.iter().zip(&new) {
                    op.replaces_uses_of_with(*old, *new);
                }
                replace_elements_in(op, &mut rewrite)
            })??;
            Ok(true)
        },
    )
}

/// Adds to `converter` the split family's callbacks, recording the operations they build.
fn split_callbacks(converter: &mut TypeConverter, built: Seen<OperationRef>) {
    let joins = built.clone();
    converter.add_source_materialization(move |rewriter, inputs, ty, span| {
        let [low, high] = inputs else {
            return Ok(None);
        };
        if ty != rewriter.context().i128() {
            return Ok(None);
        }
        let join = rewriter.create_op::<Join, _>(span, (*low, *high))?;
        joins.borrow_mut().push(join.as_operation_ref());
        Ok(Some(join.borrow().result().as_value_ref()))
    });
    converter.add_target_materialization(move |rewriter, input, types, span| {
        let i64 = rewriter.context().i64();
        if types != [i64, i64] {
            return Ok(None);
        }
        let low = rewriter.create_op::<Low, _>(span, (input,))?;
        let high = rewriter.create_op::<High, _>(span, (input,))?;
        built.borrow_mut().extend([low.as_operation_ref(), high.as_operation_ref()]);
        let low = low.borrow().result().as_value_ref();
        let high = high.borrow().result().as_value_ref();
        Ok(Some(smallvec::smallvec![low, high]))
    });
}

/// `i64` becomes `i32`; the target callback builds a `test.narrow` and the source callback a
/// `test.widen`, recorded in `built`.
fn narrow_converter(context: &Context, built: Seen<OperationRef>) -> TypeConverter {
    let (i32, i64) = (context.i32(), context.i64());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| (ty == i64).then_some(TypeConversion::One(i32)));
    let widened = built.clone();
    converter.add_source_materialization(move |rewriter, inputs, ty, span| {
        let [value] = inputs else {
            return Ok(None);
        };
        if ty != rewriter.context().i64() {
            return Ok(None);
        }
        let widen = rewriter.create_op::<Widen, _>(span, (*value,))?;
        widened.borrow_mut().push(widen.as_operation_ref());
        Ok(Some(widen.borrow().result().as_value_ref()))
    });
    converter.add_target_materialization(move |rewriter, input, types, span| {
        if types != [rewriter.context().i32()] {
            return Ok(None);
        }
        let narrow = rewriter.create_op::<Narrow, _>(span, (input,))?;
        built.borrow_mut().push(narrow.as_operation_ref());
        Ok(Some(smallvec::smallvec![narrow.borrow().result().as_value_ref()]))
    });
    converter
}

/// Converts one program of a family with a random choice of patterns and rules, and checks the
/// result. Returns a description of the failure.
fn check(rng: &mut Rng, split: bool) -> Result<(), String> {
    let program = Program::random(rng, split);
    let (context, function) = parse_function(&program.signature(), &program.body());
    let source = function.borrow().to_string();
    let expected = evaluate(function);

    let built: Seen<OperationRef> = Rc::default();
    let mut target = ConversionTarget::new(context.clone());
    target.add_legal_op::<Function>().add_legal_op::<Ret>();
    let mut patterns = vec![];
    let mut patterned = vec![];
    // The type a converted kind must no longer use.
    let gone;
    if split {
        gone = context.i128();
        let mut converter = wide_converter(&context);
        split_callbacks(&mut converter, built.clone());
        for kind in [Kind::Constant, Kind::Join, Kind::Low, Kind::High] {
            if !rng.coin() {
                continue;
            }
            patterned.push(kind);
            patterns.push(match kind {
                Kind::Constant => split_constant(&context, &converter),
                Kind::Join => split_join(&context, &converter),
                Kind::Low => split_half::<Low>(&context, &converter, false),
                _ => split_half::<High>(&context, &converter, true),
            });
        }
        // The callbacks build joins, lows and highs, which a rule for them would call illegal.
        if patterned.contains(&Kind::Constant) && rng.coin() {
            target.add_dynamically_legal_op::<Constant, _>(|op: &Operation| {
                DynamicLegalityResult::legal_if(has_no_wide_types(op))
            });
        }
    } else {
        gone = context.i64();
        let converter = narrow_converter(&context, built.clone());
        for kind in [Kind::Constant, Kind::Add] {
            if !rng.coin() {
                continue;
            }
            patterned.push(kind);
            let legal =
                |op: &Operation| DynamicLegalityResult::legal_if(!touches(op, op.context().i64()));
            let ruled = rng.coin();
            if kind == Kind::Constant {
                patterns.push(narrow_in_place::<Constant>(&context, &converter));
                if ruled {
                    target.add_dynamically_legal_op::<Constant, _>(legal);
                }
            } else {
                patterns.push(narrow_in_place::<Add>(&context, &converter));
                if ruled {
                    target.add_dynamically_legal_op::<Add, _>(legal);
                }
            }
        }
    }
    let describe_case = || format!("patterned kinds {patterned:?}\n{source}");

    convert_partial(function, target, patterns)
        .map_err(|err| format!("{}\n{}", describe(&err), describe_case()))?;
    let output = function.borrow().to_string();
    let built = built.borrow();

    for op in operations(function) {
        let op = op.borrow();
        if op.is::<UnrealizedConversionCast>() {
            return Err(format!("a cast remains:\n{output}\n{}", describe_case()));
        }
        let kind = Kind::of(&op);
        if kind.is_some_and(|kind| patterned.contains(&kind))
            && !built.contains(&op.as_operation_ref())
            && touches(&op, gone)
        {
            return Err(format!(
                "a '{}' of a converted kind uses '{gone}':\n{output}\n{}",
                op.name(),
                describe_case()
            ));
        }
    }
    let kinds = [Kind::Constant, Kind::Add, Kind::Join, Kind::Low, Kind::High];
    for kind in kinds.into_iter().filter(|kind| !patterned.contains(kind)) {
        let remaining = operations(function)
            .into_iter()
            .filter(|op| Kind::of(&op.borrow()) == Some(kind) && !built.contains(op))
            .count();
        if remaining != program.count(kind) {
            return Err(format!(
                "the kind {kind:?} has no pattern, but {} of its operations became \
                 {remaining}:\n{output}\n{}",
                program.count(kind),
                describe_case()
            ));
        }
    }

    let actual = evaluate(function);
    let agree = if split {
        actual == expected
    } else {
        actual.iter().zip(&expected).all(|(a, e)| a & mask(32) == e & mask(32))
            && actual.len() == expected.len()
    };
    if !agree {
        return Err(format!(
            "returned {actual:x?}, expected {expected:x?}:\n{output}\n{}",
            describe_case()
        ));
    }
    Ok(())
}

/// Random programs of both families, partially converted, verify, keep no cast, leave no use of the
/// converted type on a converted kind, leave the kinds without a pattern as they were, and return
/// what they returned before (the low 32 bits of it, for the narrow family).
#[test]
fn random_programs_keep_their_values_under_a_partial_conversion() {
    for seed in seeds() {
        let mut rng = Rng(seed);
        for split in [true, false] {
            for case in 0..PROGRAMS {
                if let Err(failure) = check(&mut rng, split) {
                    panic!(
                        "seed {seed} (CONVERSION_TEST_SEED={seed}), {} family, case {case}: \
                         {failure}",
                        if split { "split" } else { "narrow" }
                    );
                }
            }
        }
    }
}

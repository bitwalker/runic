use smallvec::SmallVec;

use crate::{
    Operation, OperationRef, Value, ValueRef,
    adt::SmallSet,
    dialects::builtin::UnrealizedConversionCast,
    patterns::{PatternRewriter, Rewriter},
};

/// What [reconcile_unrealized_casts] did: whether it changed the IR, and the casts that remain.
#[derive(Debug, Default)]
pub struct ReconciledCasts {
    /// A use was redirected or a cast erased.
    pub changed: bool,
    /// The casts left under the root, in program order.
    pub remaining: SmallVec<[OperationRef; 4]>,
}

/// Folds casts under `root` until nothing changes: an unused cast is erased; a cast whose input
/// types equal its result types has its results replaced by its inputs; a cast whose inputs are
/// all the results of one cast, in order, and whose result types equal the input types of that
/// cast or of a cast further up such a chain, has its results replaced by that cast's inputs.
/// Returns whether the IR changed and the casts that remain.
///
/// The casts are `builtin.unrealized_conversion_cast` operations nested in `root`, not `root`
/// itself. Types are compared pairwise and in count. A cast with no inputs is never folded, only
/// erased when unused; neither is a cast whose chain stops at one. A cast holding only some
/// results of another, all of them in another order, or results of two casts, is not part of a
/// chain. Two folds that only a graph region allows are not made: one whose replacement values
/// include a result of the cast folded, which could not leave the cast unused; and one that would
/// replace a result by a value whose defining operation reads that result, which would leave that
/// operation reading its own result (the verifier refuses that). The casts that remain are given
/// in program order (a pre-order walk of `root`).
///
/// Folding is sound when casts compose: a cast from one type list to a second, followed by a cast
/// from the second to a third, means the same as a cast from the first to the third; and a cast
/// from a type list to itself is the identity. A chain of three or more casts is folded on that
/// assumption, which round trips being the identity do not give alone. The materializations that
/// realize the casts that remain must agree with it.
pub fn reconcile_unrealized_casts(root: OperationRef) -> ReconciledCasts {
    let mut rewriter = PatternRewriter::new(root.borrow().context_rc());
    let mut reconciled = ReconciledCasts::default();
    loop {
        let mut changed = false;
        for cast in casts_under(root) {
            if !cast.borrow().is_used() {
                rewriter.erase_op(cast);
                changed = true;
                continue;
            }
            let Some(replacements) = folded_results(&cast.borrow()) else {
                continue;
            };
            let results = values(cast.borrow().results().iter().map(|r| r.borrow().as_value_ref()));
            for (result, replacement) in results.into_iter().zip(replacements) {
                if result.borrow().is_used() {
                    rewriter.replace_all_uses_of_value_with(result, replacement);
                    changed = true;
                }
            }
        }
        if !changed {
            reconciled.remaining = casts_under(root);
            return reconciled;
        }
        reconciled.changed = true;
    }
}

/// The casts nested in `root`, in pre-order.
fn casts_under(root: OperationRef) -> SmallVec<[OperationRef; 4]> {
    let mut casts = SmallVec::new();
    root.borrow().prewalk_all(|op: &Operation| {
        if op.is::<UnrealizedConversionCast>() && op.as_operation_ref() != root {
            casts.push(op.as_operation_ref());
        }
    });
    casts
}

fn values(values: impl Iterator<Item = ValueRef>) -> SmallVec<[ValueRef; 2]> {
    values.collect()
}

fn inputs_of(op: &Operation) -> SmallVec<[ValueRef; 2]> {
    values(op.operands().iter().map(|operand| operand.borrow().as_value_ref()))
}

fn same_types(values: &[ValueRef], types: &[crate::Type]) -> bool {
    values.len() == types.len() && values.iter().zip(types).all(|(v, ty)| v.borrow().ty() == *ty)
}

/// The values that replace the results of `cast`, when it folds: never one of its own results,
/// and never a value whose defining operation reads the result it would replace.
fn folded_results(cast: &Operation) -> Option<SmallVec<[ValueRef; 2]>> {
    let replacements = chain_end(cast)?;
    let own = |value: &ValueRef| value.borrow().get_defining_op() == Some(cast.as_operation_ref());
    if replacements.iter().any(own) {
        return None;
    }
    let read_by_its_definer = crate::folder::replacement_is_read_by_its_definer(
        cast,
        replacements.iter().copied().map(Some),
    );
    (!read_by_its_definer).then_some(replacements)
}

/// The inputs of `cast` when they have its result types, or else the inputs of the first cast up
/// its chain that have them.
fn chain_end(cast: &Operation) -> Option<SmallVec<[ValueRef; 2]>> {
    let inputs = inputs_of(cast);
    if inputs.is_empty() {
        return None;
    }
    let result_types = cast
        .results()
        .iter()
        .map(|result| result.borrow().ty())
        .collect::<SmallVec<[crate::Type; 2]>>();
    if same_types(&inputs, &result_types) {
        return Some(inputs);
    }

    // Walk up the chain of casts each of whose inputs are all the results of the one above.
    let mut seen = SmallSet::<OperationRef, 4>::default();
    let mut current = inputs;
    while let Some(above) = cast_whose_results_are(&current) {
        if !seen.insert(above) {
            return None;
        }
        let above_inputs = inputs_of(&above.borrow());
        if above_inputs.is_empty() {
            return None;
        }
        if same_types(&above_inputs, &result_types) {
            return Some(above_inputs);
        }
        current = above_inputs;
    }
    None
}

/// The cast whose results are exactly `values`, all of them in order.
fn cast_whose_results_are(values: &[ValueRef]) -> Option<OperationRef> {
    let defining = values.first()?.borrow().get_defining_op()?;
    let op = defining.borrow();
    if !op.is::<UnrealizedConversionCast>() {
        return None;
    }
    let results = op.results();
    let all = results.len() == values.len()
        && results
            .iter()
            .zip(values)
            .all(|(result, value)| result.borrow().as_value_ref() == *value);
    all.then_some(defining)
}

#[cfg(test)]
mod tests {
    use alloc::{
        format,
        rc::Rc,
        string::{String, ToString},
        vec::Vec,
    };

    use super::*;
    use crate::{
        BuilderExt, Context, Int, SourceSpan, Type,
        dialects::{
            builtin::{BuiltinOpBuilder, Function},
            test::{Constant, TestDialect},
        },
        parse::{ParserConfig, parse, parse_any},
        testing::{Test, text::describe},
    };

    /// Parses `source` as a function, reconciles it, and returns the function printed and the
    /// casts that remain, printed.
    fn reconciled(source: &str) -> (String, Vec<String>) {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<TestDialect>();
        let function = parse::<Function>(ParserConfig::new(context.clone()), "reconcile", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        let function = function.as_operation_ref();
        let left = reconcile_unrealized_casts(function).remaining;
        function
            .borrow()
            .recursively_verify()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        let left = left.iter().map(|op| op.borrow().to_string()).collect();
        (function.borrow().to_string(), left)
    }

    fn function(signature: &str, body: &str) -> String {
        format!("builtin.function public extern(\"C\") @f{signature} {{\n{body}\n}};")
    }

    /// A cast between equal type lists is replaced by its inputs, one to one and N to N.
    #[test]
    fn an_identity_cast_folds_to_its_inputs() {
        let source = function(
            "(%0: i32, %1: i64) -> (i32, i32, i64)",
            "    %2 = builtin.unrealized_conversion_cast %0 : i32 to i32;
    %3, %4 = builtin.unrealized_conversion_cast %0, %1 : (i32, i64) to (i32, i64);
    builtin.ret %2, %3, %4 : (i32, i32, i64);",
        );
        let (printed, left) = reconciled(&source);
        assert_eq!(
            printed,
            function(
                "(%0: i32, %1: i64) -> (i32, i32, i64)",
                "    builtin.ret %0, %0, %1 : (i32, i32, i64);"
            )
        );
        assert!(left.is_empty(), "{left:?}");
    }

    /// A split followed by its join, and a join followed by its split, give back what they started
    /// from.
    #[test]
    fn a_split_and_its_join_fold_away() {
        let source = function(
            "(%0: i128, %1: i64, %2: i64) -> (i128, i64, i64)",
            "    %3, %4 = builtin.unrealized_conversion_cast %0 : i128 to (i64, i64);
    %5 = builtin.unrealized_conversion_cast %3, %4 : (i64, i64) to i128;
    %6 = builtin.unrealized_conversion_cast %1, %2 : (i64, i64) to i128;
    %7, %8 = builtin.unrealized_conversion_cast %6 : i128 to (i64, i64);
    builtin.ret %5, %7, %8 : (i128, i64, i64);",
        );
        let (printed, left) = reconciled(&source);
        assert_eq!(
            printed,
            function(
                "(%0: i128, %1: i64, %2: i64) -> (i128, i64, i64)",
                "    builtin.ret %0, %1, %2 : (i128, i64, i64);"
            )
        );
        assert!(left.is_empty(), "{left:?}");
    }

    /// The last of three casts, whose result type is the first one's input type, is replaced by
    /// the first one's input.
    #[test]
    fn a_chain_of_three_whose_ends_agree_folds_away() {
        let source = function(
            "(%0: i32) -> i32",
            "    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %2, %3 = builtin.unrealized_conversion_cast %1 : i64 to (i16, i16);
    %4 = builtin.unrealized_conversion_cast %2, %3 : (i16, i16) to i32;
    builtin.ret %4 : (i32);",
        );
        let (printed, left) = reconciled(&source);
        assert_eq!(printed, function("(%0: i32) -> i32", "    builtin.ret %0 : (i32);"));
        assert!(left.is_empty(), "{left:?}");
    }

    /// A cast holding only some results of another, all of them in another order, or results of
    /// two casts, is not part of a chain: it stays, and so do the casts it uses.
    #[test]
    fn a_cast_that_is_not_a_whole_chain_link_stays() {
        let body = "    %2, %3 = builtin.unrealized_conversion_cast %0 : i128 to (i64, i64);
    %4, %5 = builtin.unrealized_conversion_cast %1 : i128 to (i64, i64);
    %6 = builtin.unrealized_conversion_cast %2 : i64 to i128;
    %7 = builtin.unrealized_conversion_cast %3, %2 : (i64, i64) to i128;
    %8 = builtin.unrealized_conversion_cast %2, %5 : (i64, i64) to i128;
    builtin.ret %6, %7, %8 : (i128, i128, i128);";
        let signature = "(%0: i128, %1: i128) -> (i128, i128, i128)";
        let source = function(signature, body);
        let (printed, left) = reconciled(&source);
        assert_eq!(printed, source);
        assert_eq!(
            left,
            [
                "%2, %3 = builtin.unrealized_conversion_cast %0 : i128 to (i64, i64);",
                "%4, %5 = builtin.unrealized_conversion_cast %1 : i128 to (i64, i64);",
                "%6 = builtin.unrealized_conversion_cast %2 : i64 to i128;",
                "%7 = builtin.unrealized_conversion_cast %3, %2 : (i64, i64) to i128;",
                "%8 = builtin.unrealized_conversion_cast %2, %5 : (i64, i64) to i128;",
            ]
        );
    }

    /// A chain whose ends disagree stays and is returned; so is a cast with no inputs. An unused
    /// cast is erased.
    #[test]
    fn a_chain_whose_ends_disagree_and_a_cast_without_inputs_stay() {
        let source = function(
            "(%0: i32) -> (i16, i128)",
            "    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %2 = builtin.unrealized_conversion_cast %1 : i64 to i16;
    %3 = builtin.unrealized_conversion_cast : () to i128;
    %4 = builtin.unrealized_conversion_cast %0 : i32 to i8;
    builtin.ret %2, %3 : (i16, i128);",
        );
        let (printed, left) = reconciled(&source);
        assert_eq!(
            printed,
            function(
                "(%0: i32) -> (i16, i128)",
                "    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %2 = builtin.unrealized_conversion_cast %1 : i64 to i16;
    %3 = builtin.unrealized_conversion_cast : () to i128;
    builtin.ret %2, %3 : (i16, i128);"
            )
        );
        assert_eq!(left.len(), 3);
    }

    /// In a graph region a cast `C` may feed the cast its chain reaches: folding `C` would
    /// replace its results by `(%0, %2)`, `%0` being one of its own, which cannot leave `C`
    /// unused. The fold is not made, the function returns, and a second call changes nothing.
    #[test]
    fn a_cast_whose_fold_would_feed_itself_stays() {
        let source = "builtin.module public @m {
    %0, %1 = builtin.unrealized_conversion_cast %3 : i128 to (i64, i64);
    %2 = builtin.unrealized_conversion_cast : () to i64;
    %3 = builtin.unrealized_conversion_cast %0, %2 : (i64, i64) to i128;
    %4 = test.add %1, %1;
};";
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<TestDialect>();
        let module = parse_any(ParserConfig::new(context.clone()), "reconcile", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        let printed = module.borrow().to_string();
        let reconciled = reconcile_unrealized_casts(module);
        module
            .borrow()
            .recursively_verify()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        assert!(!reconciled.changed);
        assert_eq!(reconciled.remaining.len(), 3);
        assert_eq!(module.borrow().to_string(), printed);

        let again = reconcile_unrealized_casts(module);
        assert!(!again.changed);
        assert_eq!(again.remaining.as_slice(), reconciled.remaining.as_slice());
    }

    /// In a graph region two identity casts may read each other. Folding either would replace
    /// its results by the other's, which the other reads: the other would then read its own
    /// results, which the verifier refuses. Neither fold is made, and both casts remain.
    #[test]
    fn two_casts_that_read_each_other_stay() {
        let source = "builtin.module public @m {
    %0, %1 = builtin.unrealized_conversion_cast %2, %3 : (i64, i64) to (i64, i64);
    %2, %3 = builtin.unrealized_conversion_cast %0, %1 : (i64, i64) to (i64, i64);
    %4 = test.add %0, %3;
};";
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<TestDialect>();
        let module = parse_any(ParserConfig::new(context.clone()), "reconcile", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        let printed = module.borrow().to_string();
        let reconciled = reconcile_unrealized_casts(module);
        module
            .borrow()
            .recursively_verify()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        assert!(!reconciled.changed);
        assert_eq!(reconciled.remaining.len(), 2);
        assert_eq!(module.borrow().to_string(), printed);
        assert_eq!(module.borrow().to_string(), printed);
    }

    /// The deterministic generator of the randomized test (splitmix64).
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
    }

    /// The seeds the randomized test runs: `RECONCILE_TEST_SEED` if set, else three fixed ones.
    fn seeds() -> Vec<u64> {
        match std::env::var("RECONCILE_TEST_SEED") {
            Ok(seed) => alloc::vec![seed.parse().expect("RECONCILE_TEST_SEED is a u64")],
            Err(_) => alloc::vec![0x0c45_0001, 0x0c45_0002, 0x0c45_0003],
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    enum Kind {
        I32,
        I64,
        I128,
    }

    /// One operation of a generated program; operands are indices of earlier values.
    #[derive(Clone, Debug)]
    enum Step {
        Constant(Kind, u128),
        /// An `i128` to its low and high halves.
        Split(usize),
        /// Two `i64` values, low then high, to an `i128`.
        Join(usize, usize),
        /// An `i128` to its low 64 bits, the next 32 and the top 32: `(i64, i32, i32)`.
        Split3(usize),
        /// An `i64` and two `i32` values, low to high, to an `i128`.
        Join3(usize, usize, usize),
        /// Two `i64` halves to the low one and the high one's two halves: `(i64, i32, i32)`.
        Resplit(usize, usize),
        /// An `i64` and two `i32` values to the `i64` and the two joined: `(i64, i64)`.
        Merge(usize, usize, usize),
        /// Values to values of their own types.
        Identity(Vec<usize>),
    }

    const LOW: u128 = u64::MAX as u128;
    const LOW32: u128 = u32::MAX as u128;

    /// The low and high halves of `bits`.
    fn halves(bits: u128) -> [u128; 2] {
        [bits & LOW, bits >> 64]
    }

    /// `(lo, hi)` as `(lo, hi`'s low 32 bits, `hi`'s high 32 bits`)`.
    fn resplit(lo: u128, hi: u128) -> [u128; 3] {
        [lo, hi & LOW32, hi >> 32]
    }

    /// A random program: its steps, the kinds and numbers of its values, and the values used.
    fn program(rng: &mut Rng) -> (Vec<Step>, Vec<(Kind, u128)>, Vec<usize>) {
        let mut steps = Vec::new();
        let mut values: Vec<(Kind, u128)> = Vec::new();
        // The results of splits and merges, and of three-way splits and resplits.
        let mut splits: Vec<(usize, usize)> = Vec::new();
        let mut triples: Vec<(usize, usize, usize)> = Vec::new();
        let of = |values: &[(Kind, u128)], kind| {
            values
                .iter()
                .enumerate()
                .filter(|(_, v)| v.0 == kind)
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        };
        for _ in 0..2 + rng.below(16) {
            let wide = of(&values, Kind::I128);
            let narrow = of(&values, Kind::I64);
            let step = match rng.below(10) {
                0 => {
                    let kind = if rng.below(2) == 0 {
                        Kind::I64
                    } else {
                        Kind::I128
                    };
                    let bits = (rng.next() as u128) << 64 | rng.next() as u128;
                    Step::Constant(kind, if kind == Kind::I64 { bits & LOW } else { bits })
                }
                1 | 2 if !wide.is_empty() => {
                    Step::Split(wide[wide.len() - 1 - rng.below(wide.len().min(3))])
                }
                3 if !splits.is_empty() => {
                    let (lo, hi) = splits[splits.len() - 1 - rng.below(splits.len().min(3))];
                    Step::Join(lo, hi)
                }
                4 if narrow.len() >= 2 => {
                    Step::Join(narrow[rng.below(narrow.len())], narrow[rng.below(narrow.len())])
                }
                5 if !values.is_empty() => {
                    Step::Identity((0..1 + rng.below(3)).map(|_| rng.below(values.len())).collect())
                }
                6 if !wide.is_empty() => {
                    Step::Split3(wide[wide.len() - 1 - rng.below(wide.len().min(3))])
                }
                7 if !splits.is_empty() => {
                    let (lo, hi) = splits[splits.len() - 1 - rng.below(splits.len().min(3))];
                    Step::Resplit(lo, hi)
                }
                8 if !triples.is_empty() => {
                    let (a, b, c) = triples[triples.len() - 1 - rng.below(triples.len().min(3))];
                    Step::Join3(a, b, c)
                }
                9 if !triples.is_empty() => {
                    let (a, b, c) = triples[triples.len() - 1 - rng.below(triples.len().min(3))];
                    Step::Merge(a, b, c)
                }
                _ => Step::Constant(Kind::I128, (rng.next() as u128) << 64 | rng.next() as u128),
            };
            match &step {
                Step::Constant(kind, bits) => values.push((*kind, *bits)),
                Step::Split(source) => {
                    splits.push((values.len(), values.len() + 1));
                    values.extend(halves(values[*source].1).map(|bits| (Kind::I64, bits)));
                }
                Step::Join(lo, hi) => {
                    values.push((Kind::I128, values[*lo].1 | values[*hi].1 << 64))
                }
                Step::Split3(source) => {
                    let [lo, hi] = halves(values[*source].1);
                    let n = values.len();
                    triples.push((n, n + 1, n + 2));
                    let [a, b, c] = resplit(lo, hi);
                    values.extend([(Kind::I64, a), (Kind::I32, b), (Kind::I32, c)]);
                }
                Step::Join3(a, b, c) => values
                    .push((Kind::I128, values[*a].1 | values[*b].1 << 64 | values[*c].1 << 96)),
                Step::Resplit(lo, hi) => {
                    let n = values.len();
                    triples.push((n, n + 1, n + 2));
                    let [a, b, c] = resplit(values[*lo].1, values[*hi].1);
                    values.extend([(Kind::I64, a), (Kind::I32, b), (Kind::I32, c)]);
                }
                Step::Merge(a, b, c) => {
                    splits.push((values.len(), values.len() + 1));
                    let hi = values[*b].1 | values[*c].1 << 32;
                    values.extend([(Kind::I64, values[*a].1), (Kind::I64, hi)]);
                }
                Step::Identity(sources) => {
                    for source in sources {
                        values.push(values[*source]);
                    }
                }
            }
            steps.push(step);
        }
        let uses = (0..1 + rng.below(4)).map(|_| rng.below(values.len())).collect();
        (steps, values, uses)
    }

    /// Builds `steps` as the body of a function returning the values `uses`.
    fn build(steps: &[Step], kinds: &[(Kind, u128)], uses: &[usize]) -> Test {
        let ty = |cx: &Context, kind| match kind {
            Kind::I32 => cx.i32(),
            Kind::I64 => cx.i64(),
            Kind::I128 => cx.i128(),
        };
        let mut test = Test::new("f", |cx| {
            (Vec::<Type>::new(), uses.iter().map(|u| ty(cx, kinds[*u].0)).collect::<Vec<_>>())
        });
        test.context().get_or_register_dialect::<TestDialect>();
        let context = test.context_rc();
        let span = SourceSpan::UNKNOWN;
        let mut values: Vec<ValueRef> = Vec::new();
        let mut f = test.function_builder();
        for step in steps {
            let outputs = |cast: crate::UnsafeIntrusiveEntityRef<UnrealizedConversionCast>| {
                cast.borrow()
                    .outputs()
                    .iter()
                    .map(|r| r.borrow().as_value_ref())
                    .collect::<Vec<_>>()
            };
            match step {
                Step::Constant(kind, bits) => {
                    let value = match kind {
                        Kind::I32 => Int::from(*bits as u32),
                        Kind::I64 => Int::from(*bits as u64),
                        Kind::I128 => Int::from(*bits),
                    };
                    let constant = f.builder_mut().create::<Constant, _>(span)(value).unwrap();
                    values.push(constant.borrow().result().as_value_ref());
                }
                Step::Split(source) => {
                    let (i64, input) = (context.i64(), values[*source]);
                    let cast = f.unrealized_conversion_cast([input], [i64, i64], span).unwrap();
                    values.extend(outputs(cast));
                }
                Step::Join(lo, hi) => {
                    let inputs = [values[*lo], values[*hi]];
                    let cast =
                        f.unrealized_conversion_cast(inputs, [context.i128()], span).unwrap();
                    values.extend(outputs(cast));
                }
                Step::Split3(source) => {
                    let types = [context.i64(), context.i32(), context.i32()];
                    let cast =
                        f.unrealized_conversion_cast([values[*source]], types, span).unwrap();
                    values.extend(outputs(cast));
                }
                Step::Join3(a, b, c) => {
                    let inputs = [values[*a], values[*b], values[*c]];
                    let cast =
                        f.unrealized_conversion_cast(inputs, [context.i128()], span).unwrap();
                    values.extend(outputs(cast));
                }
                Step::Resplit(lo, hi) => {
                    let types = [context.i64(), context.i32(), context.i32()];
                    let inputs = [values[*lo], values[*hi]];
                    let cast = f.unrealized_conversion_cast(inputs, types, span).unwrap();
                    values.extend(outputs(cast));
                }
                Step::Merge(a, b, c) => {
                    let (i64, inputs) = (context.i64(), [values[*a], values[*b], values[*c]]);
                    let cast = f.unrealized_conversion_cast(inputs, [i64, i64], span).unwrap();
                    values.extend(outputs(cast));
                }
                Step::Identity(sources) => {
                    let inputs = sources.iter().map(|s| values[*s]).collect::<Vec<_>>();
                    let types = inputs.iter().map(|v| v.borrow().ty()).collect::<Vec<_>>();
                    let cast = f.unrealized_conversion_cast(inputs, types, span).unwrap();
                    values.extend(outputs(cast));
                }
            }
        }
        f.ret(uses.iter().map(|u| values[*u]), span).unwrap();
        test
    }

    /// The numbers the function of `test` returns: a constant is its bits, and a cast between
    /// equal types gives its inputs. Every other cast is told by its counts of inputs and results:
    /// a split (1 to 2) gives the halves of its input and a join (2 to 1) `lo | hi << 64`; a
    /// three-way split (1 to 3) gives the low 64 bits, the next 32 and the top 32, and its join
    /// (3 to 1) puts them back; a resplit (2 to 3) splits the high half in two, and a merge (3 to
    /// 2) joins them. Any two that chain compose to the cast from the first one's input types to
    /// the second one's result types.
    fn evaluate(test: &Test) -> Vec<u128> {
        let function = test.function();
        let function = function.borrow();
        let entry = function.entry_block();
        let mut known = crate::adt::SmallDenseMap::<ValueRef, u128, 16>::default();
        let get = |known: &crate::adt::SmallDenseMap<ValueRef, u128, 16>, value: ValueRef| {
            *known.get(&value).expect("a value is defined before it is used")
        };
        for op in entry.borrow().body() {
            let inputs = inputs_of(&op);
            let results = values(op.results().iter().map(|r| r.borrow().as_value_ref()));
            if let Some(constant) = op.downcast_ref::<Constant>() {
                known.insert(results[0], constant.get_value().unsigned());
            } else if op.is::<UnrealizedConversionCast>() {
                let input_types = inputs.iter().map(|v| v.borrow().ty()).collect::<Vec<_>>();
                let result_types = results.iter().map(|v| v.borrow().ty()).collect::<Vec<_>>();
                if input_types == result_types {
                    for (result, input) in results.iter().zip(&inputs) {
                        known.insert(*result, get(&known, *input));
                    }
                } else {
                    let ins = inputs.iter().map(|v| get(&known, *v)).collect::<Vec<_>>();
                    let outs = match (ins.as_slice(), results.len()) {
                        ([bits], 2) => halves(*bits).to_vec(),
                        ([lo, hi], 1) => alloc::vec![lo | hi << 64],
                        ([bits], 3) => {
                            let [lo, hi] = halves(*bits);
                            resplit(lo, hi).to_vec()
                        }
                        ([a, b, c], 1) => alloc::vec![a | b << 64 | c << 96],
                        ([lo, hi], 3) => resplit(*lo, *hi).to_vec(),
                        ([a, b, c], 2) => alloc::vec![*a, b | c << 32],
                        _ => panic!("a cast without a meaning: {}", *op),
                    };
                    for (result, bits) in results.iter().zip(outs) {
                        known.insert(*result, bits);
                    }
                }
            } else {
                return inputs.iter().map(|v| get(&known, *v)).collect();
            }
        }
        unreachable!("the function returns")
    }

    /// How many casts up its chain `cast` reaches the one whose inputs have its result types, when
    /// there is one: 2 or more for a chain of three or more casts whose ends agree.
    fn links_to_fold(cast: OperationRef) -> Option<usize> {
        let cast = cast.borrow();
        let result_types = cast.results().iter().map(|r| r.borrow().ty()).collect::<Vec<_>>();
        let mut current = inputs_of(&cast);
        let mut links = 0;
        while let Some(above) = cast_whose_results_are(&current) {
            links += 1;
            current = inputs_of(&above.borrow());
            if current.is_empty() {
                return None;
            }
            if same_types(&current, &result_types) {
                return Some(links);
            }
        }
        None
    }

    /// Random straight-line programs of constants, two- and three-way splits and joins, resplits,
    /// merges and identity casts read the same numbers after reconciliation; no unused cast
    /// remains, and a second run changes nothing. For each seed, some casts fold, some stay, and
    /// some of those that fold end a chain of three or more casts.
    #[test]
    fn reconciliation_keeps_what_random_programs_compute() {
        for seed in seeds() {
            let mut rng = Rng(seed);
            let (mut before, mut after, mut long_chains) = (0, 0, 0);
            for case in 0..300 {
                let (steps, kinds, uses) = program(&mut rng);
                let expected = uses.iter().map(|u| kinds[*u].1).collect::<Vec<_>>();
                let test = build(&steps, &kinds, &uses);
                let fail = |what: &str| {
                    format!(
                        "seed {seed} (RECONCILE_TEST_SEED), case {case}: {what}\n{steps:?}\n{}",
                        test.function().as_operation_ref().borrow()
                    )
                };
                assert_eq!(evaluate(&test), expected, "{}", fail("the model and the IR disagree"));

                let root = test.function().as_operation_ref();
                let casts = casts_under(root);
                before += casts.len();
                let long = casts
                    .iter()
                    .copied()
                    .filter(|cast| {
                        cast.borrow().is_used() && links_to_fold(*cast).is_some_and(|n| n >= 2)
                    })
                    .collect::<Vec<_>>();
                let left = reconcile_unrealized_casts(root).remaining;
                after += left.len();
                long_chains += long.iter().filter(|cast| !left.contains(cast)).count();
                assert_eq!(evaluate(&test), expected, "{}", fail("reconciliation changed a use"));
                assert!(left.iter().all(|op| op.borrow().is_used()), "{}", fail("an unused cast"));
                assert_eq!(
                    left.as_slice(),
                    casts_under(root).as_slice(),
                    "{}",
                    fail("the casts left")
                );
                root.borrow()
                    .recursively_verify()
                    .unwrap_or_else(|err| panic!("{}", fail(&describe(&err))));

                let printed = root.borrow().to_string();
                let again = reconcile_unrealized_casts(root);
                assert!(!again.changed, "{}", fail("a second run changed the IR"));
                assert_eq!(again.remaining.as_slice(), left.as_slice(), "{}", fail("a second run"));
                assert_eq!(root.borrow().to_string(), printed, "{}", fail("a second run"));
            }
            // The programs exercise both outcomes, casts that fold and casts that stay, and fold
            // chains of three or more casts.
            assert!(0 < after && after < before, "seed {seed}: {before} casts, {after} left");
            assert!(0 < long_chains, "seed {seed}: no chain of three or more folded");
        }
    }
}

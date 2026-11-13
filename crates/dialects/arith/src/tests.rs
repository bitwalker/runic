//! Tests that build, verify, print and parse `arith` operations.

use alloc::{
    fmt::format,
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use runic_core::{
    Attribute, AttributeRef, Builder as _, Context, Int, Op, OpBuilder, OpFoldResult, OperationRef,
    Report, Signedness, SmallVec, SourceSpan, Type, Value, ValueRef,
    diagnostics::{Diagnostic, Label, VisitDiagnostic},
    dialects::builtin::{BuiltinOpBuilder, FunctionBuilder, FunctionRef, attributes::BoolAttr},
    testing::{Test, parse_function_fixpoint},
    traits::Foldable,
};

use crate::*;

type TypeFn = fn(&Context) -> Type;
type Builder<'a, 'f> = &'a mut FunctionBuilder<'f, OpBuilder>;

const SPAN: SourceSpan = SourceSpan::UNKNOWN;

/// The message of every label attached to `diagnostic`.
fn label_messages(diagnostic: &dyn Diagnostic) -> Vec<String> {
    struct Labels(Vec<String>);

    impl VisitDiagnostic for Labels {
        fn label(&mut self, label: Label<'_>) {
            self.0.push(label.message.map(format).unwrap_or_default());
        }
    }

    let mut labels = Labels(Vec::new());
    diagnostic.visit(&mut labels);
    labels.0
}

/// A test of a function `f` with the parameters `params` make and no results, in a context with
/// the arithmetic dialect.
fn test_with(params: &[TypeFn]) -> Test {
    let params = params.to_vec();
    let test = Test::new("f", move |cx| (params.iter().map(|ty| ty(cx)).collect::<Vec<_>>(), []));
    test.context().get_or_register_dialect::<ArithDialect>();
    test
}

/// The parameters of `test`'s function.
fn params(test: &Test) -> Vec<ValueRef> {
    test.entry_block()
        .borrow()
        .arguments()
        .iter()
        .map(|arg| arg.borrow().as_value_ref())
        .collect()
}

/// The operations of the entry block of `function`.
fn ops_of(function: FunctionRef) -> Vec<OperationRef> {
    let function = function.borrow();
    let body = function.body();
    let entry = body.entry();
    entry.body().iter().map(|op| op.as_operation_ref()).collect()
}

/// Ends `test`'s function, verifies it, prints it, parses the print and prints that again, and
/// returns the text and the operations of the parsed function (which `context` keeps alive).
fn finish(test: &mut Test) -> (String, Rc<Context>, Vec<OperationRef>) {
    test.function_builder().ret([], SPAN).unwrap();
    let function = test.function();
    function.borrow().as_operation().verify().expect("valid function");
    let text = format!("{}", function.borrow().as_operation());
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<ArithDialect>();
    let (parsed, printed) = parse_function_fixpoint(&context, "f", &text).expect("parses");
    assert_eq!(printed, text);
    (text, context, ops_of(parsed))
}

/// The types of the results of `op`.
fn result_types(op: OperationRef) -> Vec<Type> {
    op.borrow().results().iter().map(|result| result.borrow().ty()).collect()
}

/// Builds one operation with `build` from two parameters of the type `ty` makes, checks that its
/// results have the types `expected` gives for the operand type, round-trips the function, and
/// checks the parsed operation the same way. Returns the text.
fn binary_round_trip(
    ty: TypeFn,
    build: impl FnOnce(Builder<'_, '_>, ValueRef, ValueRef) -> Vec<ValueRef>,
    expected: impl Fn(&Context, Type) -> Vec<Type>,
) -> String {
    let mut test = test_with(&[ty, ty]);
    let [lhs, rhs] = params(&test)[..] else {
        unreachable!()
    };
    let results = build(&mut test.function_builder(), lhs, rhs);
    let operand_ty = ty(test.context());
    let types = results.iter().map(|result| result.borrow().ty()).collect::<Vec<_>>();
    assert_eq!(types, expected(test.context(), operand_ty));
    let (text, context, ops) = finish(&mut test);
    assert_eq!(result_types(ops[0]), expected(&context, ty(&context)), "{text}");
    text
}

/// As [binary_round_trip], for an operation of one operand.
fn unary_round_trip(
    ty: TypeFn,
    build: impl FnOnce(Builder<'_, '_>, ValueRef) -> ValueRef,
    expected: impl Fn(&Context, Type) -> Type,
) -> String {
    binary_round_trip(ty, |b, operand, _| vec![build(b, operand)], |cx, ty| vec![expected(cx, ty)])
}

fn same(_: &Context, ty: Type) -> Vec<Type> {
    vec![ty]
}

fn same_one(_: &Context, ty: Type) -> Type {
    ty
}

/// The types every operation of the same-type kind is tested at: `i32` and one other.
const TYPES: [(TypeFn, &str); 2] = [(Context::i32, "i32"), (Context::i64, "i64")];
const OTHER_TYPES: [TypeFn; 3] = [Context::i8, Context::i64, Context::isize];

type BinaryBuilder = fn(Builder<'_, '_>, ValueRef, ValueRef) -> Result<ValueRef, Report>;
type UnaryBuilder = fn(Builder<'_, '_>, ValueRef) -> Result<ValueRef, Report>;

#[test]
fn same_type_binary_operations_round_trip() {
    let ops: [(&str, BinaryBuilder); 14] = [
        ("arith.add", |b, l, r| b.add(l, r, SPAN)),
        ("arith.sub", |b, l, r| b.sub(l, r, SPAN)),
        ("arith.mul", |b, l, r| b.mul(l, r, SPAN)),
        ("arith.udiv", |b, l, r| b.udiv(l, r, SPAN)),
        ("arith.sdiv", |b, l, r| b.sdiv(l, r, SPAN)),
        ("arith.urem", |b, l, r| b.urem(l, r, SPAN)),
        ("arith.srem", |b, l, r| b.srem(l, r, SPAN)),
        ("arith.band", |b, l, r| b.band(l, r, SPAN)),
        ("arith.bor", |b, l, r| b.bor(l, r, SPAN)),
        ("arith.bxor", |b, l, r| b.bxor(l, r, SPAN)),
        ("arith.shl", |b, l, r| b.shl(l, r, SPAN)),
        ("arith.shr", |b, l, r| b.shr(l, r, SPAN)),
        ("arith.ashr", |b, l, r| b.ashr(l, r, SPAN)),
        ("arith.rotl", |b, l, r| b.rotl(l, r, SPAN)),
    ];
    let ops = ops
        .into_iter()
        .chain([("arith.rotr", (|b, l, r| b.rotr(l, r, SPAN)) as BinaryBuilder)]);
    for (i, (name, build)) in ops.enumerate() {
        for ty in [Context::i32, OTHER_TYPES[i % OTHER_TYPES.len()]] {
            let text = binary_round_trip(ty, |b, l, r| vec![build(b, l, r).unwrap()], same);
            assert!(text.contains(&format!("= {name} %0, %1")), "{text}");
        }
    }
}

#[test]
fn bitwise_operations_take_booleans() {
    let ops: [(&str, BinaryBuilder); 3] = [
        ("arith.band", |b, l, r| b.band(l, r, SPAN)),
        ("arith.bor", |b, l, r| b.bor(l, r, SPAN)),
        ("arith.bxor", |b, l, r| b.bxor(l, r, SPAN)),
    ];
    for (name, build) in ops {
        let text = binary_round_trip(Context::i1, |b, l, r| vec![build(b, l, r).unwrap()], same);
        assert!(text.contains(name), "{text}");
    }
    let text = unary_round_trip(Context::i1, |b, v| b.bnot(v, SPAN).unwrap(), same_one);
    assert!(text.contains("arith.bnot %0"), "{text}");
}

#[test]
fn add_round_trips_with_each_overflow_mode() {
    use Signedness::*;
    let modes: [(&str, BinaryBuilder); 5] = [
        ("wrapping", |b, l, r| b.add(l, r, SPAN)),
        ("checked signed", |b, l, r| b.add_checked(l, r, Signed, SPAN)),
        ("checked unsigned", |b, l, r| b.add_checked(l, r, Unsigned, SPAN)),
        ("unchecked signed", |b, l, r| b.add_unchecked(l, r, Signed, SPAN)),
        ("unchecked unsigned", |b, l, r| b.add_unchecked(l, r, Unsigned, SPAN)),
    ];
    for (ty, _) in TYPES {
        for (mode, build) in modes {
            let text = binary_round_trip(ty, |b, l, r| vec![build(b, l, r).unwrap()], same);
            assert!(
                text.contains(&format!(
                    "arith.add %0, %1 <{{ overflow = #builtin.overflow<{mode}> }}>"
                )),
                "{text}"
            );
        }
    }
}

#[test]
fn sub_and_mul_build_each_overflow_mode() {
    use Signedness::*;
    let modes: [(&str, BinaryBuilder); 6] = [
        ("arith.sub %0, %1 <{ overflow = #builtin.overflow<wrapping> }>", |b, l, r| {
            b.sub(l, r, SPAN)
        }),
        (
            "arith.sub %0, %1 <{ overflow = #builtin.overflow<checked signed> }>",
            |b, l, r| b.sub_checked(l, r, Signed, SPAN),
        ),
        (
            "arith.sub %0, %1 <{ overflow = #builtin.overflow<unchecked unsigned> }>",
            |b, l, r| b.sub_unchecked(l, r, Unsigned, SPAN),
        ),
        ("arith.mul %0, %1 <{ overflow = #builtin.overflow<wrapping> }>", |b, l, r| {
            b.mul(l, r, SPAN)
        }),
        (
            "arith.mul %0, %1 <{ overflow = #builtin.overflow<checked unsigned> }>",
            |b, l, r| b.mul_checked(l, r, Unsigned, SPAN),
        ),
        (
            "arith.mul %0, %1 <{ overflow = #builtin.overflow<unchecked signed> }>",
            |b, l, r| b.mul_unchecked(l, r, Signed, SPAN),
        ),
    ];
    for (expected, build) in modes {
        let text = binary_round_trip(Context::i16, |b, l, r| vec![build(b, l, r).unwrap()], same);
        assert!(text.contains(expected), "{text}");
    }
}

#[test]
fn overflowing_operations_return_the_flag_then_the_value() {
    type PairBuilder =
        fn(Builder<'_, '_>, ValueRef, ValueRef, Signedness) -> Result<(ValueRef, ValueRef), Report>;
    let ops: [(&str, PairBuilder); 3] = [
        ("arith.add_overflowing", |b, l, r, s| b.add_overflowing(l, r, s, SPAN)),
        ("arith.sub_overflowing", |b, l, r, s| b.sub_overflowing(l, r, s, SPAN)),
        ("arith.mul_overflowing", |b, l, r, s| b.mul_overflowing(l, r, s, SPAN)),
    ];
    for (name, build) in ops {
        for (ty, _) in TYPES {
            for signedness in [Signedness::Signed, Signedness::Unsigned] {
                let text = binary_round_trip(
                    ty,
                    |b, l, r| {
                        let (flag, value) = build(b, l, r, signedness).unwrap();
                        vec![flag, value]
                    },
                    |cx, ty| vec![cx.i1(), ty],
                );
                assert!(
                    text.contains(&format!(
                        "%2, %3 = {name} %0, %1 <{{ signedness = \
                         #builtin.signedness<{signedness}> }}>"
                    )),
                    "{text}"
                );
            }
        }
    }
}

#[test]
fn overflowing_signedness_is_the_parsed_property() {
    let mut test = test_with(&[Context::i8, Context::i8]);
    let [lhs, rhs] = params(&test)[..] else {
        unreachable!()
    };
    test.function_builder()
        .mul_overflowing(lhs, rhs, Signedness::Unsigned, SPAN)
        .unwrap();
    let (_, _context, ops) = finish(&mut test);
    let op = ops[0].borrow();
    let op = op.downcast_ref::<MulOverflowing>().unwrap();
    assert_eq!(*op.get_signedness(), Signedness::Unsigned);
}

#[test]
fn division_with_remainder_returns_the_quotient_then_the_remainder() {
    type PairBuilder =
        fn(Builder<'_, '_>, ValueRef, ValueRef) -> Result<(ValueRef, ValueRef), Report>;
    let ops: [(&str, PairBuilder); 2] = [
        ("arith.udivrem", |b, l, r| b.udivrem(l, r, SPAN)),
        ("arith.sdivrem", |b, l, r| b.sdivrem(l, r, SPAN)),
    ];
    for (name, build) in ops {
        for ty in [Context::i32, Context::isize] {
            let mut test = test_with(&[ty, ty]);
            let [lhs, rhs] = params(&test)[..] else {
                unreachable!()
            };
            let (quotient, remainder) = build(&mut test.function_builder(), lhs, rhs).unwrap();
            let op = quotient.borrow().get_defining_op().unwrap();
            {
                let op = op.borrow();
                assert_eq!(op.results()[0].borrow().as_value_ref(), quotient);
                assert_eq!(op.results()[1].borrow().as_value_ref(), remainder);
                let (q, r) = if name == "arith.udivrem" {
                    let op = op.downcast_ref::<Udivrem>().unwrap();
                    (op.quotient().as_value_ref(), op.remainder().as_value_ref())
                } else {
                    let op = op.downcast_ref::<Sdivrem>().unwrap();
                    (op.quotient().as_value_ref(), op.remainder().as_value_ref())
                };
                assert_eq!((q, r), (quotient, remainder));
            }
            let operand_ty = ty(test.context());
            assert_eq!(result_types(op), [operand_ty, operand_ty]);
            let (text, context, ops) = finish(&mut test);
            assert!(text.contains(&format!("%2, %3 = {name} %0, %1")), "{text}");
            assert_eq!(result_types(ops[0]), [ty(&context), ty(&context)]);
        }
    }
}

#[test]
fn min_and_max_round_trip_with_each_signedness() {
    type SignedBuilder =
        fn(Builder<'_, '_>, ValueRef, ValueRef, Signedness) -> Result<ValueRef, Report>;
    let ops: [(&str, SignedBuilder); 2] = [
        ("arith.min", |b, l, r, s| b.min(l, r, s, SPAN)),
        ("arith.max", |b, l, r, s| b.max(l, r, s, SPAN)),
    ];
    for (name, build) in ops {
        for ty in [Context::i32, Context::isize] {
            for signedness in [Signedness::Signed, Signedness::Unsigned] {
                let text = binary_round_trip(
                    ty,
                    |b, l, r| vec![build(b, l, r, signedness).unwrap()],
                    same,
                );
                assert!(
                    text.contains(&format!(
                        "{name} %0, %1 <{{ signedness = #builtin.signedness<{signedness}> }}>"
                    )),
                    "{text}"
                );
            }
        }
    }
}

#[test]
fn unary_operations_have_the_operand_type() {
    let ops: [(&str, UnaryBuilder); 7] = [
        ("arith.neg", |b, v| b.neg(v, SPAN)),
        ("arith.bnot", |b, v| b.bnot(v, SPAN)),
        ("arith.popcnt", |b, v| b.popcnt(v, SPAN)),
        ("arith.clz", |b, v| b.clz(v, SPAN)),
        ("arith.ctz", |b, v| b.ctz(v, SPAN)),
        ("arith.clo", |b, v| b.clo(v, SPAN)),
        ("arith.cto", |b, v| b.cto(v, SPAN)),
    ];
    for (i, (name, build)) in ops.into_iter().enumerate() {
        for ty in [Context::i32, OTHER_TYPES[i % OTHER_TYPES.len()], Context::i128] {
            let text = unary_round_trip(ty, |b, v| build(b, v).unwrap(), same_one);
            assert!(text.contains(&format!("%2 = {name} %0;")), "{text}");
        }
    }
}

#[test]
fn a_shift_amount_has_the_type_of_the_value() {
    let ops: [BinaryBuilder; 5] = [
        |b, l, r| b.shl(l, r, SPAN),
        |b, l, r| b.shr(l, r, SPAN),
        |b, l, r| b.ashr(l, r, SPAN),
        |b, l, r| b.rotl(l, r, SPAN),
        |b, l, r| b.rotr(l, r, SPAN),
    ];
    for build in ops {
        let mut test = test_with(&[Context::i64, Context::i32]);
        let [value, amount] = params(&test)[..] else {
            unreachable!()
        };
        let result = build(&mut test.function_builder(), value, amount).unwrap();
        let op = result.borrow().get_defining_op().unwrap();
        let err = op.borrow().verify().expect_err("the amount has another type");
        let messages = format!("{err} {:?}", label_messages(err.as_diagnostic()));
        assert!(messages.contains("invalid operation"), "{messages}");
    }
}

#[test]
fn cmpi_prints_its_predicate_as_a_keyword() {
    for predicate in CmpPredicate::ALL {
        for (ty, _) in [(Context::i32 as TypeFn, ()), (Context::isize, ())] {
            let text = binary_round_trip(
                ty,
                |b, l, r| vec![b.cmpi(predicate, l, r, SPAN).unwrap()],
                |cx, _| vec![cx.i1()],
            );
            assert!(text.contains(&format!("%2 = arith.cmpi {predicate} %0, %1;")), "{text}");
        }
    }
}

#[test]
fn cmpi_parses_each_predicate() {
    for predicate in CmpPredicate::ALL {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<ArithDialect>();
        let text = format!(
            "builtin.function public extern(\"C\") @f(%0: i8, %1: i8) {{\n    %2 = arith.cmpi \
             {predicate} %0, %1;\n    builtin.ret;\n}};"
        );
        let (function, printed) = parse_function_fixpoint(&context, "f", &text).expect("parses");
        assert_eq!(printed, text);
        let op = ops_of(function)[0];
        let op = op.borrow();
        let cmpi = op.downcast_ref::<Cmpi>().expect("an arith.cmpi");
        assert_eq!(*cmpi.get_predicate(), predicate);
        assert_eq!(cmpi.result().ty(), context.i1());
    }
}

#[test]
fn predicate_helpers_build_cmpi() {
    let helpers: [(CmpPredicate, BinaryBuilder); 10] = [
        (CmpPredicate::Eq, |b, l, r| b.eq(l, r, SPAN)),
        (CmpPredicate::Ne, |b, l, r| b.ne(l, r, SPAN)),
        (CmpPredicate::Slt, |b, l, r| b.slt(l, r, SPAN)),
        (CmpPredicate::Sle, |b, l, r| b.sle(l, r, SPAN)),
        (CmpPredicate::Sgt, |b, l, r| b.sgt(l, r, SPAN)),
        (CmpPredicate::Sge, |b, l, r| b.sge(l, r, SPAN)),
        (CmpPredicate::Ult, |b, l, r| b.ult(l, r, SPAN)),
        (CmpPredicate::Ule, |b, l, r| b.ule(l, r, SPAN)),
        (CmpPredicate::Ugt, |b, l, r| b.ugt(l, r, SPAN)),
        (CmpPredicate::Uge, |b, l, r| b.uge(l, r, SPAN)),
    ];
    for (predicate, build) in helpers {
        let text = binary_round_trip(
            Context::i16,
            |b, l, r| vec![build(b, l, r).unwrap()],
            |cx, _| vec![cx.i1()],
        );
        assert!(text.contains(&format!("arith.cmpi {predicate} %0, %1")), "{text}");
    }
}

#[test]
fn fixed_width_casts_round_trip() {
    type CastBuilder = fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>;
    let casts: [(&str, CastBuilder, TypeFn, TypeFn); 6] = [
        ("arith.trunc", |b, v, t| b.trunc(v, t, SPAN), Context::i64, Context::i32),
        ("arith.trunc", |b, v, t| b.trunc(v, t, SPAN), Context::i32, Context::i1),
        ("arith.zext", |b, v, t| b.zext(v, t, SPAN), Context::i32, Context::i64),
        ("arith.zext", |b, v, t| b.zext(v, t, SPAN), Context::i1, Context::i8),
        ("arith.sext", |b, v, t| b.sext(v, t, SPAN), Context::i32, Context::i64),
        ("arith.sext", |b, v, t| b.sext(v, t, SPAN), Context::i8, Context::i128),
    ];
    for (name, build, from, to) in casts {
        let text = binary_round_trip(
            from,
            |b, v, _| {
                let to = to(b.builder().context());
                vec![build(b, v, to).unwrap()]
            },
            |cx, _| vec![to(cx)],
        );
        assert!(text.contains(&format!("= {name} %0")), "{text}");
    }
}

#[test]
fn isize_conversions_round_trip() {
    type CastBuilder = fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>;
    let casts: [(&str, CastBuilder); 2] = [
        ("arith.sext_or_trunc", |b, v, t| b.sext_or_trunc(v, t, SPAN)),
        ("arith.zext_or_trunc", |b, v, t| b.zext_or_trunc(v, t, SPAN)),
    ];
    let pairs: [(TypeFn, TypeFn); 4] = [
        (Context::isize, Context::i32),
        (Context::i32, Context::isize),
        (Context::i64, Context::isize),
        (Context::isize, Context::i64),
    ];
    for (name, build) in casts {
        for (from, to) in pairs {
            let text = binary_round_trip(
                from,
                |b, v, _| {
                    let to = to(b.builder().context());
                    vec![build(b, v, to).unwrap()]
                },
                |cx, _| vec![to(cx)],
            );
            assert!(text.contains(&format!("= {name} %0")), "{text}");
        }
    }
}

/// Builds the cast `build` from a parameter of type `from` to `to` and returns the label messages
/// of its verification failure.
fn invalid_cast(
    build: fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>,
    from: TypeFn,
    to: TypeFn,
) -> Vec<String> {
    let mut test = test_with(&[from]);
    let [value] = params(&test)[..] else {
        unreachable!()
    };
    let to = to(test.context());
    let result = build(&mut test.function_builder(), value, to).unwrap();
    let op = result.borrow().get_defining_op().unwrap();
    let err = op.borrow().verify().expect_err("an invalid cast");
    assert_eq!(err.to_string(), "invalid operation");
    label_messages(err.as_diagnostic())
}

#[test]
fn casts_verify_their_result_type_against_their_attribute() {
    type CastBuilder = fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>;
    let casts: [(&str, CastBuilder, TypeFn, TypeFn); 5] = [
        ("trunc", |b, v, t| b.trunc(v, t, SPAN), Context::i32, Context::i8),
        ("zext", |b, v, t| b.zext(v, t, SPAN), Context::i8, Context::i32),
        ("sext", |b, v, t| b.sext(v, t, SPAN), Context::i8, Context::i32),
        (
            "sext_or_trunc",
            |b, v, t| b.sext_or_trunc(v, t, SPAN),
            Context::isize,
            Context::i32,
        ),
        (
            "zext_or_trunc",
            |b, v, t| b.zext_or_trunc(v, t, SPAN),
            Context::i32,
            Context::isize,
        ),
    ];
    for (name, build, from, to) in casts {
        let mut test = test_with(&[from]);
        let [value] = params(&test)[..] else {
            unreachable!()
        };
        let context = test.context_rc();
        let (to, other) = (to(&context), context.i16());
        let result = build(&mut test.function_builder(), value, to).unwrap();
        let mut op = result.borrow().get_defining_op().unwrap();
        op.borrow().verify().expect("a valid cast");
        {
            let mut op = op.borrow_mut();
            match name {
                "trunc" => op.downcast_mut::<Trunc>().unwrap().set_ty(other),
                "zext" => op.downcast_mut::<Zext>().unwrap().set_ty(other),
                "sext" => op.downcast_mut::<Sext>().unwrap().set_ty(other),
                "sext_or_trunc" => op.downcast_mut::<SextOrTrunc>().unwrap().set_ty(other),
                _ => op.downcast_mut::<ZextOrTrunc>().unwrap().set_ty(other),
            }
        }
        let err = op.borrow().verify().expect_err("the attribute and the result differ");
        assert_eq!(err.to_string(), "invalid operation");
        let messages = label_messages(err.as_diagnostic());
        assert_eq!(messages.len(), 1);
        assert!(
            messages[0]
                .contains(&format!("its result type `{to}` is not its `ty` attribute `{other}`")),
            "{messages:?}"
        );
    }
}

#[test]
fn overflowing_operations_verify_the_type_of_their_value_result() {
    type OverflowBuilder =
        fn(Builder<'_, '_>, ValueRef, ValueRef, Signedness) -> Result<(ValueRef, ValueRef), Report>;
    let ops: [(&str, OverflowBuilder); 3] = [
        ("arith.add_overflowing", |b, l, r, s| b.add_overflowing(l, r, s, SPAN)),
        ("arith.sub_overflowing", |b, l, r, s| b.sub_overflowing(l, r, s, SPAN)),
        ("arith.mul_overflowing", |b, l, r, s| b.mul_overflowing(l, r, s, SPAN)),
    ];
    for (name, build) in ops {
        let mut test = test_with(&[Context::i32, Context::i32]);
        let [lhs, rhs] = params(&test)[..] else {
            unreachable!()
        };
        let context = test.context_rc();
        let (_, mut value) =
            build(&mut test.function_builder(), lhs, rhs, Signedness::Signed).unwrap();
        let op = value.borrow().get_defining_op().unwrap();
        op.borrow().verify().expect("a valid operation");
        value.borrow_mut().set_type(context.i64());
        let err = op.borrow().verify().expect_err("the value result has another type");
        assert_eq!(err.to_string(), "invalid operation");
        let messages = label_messages(err.as_diagnostic());
        assert_eq!(messages.len(), 1);
        assert!(
            messages[0].contains("has type `i64`, but its operands have type `i32`")
                && messages[0].contains(name),
            "{messages:?}"
        );
    }
}

#[test]
fn fixed_width_casts_check_the_width_order() {
    let messages = invalid_cast(|b, v, t| b.trunc(v, t, SPAN), Context::i32, Context::i64);
    assert_eq!(
        messages,
        ["invalid 'arith.trunc' from `i32` to `i64`: `i64` is not narrower than `i32`"]
    );
    let messages = invalid_cast(|b, v, t| b.zext(v, t, SPAN), Context::i64, Context::i32);
    assert_eq!(
        messages,
        ["invalid 'arith.zext' from `i64` to `i32`: `i32` is not wider than `i64`"]
    );
    let messages = invalid_cast(|b, v, t| b.sext(v, t, SPAN), Context::i32, Context::i32);
    assert_eq!(
        messages,
        ["invalid 'arith.sext' from `i32` to `i32`: `i32` is not wider than `i32`"]
    );
}

#[test]
fn fixed_width_casts_reject_isize() {
    type CastBuilder = fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>;
    let casts: [(&str, CastBuilder); 3] = [
        ("arith.trunc", |b, v, t| b.trunc(v, t, SPAN)),
        ("arith.zext", |b, v, t| b.zext(v, t, SPAN)),
        ("arith.sext", |b, v, t| b.sext(v, t, SPAN)),
    ];
    let pairs: [(TypeFn, TypeFn, &str, &str); 3] = [
        (Context::isize, Context::i32, "isize", "i32"),
        (Context::i32, Context::isize, "i32", "isize"),
        (Context::isize, Context::isize, "isize", "isize"),
    ];
    for (name, build) in casts {
        for (from, to, from_name, to_name) in pairs {
            let messages = invalid_cast(build, from, to);
            assert_eq!(messages.len(), 1);
            assert!(
                messages[0]
                    .starts_with(&format!("invalid '{name}' from `{from_name}` to `{to_name}`")),
                "{messages:?}"
            );
            assert!(messages[0].contains("sext_or_trunc"), "{messages:?}");
        }
    }
}

#[test]
fn isize_conversions_need_exactly_one_isize() {
    type CastBuilder = fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>;
    let casts: [(&str, CastBuilder); 2] = [
        ("arith.sext_or_trunc", |b, v, t| b.sext_or_trunc(v, t, SPAN)),
        ("arith.zext_or_trunc", |b, v, t| b.zext_or_trunc(v, t, SPAN)),
    ];
    let pairs: [(TypeFn, TypeFn, &str, &str); 3] = [
        (Context::i32, Context::i64, "i32", "i64"),
        (Context::isize, Context::isize, "isize", "isize"),
        (Context::i64, Context::i64, "i64", "i64"),
    ];
    for (name, build) in casts {
        for (from, to, from_name, to_name) in pairs {
            let messages = invalid_cast(build, from, to);
            assert_eq!(
                messages,
                [format!(
                    "invalid '{name}' from `{from_name}` to `{to_name}`: exactly one of its types \
                     must be `isize` and the other fixed-width, but it converts `{from_name}` to \
                     `{to_name}`"
                )]
            );
        }
    }
}

#[test]
fn constants_parse_print_and_parse_again() {
    for (source, printed) in [
        ("-1 : i7", "-1 : i7"),
        ("255 : i8", "-1 : i8"),
        ("1 : i1", "1 : i1"),
        ("5 : i128", "5 : i128"),
        ("-1 : isize", "-1 : isize"),
    ] {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<ArithDialect>();
        let text = format!(
            "builtin.function public extern(\"C\") @f() {{\n    %0 = arith.constant \
             {source};\n    builtin.ret;\n}};"
        );
        let (function, reprinted) = parse_function_fixpoint(&context, "f", &text).expect("parses");
        assert!(reprinted.contains(&format!("%0 = arith.constant {printed};")), "{reprinted}");
        let op = ops_of(function)[0];
        let op = op.borrow();
        let constant = op.downcast_ref::<Constant>().expect("an arith.constant");
        assert_eq!(constant.value().ty(), constant.result().ty());
    }
}

#[test]
fn constant_builders_type_their_constants() {
    let mut test = test_with(&[]);
    let context = test.context_rc();
    let values = {
        let mut b = test.function_builder();
        [
            (b.i1(true, SPAN), context.i1(), Int::from(true)),
            (b.i8(-1, SPAN), context.i8(), Int::from(-1i8)),
            (b.i16(-2, SPAN), context.i16(), Int::from(-2i16)),
            (b.i32(3, SPAN), context.i32(), Int::from(3i32)),
            (b.i64(-4, SPAN), context.i64(), Int::from(-4i64)),
            (b.i128(5, SPAN), context.i128(), Int::from(5i128)),
            (b.isize(-1, SPAN), context.isize(), Int::from(-1i64)),
            (
                b.int(Int::from_bits(7, 0x7f), context.int_type(7), SPAN),
                context.int_type(7),
                Int::from_bits(7, 0x7f),
            ),
        ]
    };
    for (value, ty, int) in values {
        assert_eq!(value.borrow().ty(), ty);
        let op = value.borrow().get_defining_op().unwrap();
        let op = op.borrow();
        let constant = op.downcast_ref::<Constant>().unwrap();
        assert_eq!(constant.value().ty(), ty);
        assert_eq!(*constant.get_value(), int);
    }
    let (text, _context, _) = finish(&mut test);
    assert!(text.contains("arith.constant -1 : isize"), "{text}");
    assert!(text.contains("arith.constant -1 : i7"), "{text}");
}

#[test]
fn materialize_constant_installs_an_int_attr_as_it_is() {
    let mut test = test_with(&[]);
    let context = test.context_rc();
    let dialect = context.get_or_register_dialect::<ArithDialect>();

    for ty in [context.i32(), context.isize()] {
        let attr =
            context.int_attr(Int::from(-3i32).sext(64).trunc(Int::width_for_type(ty).unwrap()), ty);
        let op = dialect
            .materialize_constant(test.builder_mut(), attr.as_attribute_ref(), ty, SPAN)
            .expect("materialized");
        let op = op.borrow();
        let installed = op.get_attribute("value").unwrap();
        assert!(AttributeRef::ptr_eq(&installed, &attr.as_attribute_ref()));
        let constant = op.downcast_ref::<Constant>().unwrap();
        assert_eq!(constant.result().ty(), ty);
    }

    let flag = context.create_attribute::<BoolAttr, _>(true).as_attribute_ref();
    let op = dialect
        .materialize_constant(test.builder_mut(), flag, context.i1(), SPAN)
        .expect("materialized");
    {
        let op = op.borrow();
        let constant = op.downcast_ref::<Constant>().unwrap();
        assert_eq!(*constant.get_value(), Int::from(true));
        assert_eq!(constant.result().ty(), context.i1());
        assert!(format!("{}", op).contains("arith.constant 1 : i1"));
    }

    let attr = context.int_attr(Int::from(1i32), context.i32()).as_attribute_ref();
    assert!(
        dialect
            .materialize_constant(test.builder_mut(), attr, context.i64(), SPAN)
            .is_none()
    );
    assert!(
        dialect
            .materialize_constant(test.builder_mut(), flag, context.i32(), SPAN)
            .is_none()
    );
}

/// Folds the operation defining `value` and returns the integer attribute it folds to.
fn fold_value(value: ValueRef, operands: Option<&[Option<AttributeRef>]>) -> Option<(Int, Type)> {
    let op = value.borrow().get_defining_op().unwrap();
    let op = op.borrow();
    let foldable = op.as_trait::<dyn Foldable>().unwrap();
    let mut results = SmallVec::<[OpFoldResult; 1]>::new();
    let outcome = match operands {
        Some(operands) => foldable.fold_with(operands, &mut results),
        None => foldable.fold(&mut results),
    };
    if !outcome.is_ok() {
        return None;
    }
    let [OpFoldResult::Attribute(folded)] = results.as_slice() else {
        panic!("expected a single folded attribute");
    };
    let folded = folded.borrow();
    let int = folded
        .as_attr()
        .as_trait::<dyn runic_core::attributes::IntegerLikeAttr>()
        .unwrap();
    Some((int.as_int(), int.ty()))
}

#[test]
fn coercion_folds_preserve_shared_constants() {
    type CastBuilder = fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>;
    let cases: [(CastBuilder, Int, TypeFn, Int); 4] = [
        (|b, v, t| b.trunc(v, t, SPAN), Int::from(257i32), Context::i8, Int::from(1i8)),
        (|b, v, t| b.zext(v, t, SPAN), Int::from(-1i8), Context::i32, Int::from(255i32)),
        (|b, v, t| b.sext(v, t, SPAN), Int::from(-1i8), Context::i32, Int::from(-1i32)),
        (|b, v, t| b.trunc(v, t, SPAN), Int::from(3i64), Context::i1, Int::from(true)),
    ];
    for (build, input, to, expected) in cases {
        for explicit_operands in [false, true] {
            let mut test = test_with(&[]);
            let context = test.context_rc();
            let mut b = test.function_builder();
            let input_ty = context.int_type(input.width());
            let source = b.int(input, input_ty, SPAN);
            let independent_use = b.add(source, source, SPAN).unwrap();
            let to = to(&context);
            let result = build(&mut b, source, to).unwrap();
            let constant = source.borrow().get_defining_op().unwrap();
            let attribute = constant.borrow().get_attribute("value").unwrap();
            let operands = [Some(attribute)];
            let folded =
                fold_value(result, explicit_operands.then_some(&operands[..])).expect("folds");
            assert_eq!(folded, (expected, to));
            let attribute = attribute.borrow();
            let int = attribute
                .as_attr()
                .as_trait::<dyn runic_core::attributes::IntegerLikeAttr>()
                .unwrap();
            assert_eq!(int.as_int(), input, "source constant changed");
            assert_eq!(source.borrow().ty(), input_ty);
            assert_eq!(independent_use.borrow().ty(), input_ty);
        }
    }
}

#[test]
fn a_cast_that_does_not_verify_does_not_fold() {
    type CastBuilder = fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>;
    let cases: [(CastBuilder, &str, TypeFn, TypeFn); 4] = [
        (
            |b, v, t| b.trunc(v, t, SPAN),
            "trunc isize -> i32",
            Context::isize,
            Context::i32,
        ),
        (
            |b, v, t| b.zext(v, t, SPAN),
            "zext isize -> i128",
            Context::isize,
            Context::i128,
        ),
        (|b, v, t| b.sext(v, t, SPAN), "sext i32 -> isize", Context::i32, Context::isize),
        (|b, v, t| b.trunc(v, t, SPAN), "trunc i32 -> i64", Context::i32, Context::i64),
    ];
    for (build, name, from, to) in cases {
        for explicit_operands in [false, true] {
            let mut test = test_with(&[]);
            let context = test.context_rc();
            let mut b = test.function_builder();
            let from_ty = from(&context);
            let source = if from_ty.is_isize() {
                b.isize(5, SPAN)
            } else {
                b.int(Int::from(5i32), from_ty, SPAN)
            };
            let result = build(&mut b, source, to(&context)).unwrap();
            let constant = source.borrow().get_defining_op().unwrap();
            let attribute = constant.borrow().get_attribute("value").unwrap();
            let operands = [Some(attribute)];
            let op = result.borrow().get_defining_op().unwrap();
            assert!(op.borrow().verify().is_err(), "{name} verifies");
            assert_eq!(
                fold_value(result, explicit_operands.then_some(&operands[..])),
                None,
                "{name} folded"
            );
        }
    }
}

#[test]
fn bnot_folds_a_constant_boolean_and_a_double_negation() {
    let mut test = test_with(&[Context::i8]);
    let [param] = params(&test)[..] else {
        unreachable!()
    };
    let mut b = test.function_builder();
    let t = b.i1(true, SPAN);
    let not_t = b.bnot(t, SPAN).unwrap();
    assert_eq!(fold_value(not_t, None), Some((Int::from(false), b.builder().context().i1())));
    // Only `i1` constants fold in this stage.
    let byte = b.i8(1, SPAN);
    let not_byte = b.bnot(byte, SPAN).unwrap();
    assert_eq!(fold_value(not_byte, None), None);
    // `bnot (bnot x)` is `x`.
    let inner = b.bnot(param, SPAN).unwrap();
    let outer = b.bnot(inner, SPAN).unwrap();
    let op = outer.borrow().get_defining_op().unwrap();
    let op = op.borrow();
    let mut results = SmallVec::<[OpFoldResult; 1]>::new();
    assert!(op.as_trait::<dyn Foldable>().unwrap().fold(&mut results).is_ok());
    assert!(matches!(results.as_slice(), [OpFoldResult::Value(v)] if *v == param));
}

#[test]
fn an_isize_constant_folds_to_its_own_attribute() {
    let mut test = test_with(&[]);
    let context = test.context_rc();
    let value = test.function_builder().isize(-1, SPAN);
    assert_eq!(fold_value(value, None), Some((Int::from(-1i64), context.isize())));
}

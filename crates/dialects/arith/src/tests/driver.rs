//! Folding through the real drivers: the greedy pattern driver and `OperationFolder`.

use runic_core::{
    OperationFolder, attributes::IntegerLikeAttr, dialects::builtin::Ret,
    patterns::NoopRewriterListener,
};
use runic_dialect_ub::UndefinedBehaviorOpBuilder;

use super::{model::M, *};

/// The names of the operations of `test`'s function, in order.
fn names(test: &Test) -> Vec<String> {
    ops_of(test.function())
        .iter()
        .map(|op| format!("{}", op.borrow().name()))
        .collect()
}

/// A function that mixes operations that fold with ones that must not: a division by a constant
/// zero, `MIN / -1`, an over-wide shift, a checked addition that overflows, an `isize` `clz`, an
/// `isize` comparison on which the targets disagree, a conversion from `isize` that loses bits on
/// one target, and an operation on `ub.poison`. The unfoldable ones survive with their operands,
/// the foldable ones become constants, the function verifies, and a second run changes nothing.
#[test]
fn unfoldable_operations_survive_canonicalization_beside_foldable_ones() {
    let mut test = Test::default();
    test.context().get_or_register_dialect::<ArithDialect>();
    let context = test.context_rc();
    let (i1, i32, i64, isize) = (context.i1(), context.i32(), context.i64(), context.isize());
    let result_tys = [i32, i32, i32, i32, i32, isize, i1, i64, i32, i32, i32, i32, isize, i1, i64];
    test.with_function("f", &[], &result_tys);
    {
        let mut b = test.function_builder();
        let (zero, one, minus_one) = (b.i32(0, SPAN), b.i32(1, SPAN), b.i32(-1, SPAN));
        let (min, max, forty) = (b.i32(i32::MIN, SPAN), b.i32(i32::MAX, SPAN), b.i32(40, SPAN));
        let (big, small) = (b.isize(1 << 32, SPAN), b.isize(1, SPAN));
        let poison = b.poison(i32, SPAN);
        // Must not fold.
        let by_zero = b.udiv(one, zero, SPAN).unwrap();
        let overflow = b.sdiv(min, minus_one, SPAN).unwrap();
        let wide_shift = b.shl(one, forty, SPAN).unwrap();
        let checked = b.add_checked(max, one, Signedness::Signed, SPAN).unwrap();
        let rem_by_zero = b.srem(one, zero, SPAN).unwrap();
        let clz = b.clz(small, SPAN).unwrap();
        let compare = b.ult(big, small, SPAN).unwrap();
        let narrow = b.sext_or_trunc(big, i64, SPAN).unwrap();
        let of_poison = b.add(poison, one, SPAN).unwrap();
        // Must fold.
        let sum = b.add(max, one, SPAN).unwrap();
        let quotient = b.sdiv(min, one, SPAN).unwrap();
        let shifted = b.shl(one, one, SPAN).unwrap();
        let isize_sum = b.add(big, small, SPAN).unwrap();
        let isize_compare = b.eq(small, small, SPAN).unwrap();
        let widened = b.sext_or_trunc(small, i64, SPAN).unwrap();
        b.ret(
            [
                by_zero,
                overflow,
                wide_shift,
                checked,
                rem_by_zero,
                clz,
                compare,
                narrow,
                of_poison,
                sum,
                quotient,
                shifted,
                isize_sum,
                isize_compare,
                widened,
            ],
            SPAN,
        )
        .unwrap();
    }
    test.function().borrow().as_operation().verify().expect("valid before");
    assert!(canonicalize(&test));
    test.function().borrow().as_operation().verify().expect("valid after");
    let text = text_of(&test);

    let ops = ops_of(test.function());
    let ret = ops.last().unwrap().borrow();
    assert!(ret.is::<Ret>());
    let returned: Vec<ValueRef> =
        ret.operands().all().iter().map(|o| o.borrow().as_value_ref()).collect();
    let name_of =
        |value: ValueRef| format!("{}", value.borrow().get_defining_op().unwrap().borrow().name());
    let survivors: Vec<String> = returned[..9].iter().map(|value| name_of(*value)).collect();
    assert_eq!(
        survivors,
        [
            "arith.udiv",
            "arith.sdiv",
            "arith.shl",
            "arith.add",
            "arith.srem",
            "arith.clz",
            "arith.cmpi",
            "arith.sext_or_trunc",
            "arith.add"
        ],
        "{text}"
    );
    let folded: Vec<(M, Type)> = returned[9..]
        .iter()
        .map(|value| {
            let op = value.borrow().get_defining_op().unwrap();
            let op = op.borrow();
            let constant = op.downcast_ref::<Constant>().unwrap_or_else(|| panic!("{text}"));
            (M::of_int(*constant.get_value()), constant.result().ty())
        })
        .collect();
    assert_eq!(
        folded,
        [
            (M::new(32, 0x8000_0000), i32),
            (M::new(32, 0x8000_0000), i32),
            (M::new(32, 2), i32),
            (M::new(64, (1 << 32) + 1), isize),
            (M::new(1, 1), i1),
            (M::new(64, 1), i64),
        ],
        "{text}"
    );
    // No operation is left that is not a constant, a survivor, the poison or the return.
    let allowed = [
        "arith.constant",
        "arith.udiv",
        "arith.sdiv",
        "arith.shl",
        "arith.add",
        "arith.srem",
        "arith.clz",
        "arith.cmpi",
        "arith.sext_or_trunc",
        "ub.poison",
        "builtin.ret",
    ];
    for name in names(&test) {
        assert!(allowed.contains(&name.as_str()), "unexpected {name} in:\n{text}");
    }
    assert_eq!(names(&test).iter().filter(|name| *name == "arith.add").count(), 2, "{text}");
    // A second run is a no-op.
    assert!(!canonicalize(&test), "a second canonicalization changed:\n{}", text_of(&test));
    assert_eq!(text_of(&test), text);
}

/// `OperationFolder` (the folder CSE-like passes use) materializes each distinct folded constant
/// once, through `ArithDialect::materialize_constant`; an `i64` and an `isize` of the same bits
/// stay two constants; the operand constants keep their attributes, untouched.
#[test]
fn the_operation_folder_uniques_folded_constants_by_value_and_type() {
    let mut test = test_fn(
        |_| vec![],
        |cx| vec![cx.i32(), cx.i32(), cx.i64(), cx.isize(), cx.i1(), cx.i1()],
    );
    let context = test.context_rc();
    let (operand_attrs, folded_ops): (Vec<(AttributeRef, M, Type)>, Vec<_>) = {
        let mut b = test.function_builder();
        let (one, two, three) = (b.i32(1, SPAN), b.i32(2, SPAN), b.i32(3, SPAN));
        let zero = b.i32(0, SPAN);
        let (wide_one, wide_two) = (b.i64(1, SPAN), b.i64(2, SPAN));
        let (size_one, size_two) = (b.isize(1, SPAN), b.isize(2, SPAN));
        // Two operations that fold to `3 : i32`, one to `3 : i64`, one to `3 : isize`, and two
        // comparisons that fold to `1 : i1`.
        let a = b.add(one, two, SPAN).unwrap();
        let c = b.sub(three, zero, SPAN).unwrap();
        let d = b.add(wide_one, wide_two, SPAN).unwrap();
        let e = b.add(size_one, size_two, SPAN).unwrap();
        let f = b.slt(one, two, SPAN).unwrap();
        let g = b.ult(size_one, size_two, SPAN).unwrap();
        b.ret([a, c, d, e, f, g], SPAN).unwrap();
        let attrs = [one, two, three, zero, wide_one, wide_two, size_one, size_two]
            .iter()
            .map(|value| {
                let op = value.borrow().get_defining_op().unwrap();
                let op = op.borrow();
                let constant = op.downcast_ref::<Constant>().unwrap();
                (
                    op.get_attribute("value").unwrap(),
                    M::of_int(*constant.get_value()),
                    constant.result().ty(),
                )
            })
            .collect();
        let ops = [a, c, d, e, f, g]
            .iter()
            .map(|v| v.borrow().get_defining_op().unwrap())
            .collect();
        (attrs, ops)
    };
    let mut folder = OperationFolder::new(context.clone(), NoopRewriterListener);
    for op in folded_ops {
        assert!(folder.try_fold(op).is_ok(), "{}", text_of(&test));
    }
    test.function().borrow().as_operation().verify().expect("valid after folding");
    let text = text_of(&test);
    let ops = ops_of(test.function());
    let ret = ops.last().unwrap().borrow();
    let returned: Vec<ValueRef> =
        ret.operands().all().iter().map(|o| o.borrow().as_value_ref()).collect();
    // The two `3 : i32` results are one value; the `i64` and `isize` ones are their own.
    assert_eq!(returned[0], returned[1], "{text}");
    assert_ne!(returned[2], returned[3], "{text}");
    assert_ne!(returned[0], returned[2], "{text}");
    assert_eq!(returned[4], returned[5], "{text}");
    let constant_of = |value: ValueRef| {
        let op = value.borrow().get_defining_op().unwrap();
        let op = op.borrow();
        let constant = op.downcast_ref::<Constant>().unwrap_or_else(|| panic!("{text}"));
        // The attribute and the result agree on the type, and the value has the type's width.
        assert_eq!(constant.value().ty(), constant.result().ty());
        (M::of_int(*constant.get_value()), constant.result().ty())
    };
    assert_eq!(constant_of(returned[0]), (M::new(32, 3), context.i32()));
    assert_eq!(constant_of(returned[2]), (M::new(64, 3), context.i64()));
    assert_eq!(constant_of(returned[3]), (M::new(64, 3), context.isize()));
    assert_eq!(constant_of(returned[4]), (M::new(1, 1), context.i1()));
    // No operand constant was edited in place.
    for (attr, value, ty) in operand_attrs {
        let attr = attr.borrow();
        let int = attr.as_attr().as_trait::<dyn IntegerLikeAttr>().unwrap();
        assert_eq!((M::of_int(int.as_int()), int.ty()), (value, ty));
    }
}

/// `materialize_constant` declines everything that is not an integer constant of exactly the
/// requested type: no widening, no narrowing, no `i64` for `isize`, no poison.
#[test]
fn materialize_constant_declines_every_mismatch() {
    let mut test = test_fn(|_| vec![], |_| vec![]);
    let context = test.context_rc();
    let dialect = context.get_or_register_dialect::<ArithDialect>();
    let (i32, i64, isize) = (context.i32(), context.i64(), context.isize());
    let five_i64 = context.int_attr(Int::from(5i64), i64).as_attribute_ref();
    let five_isize = context.int_attr(Int::from(5i64), isize).as_attribute_ref();
    let five_i32 = context.int_attr(Int::from(5i32), i32).as_attribute_ref();
    for (attr, ty, expect) in [
        (five_i64, i64, true),
        (five_isize, isize, true),
        (five_i64, isize, false),
        (five_isize, i64, false),
        (five_i32, i64, false),
        (five_i64, i32, false),
        (five_i32, context.ptr_type(i32), false),
    ] {
        let made = dialect.materialize_constant(test.builder_mut(), attr, ty, SPAN);
        assert_eq!(made.is_some(), expect, "{ty}");
        if let Some(op) = made {
            assert!(AttributeRef::ptr_eq(&op.borrow().get_attribute("value").unwrap(), &attr));
        }
    }
}

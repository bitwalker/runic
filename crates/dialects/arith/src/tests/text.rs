//! Text round trips of every operation, and the parser of the predicate attribute.

use super::*;

/// The message of `report` and of every label attached to it.
fn describe(report: &Report) -> String {
    format!("{report} {:?}", label_messages(report.as_diagnostic()))
}

/// Parses `body` as the body of `f(params)`, returning the reprinted text or the error.
fn parse_body(params: &str, body: &str) -> Result<String, String> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<ArithDialect>();
    let text = format!(
        "builtin.function public extern(\"C\") @f({params}) {{\n{body}\n    builtin.ret;\n}};"
    );
    let (function, printed) =
        parse_function_fixpoint(&context, "f", &text).map_err(|report| describe(&report))?;
    function
        .borrow()
        .as_operation()
        .verify()
        .map_err(|err| format!("does not verify: {}", describe(&err)))?;
    Ok(printed)
}

/// Every operation, built at `i1`, `i7`, `i32`, `i64`, `i128` and `isize`, in every mode: the
/// function verifies, prints, parses and prints the same, and the parsed operations have the
/// built ones' result types.
#[test]
fn every_operation_round_trips_at_every_kind_of_type() {
    let types: [TypeFn; 6] = [
        Context::i1,
        |cx| cx.int_type(7),
        Context::i32,
        Context::i64,
        Context::i128,
        Context::isize,
    ];
    let signs = [Signedness::Signed, Signedness::Unsigned];
    for ty in types {
        let mut test = test_with(&[ty, ty]);
        let context = test.context_rc();
        let ty = ty(&context);
        let [x, y] = params(&test)[..] else {
            unreachable!()
        };
        let mut expect: Vec<String> = Vec::new();
        {
            let mut b = test.function_builder();
            b.add(x, y, SPAN).unwrap();
            b.sub(x, y, SPAN).unwrap();
            b.mul(x, y, SPAN).unwrap();
            // The wrapping operations: `overflow` at its default is not printed
            for op in ["add", "sub", "mul"] {
                expect.push(format!("= arith.{op} %0, %1;"));
            }
            for sign in signs {
                b.add_checked(x, y, sign, SPAN).unwrap();
                b.add_unchecked(x, y, sign, SPAN).unwrap();
                b.add_overflowing(x, y, sign, SPAN).unwrap();
                b.sub_checked(x, y, sign, SPAN).unwrap();
                b.sub_unchecked(x, y, sign, SPAN).unwrap();
                b.sub_overflowing(x, y, sign, SPAN).unwrap();
                b.mul_checked(x, y, sign, SPAN).unwrap();
                b.mul_unchecked(x, y, sign, SPAN).unwrap();
                b.mul_overflowing(x, y, sign, SPAN).unwrap();
                b.min(x, y, sign, SPAN).unwrap();
                b.max(x, y, sign, SPAN).unwrap();
                for mode in ["checked", "unchecked"] {
                    for op in ["add", "sub", "mul"] {
                        expect.push(format!(
                            "arith.{op} %0, %1 <{{ overflow = #builtin.overflow<{mode} {sign}> \
                             }}>;"
                        ));
                    }
                }
                for op in ["add_overflowing", "sub_overflowing", "mul_overflowing", "min", "max"] {
                    expect.push(format!(
                        "arith.{op} %0, %1 <{{ signedness = #builtin.signedness<{sign}> }}>;"
                    ));
                }
            }
            b.udiv(x, y, SPAN).unwrap();
            b.sdiv(x, y, SPAN).unwrap();
            b.urem(x, y, SPAN).unwrap();
            b.srem(x, y, SPAN).unwrap();
            b.udivrem(x, y, SPAN).unwrap();
            b.sdivrem(x, y, SPAN).unwrap();
            b.band(x, y, SPAN).unwrap();
            b.bor(x, y, SPAN).unwrap();
            b.bxor(x, y, SPAN).unwrap();
            b.bnot(x, SPAN).unwrap();
            b.shl(x, y, SPAN).unwrap();
            b.shr(x, y, SPAN).unwrap();
            b.ashr(x, y, SPAN).unwrap();
            b.rotl(x, y, SPAN).unwrap();
            b.rotr(x, y, SPAN).unwrap();
            for predicate in CmpPredicate::ALL {
                b.cmpi(predicate, x, y, SPAN).unwrap();
                expect.push(format!("= arith.cmpi {predicate} %0, %1;"));
            }
            b.neg(x, SPAN).unwrap();
            b.popcnt(x, SPAN).unwrap();
            b.clz(x, SPAN).unwrap();
            b.ctz(x, SPAN).unwrap();
            b.clo(x, SPAN).unwrap();
            b.cto(x, SPAN).unwrap();
            for name in [
                "udiv", "sdiv", "urem", "srem", "udivrem", "sdivrem", "band", "bor", "bxor", "shl",
                "shr", "ashr", "rotl", "rotr",
            ] {
                expect.push(format!("= arith.{name} %0, %1;"));
            }
            for name in ["bnot", "neg", "popcnt", "clz", "ctz", "clo", "cto"] {
                expect.push(format!("= arith.{name} %0;"));
            }
            // The casts this type can be the source of.
            let width = ty.integer_width();
            match width {
                None => {
                    for to in [context.i1(), context.i32(), context.i64(), context.i128()] {
                        b.sext_or_trunc(x, to, SPAN).unwrap();
                        b.zext_or_trunc(x, to, SPAN).unwrap();
                    }
                }
                Some(width) => {
                    b.sext_or_trunc(x, context.isize(), SPAN).unwrap();
                    b.zext_or_trunc(x, context.isize(), SPAN).unwrap();
                    if width > 1 {
                        b.trunc(x, context.i1(), SPAN).unwrap();
                    }
                    if width < 128 {
                        b.zext(x, context.i128(), SPAN).unwrap();
                        b.sext(x, context.i128(), SPAN).unwrap();
                    }
                }
            }
            // A constant of the type, at its extremes.
            let w = Int::width_for_type(ty).unwrap();
            for value in [Int::zero(w), Int::one(w), Int::all_ones(w), Int::signed_min(w)] {
                b.int(value, ty, SPAN);
            }
            b.ret([], SPAN).unwrap();
        }
        test.function().borrow().as_operation().verify().expect("valid function");
        let text = text_of(&test);
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<ArithDialect>();
        let (parsed, printed) = parse_function_fixpoint(&context, "f", &text).expect("parses");
        assert_eq!(printed, text);
        parsed.borrow().as_operation().verify().expect("the parsed function is valid");
        for line in &expect {
            assert!(text.contains(line.as_str()), "missing `{line}` in:\n{text}");
        }
        // The parsed operations carry the same result types as the built ones.
        let built = ops_of(test.function());
        let reparsed = ops_of(parsed);
        assert_eq!(built.len(), reparsed.len());
        for (a, b) in built.iter().zip(&reparsed) {
            let (a, b) = (a.borrow(), b.borrow());
            assert_eq!(a.name(), b.name());
            let tys = |op: &runic_ir::Operation| {
                op.results().iter().map(|r| format!("{}", r.borrow().ty())).collect::<Vec<_>>()
            };
            assert_eq!(tys(&a), tys(&b), "{}", a.name());
        }
    }
}

/// The parser of `#arith.cmp_predicate<..>`, reached through an attribute dictionary (`arith.cmpi`
/// itself parses the bare keyword, not the attribute): every predicate round-trips, a body-less
/// attribute is `eq`, and an unknown predicate is an error.
#[test]
fn the_predicate_attribute_parses_through_an_attribute_dictionary() {
    for predicate in CmpPredicate::ALL {
        let body = format!(
            "    %2 = arith.band %0, %1 attributes {{ p = #arith.cmp_predicate<{predicate}> }};"
        );
        let printed = parse_body("%0: i32, %1: i32", &body).expect("parses");
        assert!(printed.contains(body.trim()), "{printed}");
    }
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<ArithDialect>();
    let text = "builtin.function public extern(\"C\") @f(%0: i32, %1: i32) {\n    %2 = arith.band \
                %0, %1 attributes { p = #arith.cmp_predicate };\n    builtin.ret;\n};";
    let (function, printed) = parse_function_fixpoint(&context, "f", text).expect("parses");
    let op = ops_of(function)[0];
    let attr = op.borrow().get_attribute("p").expect("the attribute");
    let attr = attr.borrow();
    let predicate = attr.downcast_ref::<CmpPredicateAttr>().expect("a predicate attribute");
    assert_eq!(*predicate.as_value(), CmpPredicate::Eq, "{printed}");
    for bad in ["lt", "neq", "SLT", ""] {
        let body =
            format!("    %2 = arith.band %0, %1 attributes {{ p = #arith.cmp_predicate<{bad}> }};");
        assert!(parse_body("%0: i32, %1: i32", &body).is_err(), "`{bad}` parsed");
    }
}

/// `overflow` on `add`/`sub`/`mul` defaults to wrapping: left out of the text, the operation
/// wraps, and the written default is not printed back. A `signedness` has no default and stays
/// required.
#[test]
fn overflow_defaults_to_wrapping_and_signedness_is_required() {
    for op in ["add", "sub", "mul"] {
        let body = format!("    %2 = arith.{op} %0, %1;");
        let printed = parse_body("%0: i32, %1: i32", &body).expect("parses");
        assert!(printed.contains(body.trim()), "{printed}");

        let explicit =
            format!("    %2 = arith.{op} %0, %1 <{{ overflow = #builtin.overflow<wrapping> }}>;");
        let printed = parse_body("%0: i32, %1: i32", &explicit).expect("parses");
        assert!(printed.contains(body.trim()), "{printed}");
    }

    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<ArithDialect>();
    let text = "builtin.function public extern(\"C\") @f(%0: i32, %1: i32) {\n    %2 = arith.add \
                %0, %1;\n    builtin.ret;\n};";
    let (function, _) = parse_function_fixpoint(&context, "f", text).expect("parses");
    let op = ops_of(function)[0];
    let op = op.borrow();
    assert_eq!(*op.downcast_ref::<Add>().unwrap().get_overflow(), runic_ir::Overflow::Wrapping);

    for op in ["min", "max", "add_overflowing"] {
        let err = parse_body("%0: i32, %1: i32", &format!("    %2 = arith.{op} %0, %1;"))
            .expect_err("a missing signedness is an error");
        assert!(err.contains("missing required property 'signedness'"), "{err}");
    }
}

/// The short builder form, a parsed operation without `overflow`, and an explicitly wrapping one
/// are the same operation: none has a property away from its default, the derived form prints no
/// dictionary, and the generic form prints the default.
#[test]
fn the_three_forms_of_the_default_agree() {
    use runic_ir::{
        BuilderExt, Overflow,
        testing::text::{print_custom, print_generic},
    };

    let mut test = Test::new("f", |cx| ([cx.i32(), cx.i32()], []));
    let [x, y] = params(&test)[..] else {
        unreachable!()
    };
    let (short, explicit) = {
        let mut b = test.function_builder();
        let short = b.builder_mut().create::<Mul, (_, _)>(SPAN)(x, y).unwrap();
        let explicit =
            b.builder_mut().create::<Mul, (_, _, _)>(SPAN)(x, y, Overflow::Wrapping).unwrap();
        b.ret([], SPAN).unwrap();
        (short, explicit)
    };
    assert_eq!(*short.borrow().get_overflow(), Overflow::Wrapping);
    assert_eq!(short.borrow().as_operation().non_default_properties().count(), 0);
    assert_eq!(explicit.borrow().as_operation().non_default_properties().count(), 0);
    assert_eq!(print_custom(short.borrow().as_operation()), "%2 = arith.mul %0, %1;");
    assert_eq!(
        print_generic(explicit.borrow().as_operation()),
        "%3 = \"arith.mul\"(%0, %1) <{ overflow = #builtin.overflow<wrapping> }> : (i32, i32) -> \
         i32;"
    );

    let (_cx, parsed) = runic_ir::testing::text::parse::<runic_ir::dialects::builtin::Function>(
        "builtin.function public extern(\"C\") @f(%0: i32, %1: i32) {\n    %2 = arith.mul %0, \
         %1;\n    builtin.ret;\n};",
    )
    .unwrap();
    let parsed = parsed.borrow();
    let entry = parsed.entry_block();
    let op = entry.borrow().front().unwrap();
    let op = op.borrow();
    assert_eq!(*op.downcast_ref::<Mul>().unwrap().get_overflow(), Overflow::Wrapping);
    assert_eq!(op.non_default_properties().count(), 0);
}

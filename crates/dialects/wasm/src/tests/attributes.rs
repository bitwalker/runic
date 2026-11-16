use alloc::{rc::Rc, vec};

use runic_ir::testing::elements::{Row, check_every_kind};

use super::*;

#[test]
fn limits_print_and_parse_with_and_without_a_maximum() {
    assert_eq!(round_trip_attribute("#wasm.limits<1>").unwrap(), "#wasm.limits<1>");
    assert_eq!(round_trip_attribute("#wasm.limits<1..16>").unwrap(), "#wasm.limits<1..16>");
    assert_eq!(round_trip_attribute("#wasm.limits<0..0>").unwrap(), "#wasm.limits<0..0>");
    assert_eq!(
        round_trip_attribute("#wasm.limits<0..18446744073709551615>").unwrap(),
        "#wasm.limits<0..18446744073709551615>"
    );
}

#[test]
fn limits_of_a_minimum_of_zero_with_no_maximum_are_the_default() {
    let context = context();
    let attr = context.create_attribute::<LimitsAttr, _>(Limits::default());
    assert_eq!(Limits::default(), Limits::at_least(0));
    assert_eq!(attr.borrow().min(), 0);
    assert_eq!(attr.borrow().max(), None);
    assert_eq!(round_trip_attribute("#wasm.limits").unwrap(), "#wasm.limits<0>");
}

#[test]
fn limits_with_a_maximum_below_the_minimum_do_not_verify() {
    let err = round_trip_attribute("#wasm.limits<16..1>").unwrap_err();
    assert!(
        err.contains(
            "limits `16..1` with a maximum below the minimum, but Wasm limits satisfy min <= max"
        ),
        "{err}"
    );
    assert!(err.contains("invalid operation"), "{err}");
}

#[test]
fn limits_with_a_malformed_body_are_a_diagnostic() {
    for text in [
        "#wasm.limits<>",
        "#wasm.limits<-1>",
        "#wasm.limits<1..>",
        "#wasm.limits<1.5>",
        "#wasm.limits<1.16>",
        "#wasm.limits<1...16>",
        "#wasm.limits<18446744073709551616>",
        "#wasm.limits<a>",
    ] {
        assert!(round_trip_attribute(text).is_err(), "{text}");
    }
}

#[test]
fn a_memarg_prints_and_parses_with_its_offset_and_alignment() {
    assert_eq!(
        round_trip_attribute("#wasm.memarg<offset = 8, align = 4>").unwrap(),
        "#wasm.memarg<offset = 8, align = 4>"
    );
    assert_eq!(
        round_trip_attribute("#wasm.memarg<offset = 0, align = 1>").unwrap(),
        "#wasm.memarg<offset = 0, align = 1>"
    );
    assert_eq!(
        round_trip_attribute("#wasm.memarg<offset = 4294967296, align = 1024>").unwrap(),
        "#wasm.memarg<offset = 4294967296, align = 1024>"
    );
}

#[test]
fn a_memarg_without_a_body_is_offset_zero_and_alignment_one() {
    let context = context();
    assert_eq!(MemArg::default(), MemArg::new(0, 1));
    assert_eq!(MemArg::default().offset(), 0);
    assert_eq!(MemArg::default().align(), 1);
    let _ = context;
    assert_eq!(
        round_trip_attribute("#wasm.memarg").unwrap(),
        "#wasm.memarg<offset = 0, align = 1>"
    );
}

#[test]
fn a_memarg_with_an_alignment_that_is_not_a_power_of_two_does_not_verify() {
    for align in [0, 3, 6, 12] {
        let err = round_trip_attribute(&format!("#wasm.memarg<offset = 0, align = {align}>"))
            .unwrap_err();
        assert!(
            err.contains(&format!(
                "an alignment of {align} bytes, but a Wasm memarg alignment is a power of two, at \
                 least 1"
            )),
            "{err}"
        );
    }
}

#[test]
fn a_memarg_with_a_malformed_body_is_a_diagnostic() {
    for text in [
        "#wasm.memarg<>",
        "#wasm.memarg<8, 4>",
        "#wasm.memarg<align = 4, offset = 8>",
        "#wasm.memarg<offset = 8>",
        "#wasm.memarg<offset = -8, align = 4>",
        "#wasm.memarg<offset = 8, align = 4, x = 1>",
        "#wasm.memarg<offset = 18446744073709551616, align = 4>",
    ] {
        assert!(round_trip_attribute(text).is_err(), "{text}");
    }
}

#[test]
fn mutability_prints_and_parses_its_two_keywords() {
    assert_eq!(
        round_trip_attribute("#wasm.mutability<const>").unwrap(),
        "#wasm.mutability<const>"
    );
    assert_eq!(round_trip_attribute("#wasm.mutability<mut>").unwrap(), "#wasm.mutability<mut>");
    assert_eq!(Mutability::default(), Mutability::Const);
    assert!(Mutability::Mut.is_mut());
    assert!(!Mutability::Const.is_mut());
    assert_eq!(round_trip_attribute("#wasm.mutability").unwrap(), "#wasm.mutability<const>");
}

#[test]
fn a_mutability_other_than_const_or_mut_is_a_diagnostic() {
    for text in ["#wasm.mutability<var>", "#wasm.mutability<1>", "#wasm.mutability<const mut>"] {
        assert!(round_trip_attribute(text).is_err(), "{text}");
    }
}

#[test]
fn every_attribute_kind_lists_and_replaces_what_it_holds() {
    let context = context();
    let rows = vec![
        Row::new("limits", context.create_attribute::<LimitsAttr, _>(Limits::between(1, 16)), []),
        Row::new("memarg", context.create_attribute::<MemArgAttr, _>(MemArg::new(8, 4)), []),
        Row::new("mutability", context.create_attribute::<MutabilityAttr, _>(Mutability::Mut), []),
    ];
    check_every_kind(&context, &["wasm"], &rows);
    let _: Rc<Context> = context;
}

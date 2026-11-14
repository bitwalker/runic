//! Termination and cost: a chain of N split additions, each reading the one before, converted in
//! program order.

use std::println;

use super::{harness::*, *};

/// `%0 = constant; %1 = constant; %i = add %{i-1}, %{i-2}` for `n` adds over `i128`, sunk at the
/// end.
fn chain(n: usize) -> String {
    let mut body =
        String::from("    %0 = test.constant 1 : i128;\n    %1 = test.constant 2 : i128;\n");
    for index in 2..n + 2 {
        body.push_str(&format!("    %{index} = test.add %{}, %{};\n", index - 1, index - 2));
    }
    body.push_str(&format!("    test.sink %{} : (i128);\n    builtin.ret;", n + 1));
    body
}

/// The number of operations under `function`, and of casts among them.
fn count_ops(function: OperationRef) -> (usize, usize) {
    let (mut ops, mut casts) = (0, 0);
    function.borrow().prewalk_all(|op: &Operation| {
        ops += 1;
        casts += usize::from(op.is::<UnrealizedConversionCast>());
    });
    (ops, casts)
}

/// Converts [chain] of `n` under a complete configuration; returns the result, the number of
/// operations left and the number of casts among them.
fn convert_chain(
    n: usize,
    reconcile: bool,
    limit: usize,
) -> (Result<ConversionResult, String>, usize, usize) {
    let (context, function) = parse_function("()", &chain(n));
    let mut rng = Rng(7);
    let mut config = Config::complete(&mut rng);
    config.full = true;
    config.reconcile = reconcile;
    let mut conversion = ConversionConfig::default();
    conversion
        .with_reconcile_unrealized_casts(reconcile)
        .with_max_rewrites_per_operation(limit);
    let violations: Seen<String> = Rc::default();
    let calls = Rc::new(core::cell::Cell::new(0));
    let result = run_conversion_with(function, &context, &config, &violations, &calls, conversion)
        .map_err(|err| format!("{err}"));
    let (ops, casts) = count_ops(function);
    assert!(violations.borrow().is_empty());
    (result, ops, casts)
}

/// The casts made for a chain: with reconciliation off they all remain, so they can be counted.
#[test]
fn casts_made_for_a_chain_are_linear() {
    for n in [100, 200, 400, 800] {
        let (result, ops, casts) = convert_chain(n, false, usize::MAX);
        let result = result.unwrap();
        println!(
            "n = {n:5}: reconcile off: {} applications, {ops} operations, {casts} casts remain",
            result.converted_ops()
        );
        assert!(casts <= 2 * n + 8, "{casts} casts for {n} adds");
    }
}

/// 5,000 additions convert with the default configuration: the rewrite limit bounds the
/// applications that follow from one operation, not the size of the function. And the depth of the
/// driver's calls does not grow with the length of a chain of uses (a driver that legalized a user
/// inside its producer overflowed this stack at about 1,000 links in the dev profile).
#[test]
fn a_chain_of_five_thousand_adds_converts_with_the_default_limit_on_a_test_thread() {
    let limit = ConversionConfig::default().max_rewrites_per_operation();
    let handle = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || convert_chain(5_000, true, limit).0.map(|r| r.converted_ops()))
        .unwrap();
    assert_eq!(handle.join().unwrap(), Ok(5_003));
}

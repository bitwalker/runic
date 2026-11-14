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
    conversion.with_reconcile_unrealized_casts(reconcile).with_max_iterations(limit);
    let violations: Seen<String> = Rc::default();
    let calls = Rc::new(core::cell::Cell::new(0));
    let result = run_conversion_with(function, &context, &config, &violations, &calls, conversion)
        .map_err(|err| format!("{err}"));
    let (ops, casts) = count_ops(function);
    assert!(violations.borrow().is_empty());
    (result, ops, casts)
}

/// With the default configuration, a function with more than 1024 operations to convert fails:
/// the limit counts every pattern application of the conversion, not the applications to one
/// operation.
#[test]
fn the_default_limit_is_a_limit_on_the_size_of_the_function() {
    // In a thread with a large stack: in the dev profile a chain of 1,000 overflows the stack of
    // a test thread before the limit is reached.
    let run = |n: usize| {
        std::thread::Builder::new()
            .stack_size(1 << 30)
            .spawn(move || convert_chain(n, true, ConversionConfig::default().max_iterations()).0)
            .unwrap()
            .join()
            .unwrap()
    };
    let result = run(1000);
    assert!(result.is_ok(), "{result:?}");
    let result = run(1100);
    let err = result.unwrap_err();
    println!("1100 adds, default limit: {err}");
    assert!(err.contains("exceeded the configured rewrite limit of 1024"), "{err}");
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

/// The depth of the driver's calls does not grow with the length of a chain of uses (a driver
/// that legalized a user inside its producer overflowed this stack at about 1,000 links in the dev
/// profile).
#[test]
fn a_chain_of_five_thousand_adds_fits_a_default_test_thread() {
    let handle = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || convert_chain(5_000, true, usize::MAX).0.map(|r| r.converted_ops()))
        .unwrap();
    assert_eq!(handle.join().unwrap(), Ok(5_003));
}

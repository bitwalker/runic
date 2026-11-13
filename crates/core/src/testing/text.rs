//! Text round trips for tests: print a built operation tree in its derived or hand-written form and
//! in the generic form, parse it back, and compare what each operation is. Used by the tests of
//! this crate and of the dialect crates.

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use crate::{
    Context, OpParser, OpRegistration, Operation, OperationRef, UnsafeIntrusiveEntityRef,
    diagnostics::{Diagnostic, Label, Report, VisitDiagnostic},
    print::AsmPrinter,
};

/// A small deterministic generator (xorshift64*), so a fuzzing test needs no dependency and
/// repeats itself from its seed.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn next_u128(&mut self) -> u128 {
        (u128::from(self.next_u64()) << 64) | u128::from(self.next_u64())
    }

    /// A value in `0..n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

/// The message of `report` and of every label attached to it.
pub fn describe(report: &Report) -> String {
    struct Labels(Vec<String>);
    impl VisitDiagnostic for Labels {
        fn label(&mut self, label: Label<'_>) {
            self.0.push(label.message.map(alloc::fmt::format).unwrap_or_default());
        }
    }
    let mut labels = Labels(Vec::new());
    let diagnostic: &dyn Diagnostic = report.as_diagnostic();
    diagnostic.visit(&mut labels);
    format!("{report} {:?}", labels.0)
}

/// `op` in the generic form.
pub fn print_generic(op: &Operation) -> String {
    let flags = Default::default();
    let mut printer = AsmPrinter::fragment(op.context_rc(), &flags);
    printer.print_operation_generic(op);
    printer.finish().to_string()
}

/// `op` in its derived or hand-written form.
pub fn print_custom(op: &Operation) -> String {
    let flags = Default::default();
    let mut printer = AsmPrinter::fragment(op.context_rc(), &flags);
    crate::print::OpPrinter::print(op, &mut printer);
    printer.finish().to_string()
}

/// Replaces every value and block number by `_`, so texts built in different orders compare.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '%' {
            let mut any = false;
            while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                chars.next();
                any = true;
            }
            if any {
                out.push('_');
            }
        }
    }
    // `^blockN`
    let mut result = String::with_capacity(out.len());
    let mut rest = out.as_str();
    while let Some(at) = rest.find("^block") {
        result.push_str(&rest[..at + 6]);
        rest = rest[at + 6..].trim_start_matches(|c: char| c.is_ascii_digit());
        result.push('_');
    }
    result.push_str(rest);
    result
}

/// What an operation is, as text: its generic form with value numbers erased, and for each
/// property whether it equals its default.
pub fn signature(op: &Operation) -> String {
    let context = op.context_rc();
    let name = op.name();
    let infos = name.properties();
    let defaults = op
        .properties()
        .zip(infos.iter())
        .map(|(attr, info)| format!("{}:{}", info.name, info.is_default(&context, &attr.value)))
        .collect::<Vec<_>>();
    format!("{} defaults={defaults:?}", normalize(&print_generic(op)))
}

/// The signature of every operation nested in `root`, in pre-order. For an operation with
/// regions only its name and properties are given, since its generic form would repeat the nested
/// operations.
pub fn signatures(root: &Operation) -> Vec<String> {
    let mut all = Vec::new();
    root.prewalk_all(|op: &Operation| {
        if !op.has_regions() {
            all.push(signature(op));
        } else {
            let context = op.context_rc();
            let name = op.name();
            let props = op
                .properties()
                .zip(name.properties().iter())
                .map(|(attr, info)| {
                    let flags = Default::default();
                    let mut printer = AsmPrinter::fragment(context.clone(), &flags);
                    printer.print_attribute_value(&*attr.value.borrow());
                    format!(
                        "{}={} default:{}",
                        info.name,
                        printer.finish(),
                        info.is_default(&context, &attr.value)
                    )
                })
                .collect::<Vec<_>>();
            all.push(format!("{} {props:?}", op.name()));
        }
    });
    all
}

/// Parses `source` as a `T` in a new context; an error is given as [describe] gives it.
pub fn parse<T: OpParser + OpRegistration>(
    source: &str,
) -> Result<(Rc<Context>, UnsafeIntrusiveEntityRef<T>), String> {
    let context = Rc::new(Context::default());
    let config = crate::parse::ParserConfig::new(context.clone());
    match crate::parse::parse::<T>(config, "text", source).into_result() {
        Ok(op) => Ok((context, op)),
        Err(report) => Err(describe(&report)),
    }
}

/// Round trips `root` (a `T`) through text and returns what went wrong, one line per failure:
///
/// * the printed text parses, prints the same, and every operation has the same signature;
/// * when `generic` is set: with any one operation without regions written in the generic form
///   instead, the text parses to operations of the same signatures.
///
/// `skipped` receives the operations whose generic form could not be substituted (their printed
/// form does not occur exactly once in the text).
pub fn round_trip<T: OpParser + OpRegistration>(
    root: OperationRef,
    generic: bool,
    skipped: &mut Vec<String>,
) -> Vec<String> {
    let mut failures = Vec::new();
    let root = root.borrow();
    let printed = format!("{}", *root);
    let original = signatures(&root);

    match parse::<T>(&printed) {
        Ok((_context, op)) => {
            let op_ref = op.as_operation_ref();
            let op = op_ref.borrow();
            let reprinted = format!("{}", *op);
            if reprinted != printed {
                failures.push(format!(
                    "custom form is not a fixpoint:\n--- printed\n{printed}\n--- \
                     reprinted\n{reprinted}"
                ));
            }
            let sigs = signatures(&op);
            if sigs != original {
                failures.push(format!(
                    "custom form changes an operation:\n--- built\n{original:#?}\n--- \
                     parsed\n{sigs:#?}"
                ));
            }
        }
        Err(err) => {
            failures.push(format!("custom form does not parse: {err}\n{printed}"));
            return failures;
        }
    }

    if !generic {
        return failures;
    }

    let mut candidates = Vec::new();
    root.prewalk_all(|op: &Operation| {
        if !op.has_regions() {
            candidates.push((op.name().to_string(), print_custom(op), print_generic(op)));
        }
    });
    for (name, custom, generic) in candidates {
        if printed.matches(custom.as_str()).count() != 1 {
            skipped.push(format!("{name}: `{custom}`"));
            continue;
        }
        let text = printed.replacen(custom.as_str(), generic.as_str(), 1);
        match parse::<T>(&text) {
            Ok((_context, op)) => {
                let op_ref = op.as_operation_ref();
                let sigs = signatures(&op_ref.borrow());
                if sigs != original {
                    failures.push(format!(
                        "generic form of '{name}' changes an operation: `{generic}`\n--- \
                         built\n{original:#?}\n--- parsed\n{sigs:#?}"
                    ));
                }
            }
            Err(err) => {
                failures
                    .push(format!("generic form of '{name}' does not parse: {err}\n  `{generic}`"));
            }
        }
    }
    failures
}

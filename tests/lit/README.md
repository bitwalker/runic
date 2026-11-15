# The lit suite

Tests of `runic-opt` written as text: each `*.runic` file holds IR and `// RUN:` lines that run the
tool on it, and checks of what it prints. The suite is run by `litcheck` (a lit runner and FileCheck
in one binary), which provides `filecheck` and `not` itself. The suite is written for `litcheck`
0.5.0 or newer.

## Running it

```sh
cargo make test-lit
```

builds `runic-opt` into `bin/` (git-ignored) and runs every test. To run tests by hand, build once
and point `litcheck` at `bin`:

```sh
cargo make runic-opt
litcheck lit run --path bin tests/lit
litcheck lit run --path bin tests/lit/transforms/cse/graph-no-replaceable-uses.runic
litcheck lit run --path bin tests/lit/transforms
litcheck lit run --path bin --filter 'runic::transforms/' tests/lit
```

After the build, the lines run the suite, one file, one area (a directory), and the same area
by name. `--filter` is a regular expression over `suite::path`, where the suite is `runic` and the
path is relative to this directory: `'^transforms/'` selects nothing. `-v`
(`--verbose`) prints the commands and output of every test that fails. `cargo make lit` builds the
tool and passes its arguments to `litcheck lit`:

```sh
cargo make lit run --path bin tests/lit/transforms
```

The tool is found through `--path bin`: the suite has no substitution and needs no environment
variable.

A clean machine needs the pinned toolchain (`rust-toolchain.toml`), the `cargo-make` crate
installed, and network access once, for the `litcheck` task's `cargo install`.

### Which operations have a parse test

```sh
cargo make lit-coverage
tests/lit/coverage.sh bin/runic-opt
```

Both list, one per line, each operation runic-opt registers (`runic-opt --list-operations`)
whose name appears in no test under `parse/` outside a `//` comment, and fail if there is one;
the first builds the tool first. They print nothing when every operation has a parse test. Each
dialect's operations are in `parse/<dialect>.runic`, which the Rust tests of the generic form
read too (`crates/transform/src/tests/dialect_modules.rs`).

## Layout

Directories are added with their first test.

```
lit.suite.toml        the suite: name `runic`, files `*.runic`, `pipefail = true`
parse/                reading and printing IR
verify/               the verifier's rules, one diagnostic per case
roundtrip/            both forms read back
dialects/<dialect>/   the operations of a dialect
transforms/<pass>/    a pass: cse, dce, sccp, canonicalizer, sink, local2reg, cfg-to-scf
conversion/           dialect conversion, through the test passes
analysis/             analyses, through the test passes
tool/                 options and behaviour of runic-opt itself
```

A test is named in lower case with hyphens, `*.runic`, in the directory of its area:
`transforms/local2reg/chained-locals.runic`.

## Writing a test

Run the tool on the input first and read what it prints; then write checks for what the test is
about. Never paste the whole output as checks. Break the test once on purpose (an expected value, or
the option on the RUN line) and see it fail.

A test is RUN lines, IR, and checks in comments:

```
// RUN: runic-opt %s -p 'builtin.module(cse)' | filecheck %s
// RUN: runic-opt %s --split-input-file --verify-diagnostics | filecheck %s
```

`%s` is the test file. `--split-input-file` runs each case between lines `// -----` on its own;
diagnostics keep the file's line numbers, and a case that fails prints nothing.

### A failure is checked, never ignored

`lit.suite.toml` sets `[format.shtest] pipefail = true`. Without it a RUN line
`runic-opt .. | filecheck %s` passes when `runic-opt` fails, as long as `filecheck` matches what was
printed (by other cases, say). With it, every command of a pipeline must succeed. A RUN line whose
tool fails on purpose says so, with `--verify-diagnostics` when the failure is a diagnostic the
input can expect, and with `not` otherwise: to check how a diagnostic is shown on standard error,
or for a pipeline the tool rejects before it reads the input (so no expectation is read either):

```
// RUN: runic-opt %s -p 'builtin.module(builtin.function(test-fail{message="no good"}))' --verify-diagnostics
// RUN: not runic-opt %s -p test-fail 2>&1 | filecheck %s --check-prefix=PLAIN
// RUN: not runic-opt %s -p 'builtin.module(builtin.function(dce{x=1}))' 2>&1 | filecheck %s --check-prefix=NOOPT
```

### One function per case, under a label that names it

Each case's checks start with a `CHECK-LABEL` that names its function (or its module, when the case
has no function). Values and blocks print by number, in the order they were created across the
whole context; the names written in the input are not kept. A check therefore captures a value or
block name, `[[V:%[0-9]+]]` or `^[[B:block[0-9]+]]`, and uses it as `[[V]]`; it never spells a
number. A value not used again is matched with `{{%[0-9]+}}`.

A `CHECK-LABEL` holds no variable: the label is the function's name, `CHECK-LABEL: @name(`, and
the rest of that line, with its captures, follows in a `CHECK-SAME`, which must match on the
label's line, so the signature checked is the labelled function's. The body follows with
`CHECK-NEXT`.

```
// CHECK-LABEL: @select_false_true(
// CHECK-SAME: [[C:%[0-9]+]]: i1) -> i1 {
// CHECK-NEXT: [[N:%[0-9]+]] = arith.bnot [[C]];
// CHECK-NEXT: builtin.ret [[N]] : (i1);
```

### A test checks what it is about

A test of a pass checks what the pass did, not how the printer lays the IR out (indentation, the
closing lines of a function or module). A test of the printer, under `parse/` or `roundtrip/`,
checks the layout: a parse test checks every line of the reprint, and writes its input in another
spelling wherever the parser takes more than the printer writes, so that it is more than an
identity. A test that expects layout folds states the module's `data_layout`:

```
builtin.module public @m32 attributes { data_layout = #builtin.data_layout<little, ptr = 32> } {
```

To check that something is gone, check where it would be: a `CHECK-NOT` before the first label
covers the output before it, one after the last check covers only what follows. A separate prefix
checks the whole output:

```
// RUN: runic-opt %s -p test-convert-wide-ints | filecheck %s --check-prefix=GONE
// GONE: builtin.module public @m {
// GONE-NOT: {{[(: ]i128[),; ]}}
// GONE: {{^}}};
```

### Reading output back

`--verify-roundtrip` checks the parsed input: that both forms of it read back to the same program.
A test of a pass that wants its output to read back pipes it into a second run:

```
// RUN: runic-opt %s -p 'builtin.module(cse)' | runic-opt - --verify-roundtrip | filecheck %s
```

A test that prints locations (`--print-locations`) reads its input from `%s`, never from a pipe: a
location that names standard input cannot be resolved by a second run. The path in a location is
the path as given, matched with `{{.*}}`, or with `{{[^"]*}}` on a line that holds several
locations (`litcheck` does not take `{{.*}}` back to an earlier match, so it runs to the last):

```
// RUN: runic-opt %s --print-locations | filecheck %s
// CHECK-NEXT: builtin.ret {{%[0-9]+}} : (i32) loc("{{.*}}locations.runic":[[@LINE+3]]:5 to [[@LINE+3]]:16);
```

An operation that reads a value printed later (in a graph region, or a block printed later) prints
in the generic form unless its custom form states its operand types; the input writes it the same
way:

```
"mem.store"(%dup, %ptr) <{ align = #mem.align<natural> }> : (i32, ptr<i32>) -> ();
// CHECK-NEXT: "mem.store"([[DUP:%[0-9]+]], [[PTR]]) <{ align = #mem.align<natural> }> : (i32, ptr<i32>) -> ();
```

### What the IR may write

- Every block ends with a terminator, unless the test is about that rule.
- Values and blocks are named for what they are (`%sum`, `%cond`, `^after`), never by number
  (`%3`, `^block1`): numbers in an input read like printer output, and are not kept anyway.
- A symbol reference as an attribute value is written `#builtin.symbol<@name>`, `@name`, or, rooted,
  `::@m::@name`; each prints as `#builtin.symbol<..>`, which is what a check matches:

  ```
  } attributes { bare = @callee, rooted = ::@test::@callee, spelled = #builtin.symbol<@callee> };
  // CHECK-NEXT: } attributes { bare = #builtin.symbol<@callee>, rooted = #builtin.symbol<::@test::@callee>, spelled = #builtin.symbol<@callee> };
  ```
- A location is not usable as an attribute value.

### Diagnostics

Diagnostics are tested with `--verify-diagnostics` and comments that expect them. Every diagnostic
must be expected and every expectation met; the tool's exit code is that check's. The kinds are
`error`, `warning`, `info` and `hint` (there is no `note` or `remark`); `-re` makes the text a
regular expression. An expectation is on the line of the diagnostic, or names it with `@+N`, `@-N`,
`@above`, `@below`, or `@unknown` (a diagnostic with no location):

```
%x = arith.constant 10 : i32; // expected-hint {{range of result 0: u[10, 10] s[10, 10] width 32}}
// expected-hint@+1 {{range of result 0: u[30, 30] s[30, 30] width 32}}
// expected-hint@below {{range of result 0: u[0, 4294967295] s[-2147483648, 2147483647] width 32}}
%lt = arith.cmpi ult %a, %x; // expected-hint-re {{u\[0, 1\] s\[-1, 0\] width 1$}}
// expected-error@unknown {{failed to legalize operation 'arith.add'}}
```

The verifier stops at its first error, so each rule is a case of its own, separated by `// -----`
under `--split-input-file`. A test with only diagnostics has no `filecheck` on its RUN line
(`filecheck` refuses a file with no check):

```
// RUN: runic-opt %s -p 'builtin.module(builtin.function(test-fail{message="no good"}))' --verify-diagnostics
```

All of standard error is written before standard output. A check of standard error (`2>&1`) matches
the path in a `-->` line with `{{.*}}`, since it is the path as given:

```
// PLAIN: error: test-fail
// PLAIN-NEXT: --> {{.*}}test-fail.runic:[[@LINE+6]]:1
```

### A known defect

A test that shows a defect is not bent to pass. It is committed with `// XFAIL: *` and a comment
that says what is wrong, until the defect is fixed; a test marked so that passes fails the run (no
test of the suite is marked so today).

```
// XFAIL: *
```

### Independent of where and how it runs

No test depends on the working directory, the terminal, the order of a hash map, or timing. Nothing
in the suite names the project the old tests were ported from, its dialects or its tools.

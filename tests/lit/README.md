# The lit suite

Tests of `runic-opt` and of the toy compiler `toyc` written as text: each `*.runic` file holds IR,
and each `*.calc` file a program of the toy language `calc`, with `// RUN:` lines that run a tool
on it, and checks of what it prints. The suite is run by `litcheck` (a lit runner and FileCheck
in one binary), which provides `filecheck` and `not` itself. The suite is written for `litcheck`
0.5.0 or newer.

## Running it

```sh
cargo make test-lit
```

builds `runic-opt` and `toyc` into `bin/` (git-ignored) and runs every test. `runic-opt` is built
with the cargo feature `toy`, which registers the toy frontend `calc` and the toy backends `stackvm`
and `altvm`; `toyc` is the example `toyc` of `crates/driver`, a compiler built on the driver crate
with the same stages (`cargo make toyc` builds it alone). To run tests by hand, build once and point
`litcheck` at `bin`. The suite is run from the repository root, as below: run from a
directory inside the suite, `litcheck` does not find it, and it refuses a path outside the working
directory:

```sh
cargo make runic-opt
cargo make toyc
litcheck lit run --path bin tests/lit
litcheck lit run --path bin tests/lit/transforms/cse/graph-no-replaceable-uses.runic
litcheck lit run --path bin tests/lit/transforms
litcheck lit run --path bin --filter 'runic::transforms/' tests/lit
```

After the build, the lines run the suite, one file, one area (a directory), and the same area
by name. `--filter` is a regular expression over `suite::path`, where the suite is `runic` and the
path is relative to this directory: `'^transforms/'` selects nothing. `-v`
(`--verbose`, written before `--path`) prints the commands and output of every test that fails. `cargo make lit` builds the
tools and passes its arguments to `litcheck lit`:

```sh
cargo make lit run --path bin tests/lit/transforms
```

The tool is found through `--path bin`: the suite has no substitution and needs no environment
variable.

A clean machine needs the pinned toolchain (`rust-toolchain.toml`), the `cargo-make` crate
installed, and network access once, for the `litcheck` task's `cargo install`: the task installs
`litcheck` only when no `litcheck` is installed. The `toyc` tests that tell exit code 1 from 2
run the command in `sh -c` with `test`, so they need a POSIX `sh`.

A checkout whose path holds a space is not supported: RUN lines write `%s` bare, as LLVM's suites
do, and `litcheck` substitutes the path unquoted.

### Which operations have a parse test, and which passes a test

```sh
cargo make lit-coverage
tests/lit/coverage.sh bin/runic-opt
```

Both list, one per line, each operation runic-opt registers (`runic-opt --list-operations`)
whose name appears in no test under `parse/` outside a `//` comment (`operation arith.add`), and
each pass it registers (`runic-opt --list-passes`) that no `// RUN:` line under `tests/lit` names
(`pass cse`), and fail if there is one; the first builds the tool first. They print nothing when
every operation has a parse test and every pass a test. They check that each operation's name
appears, not that every form of it does (each optional part, an attribute dictionary): that is
what the parse tests themselves are written to cover. Each
dialect's operations are in `parse/<dialect>.runic`, which the Rust tests of the generic form
read too (`crates/transform/src/tests/dialect_modules.rs`).

## Layout

Directories are added with their first test.

```
lit.suite.toml        the suite: name `runic`, files `*.runic` and `*.calc`, `pipefail = true`
parse/                reading and printing IR
verify/               the verifier's rules, one diagnostic per case
roundtrip/            both forms read back
dialects/<dialect>/   a dialect's folds (fold.runic) and canonicalization patterns (canonicalize.runic),
                      and the operations added after its parse test was written (tail-call.runic,
                      float.runic, float-conv.runic), with their own folds beside them
                      The `wasm` dialect has no folds: `dialects/wasm/` holds its effects (effects.runic),
                      the passes on its control flow (passes.runic) and whole programs written
                      in it (programs.runic, separated by `// -----` and read with
                      `--split-input-file`); its parse tests are `parse/wasm-*.runic` and its
                      verifier's rules `verify/wasm-*.runic`
transforms/<pass>/    a pass: cse, dce, sccp, canonicalizer, control-flow-sink, sink
                      (sink-operand-defs), local2reg, cfg-to-scf
transforms/pipeline/   several passes in a row, where what matters is what survives all of them
conversion/           dialect conversion, through the test passes
analysis/             analyses, through the test passes
tool/                 options and behaviour of runic-opt itself
harness/              runic-opt's harness path: an input a frontend reads (`*.calc`) or a
                      `--target` that sends it through a backend; the stage flags, `--emit`,
                      `--stop-after`, `--list-stages`
toyc/                 the command line of a compiler built on the driver crate, through `toyc`
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

`%s` is the test file, and `%t` a file of the test's own that a RUN line may write
(`runic-opt %s -o %t`); the path is fresh for each run of the test, and a suffix makes another
(`%t.svm`, or `%t.dir` for a directory the tool creates). `--split-input-file` runs each case between lines `// -----` on its own;
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

### The toy compiler and the harness path

`harness/` runs `runic-opt` on `*.calc` files and with `--target`; `toyc/` runs `toyc`. A `calc`
file reads `//` as a comment, so its RUN lines and checks are written in comments like those of a
`.runic` file, and the program follows them. The module of a program is named after the file
(`harness/emit.calc` is `@emit`; a name with a hyphen prints quoted), so a check matches it with
`{{.*}}`, as in `{{.*}}::k:`. A test that needs no input (`--print`, `--help`, `-Z help`) is a
`*.runic` file with only RUN lines and checks.

`toyc` writes its output in the current directory when it is given neither `-o` nor
`--output-dir`: every test points the output at `-o -` (standard output), at `-o %t..`, or at
`--output-dir %t.dir`, or writes nothing (`--print`, `--help`, a request that fails).

`runic-opt` exits 1 for every failure, and `not` is enough. `toyc` exits 1 when a compilation
failed and 2 for a usage error (an unknown flag, a request that does not resolve), and a test
tells the two apart: `litcheck`'s `not` checks only that the command did not succeed, and its
shell has no `;`. The form the suite uses runs the command in `sh -c`, with standard error
joined to standard output, and ends the string with the exact code, so that the pipeline's status
is that of the check and the output goes on to `filecheck`:

```
// RUN: sh -c "toyc %s --no-such-flag -o - 2>&1; test \$? -eq 2" | filecheck %s
```

Do not write `toyc .. || test $? -eq 2`: it passes when `toyc` succeeds.

### One function per case, under a label that names it

Each case's checks start with a `CHECK-LABEL` that names its function (or its module, when the case
has no function). Values and blocks print by number, in the order they were created across the
whole context; the names written in the input are not kept. A check therefore captures a value or
block name, `[[V:%[0-9]+]]` or `^[[B:block[0-9]+]]`, and uses it as `[[V]]`; it never spells a
number. A value not used again is matched with `{{%[0-9]+}}`. A value at the end of a check is followed by what follows it in the output (`;`, ` :`,
` <{`, or `{{[ ]*$}}` where the line ends), because a captured `%1` otherwise matches the start of `%13`.

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

A test that shows a pass leaving something alone also holds a case the pass changes, so that the
test fails when the pass does not run at all. A test of one canonicalization pattern restricts the
canonicalizer to it, one RUN line and one check prefix per pattern; the greedy driver still folds
and erases dead operations:

```
// RUN: runic-opt %s -p 'builtin.module(builtin.function(canonicalizer{enable-patterns=erase-true-assert}))' | filecheck %s --check-prefix=ASSERT
```

### Reading output back

`--verify-roundtrip` checks the parsed input: that both forms of it read back to the same program.
A test of a pass that wants its output to read back pipes it into a second run:

```
// RUN: runic-opt %s -p 'builtin.module(cse)' | runic-opt - --verify-roundtrip | filecheck %s
```

A test that prints locations (`--print-locations`) reads its input from `%s`, never from a pipe: a
location that names standard input cannot be resolved by a second run. The one exception is piped
text that carries written locations naming the file, such as the generic form printed with
`--print-locations` from `%s`: a second run reads those locations back and prints them again
(`roundtrip/locations.runic`). The path in a location is
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
regular expression. The text is looked for in the diagnostic's message, the text of its labels,
its helps and its context frames; a parser error's message is often `invalid syntax` alone, and
its label names the token it did not expect and what it expected, so its expectation names both:
`{{unexpected token '<', expected )}}`. An expectation is on the line of the diagnostic, or names
it with `@+N`, `@-N`, `@above`, `@below`, or `@unknown` (a diagnostic with no location):

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

A test that shows a defect is not bent to pass. The defect is fixed if the fix is small. Otherwise
the test is committed with `// XFAIL: *` and a comment that says what is wrong, until the defect is
fixed; a test marked so that passes fails the run (no test of the suite is marked so today). Or,
where the defect is in text the test checks anyway (a parse test checks every line it prints), the
check that holds it is pinned under a `Known defect (ledger)` comment that says what is wrong and
that the check changes when it is fixed.

```
// XFAIL: *
```

### Independent of where and how it runs

No test depends on the working directory, the terminal, the order of a hash map, or timing. Nothing
in the suite names the project the old tests were ported from, its dialects or its tools.

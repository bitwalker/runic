# Timing a command

`tools/time-command.py` runs one command (without a shell), measures its wall time, and appends a
record of the run to a JSONL file. No `cargo make` task runs it.

```sh
python3 tools/time-command.py --label warm-unchanged-1 \
  --output target/timings.jsonl -- cargo nextest run --profile ci -p runic-ir
```

`--label` names the run (for example the state of the caches, or the edit measured) and
`--output` is the file the record is appended to; the command follows `--`. The record, also
printed to standard output, holds the label, the time in UTC, the command, the working directory,
the platform, the git revision (`git rev-parse HEAD`) and whether the working tree is dirty
(`git status --porcelain`), the `rustc --version`, the values of those of `CARGO_TARGET_DIR`,
`CARGO_BUILD_BUILD_DIR`, `CARGO_BUILD_JOBS`, `CARGO_MANIFEST_DIR`, `RUSTUP_TOOLCHAIN`,
`RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, `NEXTEST_PROFILE`, `NEXTEST_TEST_THREADS` and
`RUNIC_TEST_TIMINGS` that are set, the elapsed seconds and the exit code. The script exits with the
command's exit code. It is run from the repository root (it calls `git` and `rustc` there).
`RUNIC_TEST_TIMINGS` is only recorded: no workspace member reads it.

## How to measure

Record timings before changing build profiles or caching policy. A faster isolated slice is evidence
for that slice; it does not establish a whole-suite speedup. Never compare runs performed alongside
other builds. Record the exact revision, dirty state, command, target directory, toolchain, and test
concurrency: the script records these fields.

Run each warm command at least three times after an unmeasured warm-up and compare the median. For
an actual cold Cargo build, use a new, empty `CARGO_TARGET_DIR`; record whether the registry and
compiler caches were already populated. Do not delete the shared target directory or call a build
cold merely because it is the first measurement. Run the exact same command again for its warm
comparison. For an edit/rebuild comparison, record the exact source diff as well as the command; a
timestamp-only touch measures Cargo invalidation without representing a meaningful source edit.

## Per-test durations

CI selects nextest's `ci` profile (`.config/nextest.toml`), which writes
`target/nextest/ci/junit.xml`; CI uploads it, even when tests fail. JUnit includes per-test
durations and failed-test output. A build failure before nextest starts may produce no report. The
lit suite is a separate runner and is not represented by this XML. Local runs can select
`--profile ci` too. See the
[nextest JUnit documentation](https://www.nexte.st/docs/machine-readable/junit/).

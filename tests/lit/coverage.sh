#!/bin/sh
# Lists every operation registered in runic-opt whose name appears in no test under parse/, and
# every pass registered in runic-opt that no RUN line of a test under tests/lit names, one per
# line, and exits 1 if there is one, 0 otherwise.
#
#   tests/lit/coverage.sh bin/runic-opt
#
# An operation counts as tested when its name, bare (`arith.add`) or quoted (`"arith.add"`, the
# generic form), appears outside a `//` comment in a `*.runic` file under tests/lit/parse. A pass
# counts as tested when its name (`cse`, `sink-operand-defs`) appears in a `// RUN:` line of a
# `*.runic` file under tests/lit, not as part of a longer name.

set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 RUNIC-OPT" >&2
    exit 2
fi
tool=$1
lit_dir=$(dirname -- "$0")
parse_dir=$lit_dir/parse

operations=$("$tool" --list-operations)
# The text of the parse tests, with comments removed.
text=$(find "$parse_dir" -name '*.runic' -type f -exec sed -e 's://.*$::' {} +)

missing=0
for op in $operations; do
    # `-w`: `arith.add` is not found in `arith.add_overflowing`.
    if ! printf '%s\n' "$text" | grep -F -w -q -- "$op"; then
        printf 'operation %s\n' "$op"
        missing=1
    fi
done

# The first word of each line of `--list-passes` is a pass name.
passes=$("$tool" --list-passes | sed -e 's/[[:space:]].*$//')
# The RUN lines of every test.
runs=$(find "$lit_dir" -name '*.runic' -type f -exec grep -h -e '// RUN:' {} +)

for pass in $passes; do
    # A pass name is letters, digits and `-`: `sink` is not found in `sink-operand-defs`.
    if ! printf '%s\n' "$runs" | grep -E -q -- "(^|[^a-z0-9-])$pass([^a-z0-9-]|\$)"; then
        printf 'pass %s\n' "$pass"
        missing=1
    fi
done
exit "$missing"

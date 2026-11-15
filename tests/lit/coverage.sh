#!/bin/sh
# Lists every operation registered in runic-opt whose name appears in no test under parse/, one
# per line, and exits 1 if there is one, 0 otherwise.
#
#   tests/lit/coverage.sh bin/runic-opt
#
# An operation counts as tested when its name, bare (`arith.add`) or quoted (`"arith.add"`, the
# generic form), appears outside a `//` comment in a `*.runic` file under tests/lit/parse.

set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 RUNIC-OPT" >&2
    exit 2
fi
tool=$1
parse_dir=$(dirname -- "$0")/parse

operations=$("$tool" --list-operations)
# The text of the parse tests, with comments removed.
text=$(find "$parse_dir" -name '*.runic' -type f -exec sed -e 's://.*$::' {} +)

missing=0
for op in $operations; do
    # `-w`: `arith.add` is not found in `arith.add_overflowing`.
    if ! printf '%s\n' "$text" | grep -F -w -q -- "$op"; then
        printf '%s\n' "$op"
        missing=1
    fi
done
exit "$missing"

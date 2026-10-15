#!/usr/bin/env bash
# scripts/run_tests.sh <required-test-binary>[,<more>...] -- <cargo test command...>
#
# Runs the given cargo test command and fails unless
#   - the command itself succeeds,
#   - at least one test passed in total, and
#   - every named integration test binary ran and passed at least one test.
# A feature or target mismatch otherwise turns a test step into a silent
# no-op that still reports success.
#
# Several names are given comma separated. Naming a binary is what keeps a
# suite from disappearing: `cargo test` without `--test` runs every target, so
# a suite that stops being built reports nothing rather than failing.
set -euo pipefail

IFS=',' read -r -a required <<< "$1"
shift
[[ "${1:-}" == "--" ]] && shift

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

"$@" 2>&1 | tee "$log"

total=0
# ⚠️ Indexed arrays, not an associative one: `declare -A` needs bash 4, and the
# macOS runner's /bin/bash is 3.2 — the CI step failed with
# `declare: -A: invalid option` while a Homebrew bash 5 ran it locally.
required_passed=()
for i in "${!required[@]}"; do required_passed[$i]=0; done
current=""
while IFS= read -r line; do
  case "$line" in
    *"Running "*)
      current="$line"
      ;;
    "test result: "*)
      n=$(printf '%s\n' "$line" | sed -E 's/.* ([0-9]+) passed.*/\1/')
      total=$((total + n))
      for i in "${!required[@]}"; do
        name="${required[$i]}"
        if [[ "$current" == *"tests/${name}.rs"* || "$current" == *"tests\\${name}.rs"* ]]; then
          required_passed[$i]=$((required_passed[$i] + n))
        fi
      done
      ;;
  esac
# strip CR (Windows) and ANSI colour codes (CARGO_TERM_COLOR=always puts one
# right after "Running", which would hide the binary name from the match above)
done < <(tr -d '\r' < "$log" | sed "s/$(printf '\033')\[[0-9;]*m//g")

summary=""
for i in "${!required[@]}"; do
  summary="${summary} ${required_passed[$i]} in ${required[$i]},"
done
echo "run_tests: ${total} tests passed in total,${summary%,}"
if [[ "$total" -eq 0 ]]; then
  echo "run_tests: no test ran" >&2
  exit 1
fi
missing=0
for i in "${!required[@]}"; do
  if [[ "${required_passed[$i]}" -eq 0 ]]; then
    echo "run_tests: ${required[$i]} did not run any test" >&2
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || exit 1

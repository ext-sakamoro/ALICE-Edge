#!/usr/bin/env bash
# Production code must not ship stubs or debug output: todo! / unimplemented! /
# panic!("...STUB...") / dbg! anywhere under src/. Used by security-audit.yml
# and scripts/preflight.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

files=$(git ls-files 'src/*.rs' | wc -l | tr -d ' ')
if [ "$files" -eq 0 ]; then
  echo "stub guard compared 0 files under src/" >&2
  exit 1
fi

hits=$(grep -rnE 'todo!\(|unimplemented!\(|panic!\([^)]*STUB|dbg!\(' src/ --include='*.rs' || true)
if [ -n "$hits" ]; then
  echo "stub / debug macro in src/ (production path):" >&2
  echo "$hits" >&2
  exit 1
fi
echo "stub guard: $files files under src/, no todo! / unimplemented! / panic!(STUB) / dbg!"

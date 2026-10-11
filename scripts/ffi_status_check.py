#!/usr/bin/env python3
"""Check that the C ABI status codes agree in Rust, C and C#.

The status codes returned by the robust / filter functions are defined in
three places:

  * src/ffi.rs            `pub const ALICE_EDGE_<NAME>: i32 = <value>;`
  * bindings/alice_edge.h `#define ALICE_EDGE_<NAME> <value>` (or `(<value>)`)
  * bindings/AliceEdge.cs `public const int <CamelName> = <value>;` inside
                          `public static class AliceStatus`

C# names are mapped to the C names: `Ok` -> `OK`, any other `CamelName` ->
`ERR_` + upper snake case (`NonPositiveK` -> `ERR_NON_POSITIVE_K`).

Each file must define exactly EXPECTED_COUNT codes (no duplicates), and the
three name -> value maps must be identical. A file that yields a different
count, including 0, fails, so a renamed file, constant or class cannot turn
the check into a no-op.

Usage: `python3 scripts/ffi_status_check.py` (exit 1 on any finding).
`--root DIR` runs against another tree (used by
scripts/test_ffi_status_check.py).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

EXPECTED_COUNT = 5

RUST_FILE = os.path.join("src", "ffi.rs")
C_FILE = os.path.join("bindings", "alice_edge.h")
CS_FILE = os.path.join("bindings", "AliceEdge.cs")

RUST_RE = re.compile(r"^\s*pub\s+const\s+ALICE_EDGE_(OK|ERR_[A-Z0-9_]+)\s*:\s*i32\s*=\s*(-?\d+)\s*;", re.M)
C_RE = re.compile(r"^\s*#\s*define\s+ALICE_EDGE_(OK|ERR_[A-Z0-9_]+)\s+\(?\s*(-?\d+)\s*\)?", re.M)
CS_CLASS_RE = re.compile(r"\bclass\s+AliceStatus\s*\{(.*?)\}", re.S)
CS_RE = re.compile(r"\bpublic\s+const\s+int\s+([A-Za-z][A-Za-z0-9]*)\s*=\s*(-?\d+)\s*;")


def read(root: str, rel: str) -> str:
    # explicit encoding: the platform default is not UTF-8 on every OS
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return f.read()


def cs_to_c(name: str) -> str:
    if name == "Ok":
        return "OK"
    return "ERR_" + re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def collect(pairs, rel: str, errors: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, value in pairs:
        if name in out:
            errors.append(f"{rel}: {name} is defined twice")
        out[name] = int(value)
    if len(out) != EXPECTED_COUNT:
        errors.append(f"{rel}: found {len(out)} status codes, expected {EXPECTED_COUNT}")
    return out


def check(root: str) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    rust = collect(RUST_RE.findall(read(root, RUST_FILE)), RUST_FILE, errors)
    c = collect(C_RE.findall(read(root, C_FILE)), C_FILE, errors)
    cs_text = read(root, CS_FILE)
    m = CS_CLASS_RE.search(cs_text)
    if m is None:
        errors.append(f"{CS_FILE}: class AliceStatus not found")
        cs_pairs = []
    else:
        cs_pairs = [(cs_to_c(n), v) for n, v in CS_RE.findall(m.group(1))]
    cs = collect(cs_pairs, CS_FILE, errors)

    maps = {RUST_FILE: rust, C_FILE: c, CS_FILE: cs}
    names = sorted(set(rust) | set(c) | set(cs))
    compared = 0
    for name in names:
        for rel, mp in maps.items():
            if name not in mp:
                errors.append(f"{rel}: ALICE_EDGE_{name} is missing")
        values = {rel: mp[name] for rel, mp in maps.items() if name in mp}
        if len(values) == len(maps):
            compared += 1
        if len(set(values.values())) > 1:
            detail = ", ".join(f"{rel}={v}" for rel, v in values.items())
            errors.append(f"ALICE_EDGE_{name} differs: {detail}")
    counts = {"rust": len(rust), "c": len(c), "cs": len(cs), "compared": compared}
    if compared != EXPECTED_COUNT:
        errors.append(f"compared {compared} codes present in all three files, expected {EXPECTED_COUNT}")
    return errors, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv)
    errors, counts = check(args.root)
    print(
        "ffi status codes: rust={rust} c={c} cs={cs} compared={compared}".format(**counts)
    )
    for e in errors:
        print(f"error: {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

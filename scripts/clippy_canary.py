#!/usr/bin/env python3
"""Canary for the clippy determinism gates (G1 disallowed transcendentals and
G10 wall clock / OS entropy): the lists in clippy.toml only protect anything if
clippy actually flags a call to each entry.

A throwaway crate is built in a temporary directory with an exact copy of the
repository's clippy.toml, rust-toolchain.toml and `[lints.*]` tables. It holds
one library function per row of scripts/clippy_canary_calls.tsv, each planting
one banned call, plus one clean control function. `cargo clippy --lib` runs
with the CI clippy job's flags (`-W clippy::pedantic -D warnings`) and the
script fails unless

  (i)   clippy exits non-zero,
  (ii)  every row's function gets exactly one `disallowed_*` diagnostic naming
        that row's path, and no other `disallowed_*` diagnostic appears,
  (iii) every row's path is listed under its key in clippy.toml, and
  (iv)  something was compared: the data file has at least one row and
        clippy.toml has at least one entry.

The control function must get no `disallowed_*` diagnostic.

The banned-call list lives in the one data file so that it can later be
generated from the ALICE-Law gate list; this script carries no
repository-specific logic beyond the file names below.

usage: scripts/clippy_canary.py [--calls FILE] [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALLS = ROOT / "scripts" / "clippy_canary_calls.tsv"
KEYS = ("disallowed-methods", "disallowed-types")
LINTS = {"clippy::disallowed_methods", "clippy::disallowed_types"}
# the same flags as the clippy job in .github/workflows/ci.yml
CLIPPY_FLAGS = ["-W", "clippy::pedantic", "-D", "warnings"]
CONTROL = "canary_control"


@dataclass(frozen=True)
class Row:
    line: int
    key: str
    path: str
    stmt: str


def parse_rows(text: str) -> tuple[list[Row], list[str]]:
    """Rows of the data file; `#` lines and blank lines are skipped."""
    rows: list[Row] = []
    problems: list[str] = []
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 3 or not all(c.strip() for c in cols):
            problems.append(f"data line {n}: expected 3 tab-separated columns, got {raw!r}")
            continue
        key, path, stmt = (c.strip() for c in cols)
        if key not in KEYS:
            problems.append(f"data line {n}: unknown key {key!r} (expected one of {', '.join(KEYS)})")
            continue
        rows.append(Row(n, key, path, stmt))
    return rows, problems


def clippy_toml_paths(text: str) -> dict[str, set[str]]:
    """`path = "..."` entries under each disallowed-* array of clippy.toml."""
    found: dict[str, set[str]] = {k: set() for k in KEYS}
    current = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if current is None:
            m = re.match(r"^(disallowed-[a-z]+)\s*=\s*\[", line)
            if m and m.group(1) in found:
                current = m.group(1)
                line = line[m.end():]
            else:
                continue
        for p in re.findall(r'path\s*=\s*"([^"]+)"', line):
            found[current].add(p)
        if line.endswith("]"):
            current = None
    return found


def lints_tables(cargo_toml: str) -> str:
    """The `[lints]` / `[lints.*]` tables of Cargo.toml, verbatim."""
    out: list[str] = []
    keep = False
    for raw in cargo_toml.splitlines():
        s = raw.strip()
        if s.startswith("["):
            keep = s == "[lints]" or s.startswith("[lints.")
        if keep:
            out.append(raw)
    return "\n".join(out) + ("\n" if out else "")


def render_lib(rows: list[Row]) -> tuple[str, dict[str, tuple[int, int]]]:
    """lib.rs text and the 1-based line range of each function by name."""
    lines = ["//! clippy canary (generated, throwaway)"]
    ranges: dict[str, tuple[int, int]] = {}

    def add(name: str, body: str) -> None:
        start = len(lines) + 1
        lines.extend([f"pub fn {name}() {{", f"    {body}", "}"])
        ranges[name] = (start, len(lines))

    for i, row in enumerate(rows):
        add(f"canary_{i}", row.stmt)
    add(CONTROL, "let _ = core::hint::black_box(0.5_f64).sqrt();")
    return "\n".join(lines) + "\n", ranges


def disallowed_diagnostics(stdout: str) -> list[tuple[str, int, str]]:
    """(lint, line, message) of every disallowed_* diagnostic in cargo's JSON."""
    out = []
    for raw in stdout.splitlines():
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        d = msg.get("message") or {}
        code = (d.get("code") or {}).get("code")
        if code not in LINTS:
            continue
        spans = [s for s in d.get("spans", []) if s.get("is_primary")] or d.get("spans", [])
        line = spans[0]["line_start"] if spans else 0
        out.append((code, line, d.get("message", "")))
    return out


def check(rows: list[Row], toml_paths: dict[str, set[str]], exit_code: int,
          diags: list[tuple[str, int, str]], ranges: dict[str, tuple[int, int]]) -> list[str]:
    problems: list[str] = []
    if not rows:
        problems.append("compared nothing: the data file has no rows")
    if not any(toml_paths.values()):
        problems.append("compared nothing: clippy.toml has no disallowed-* entries")
    if problems:
        return problems
    for row in rows:
        if row.path not in toml_paths.get(row.key, set()):
            problems.append(f"data line {row.line}: `{row.path}` is not listed under {row.key} in clippy.toml")
    if exit_code == 0:
        problems.append("clippy exited 0 although every canary function calls a banned item")
    used: set[int] = set()
    for i, row in enumerate(rows):
        lo, hi = ranges[f"canary_{i}"]
        hits = [j for j, (_, ln, _) in enumerate(diags) if lo <= ln <= hi]
        used.update(hits)
        named = [j for j in hits if f"`{row.path}`" in diags[j][2]]
        if len(hits) != 1 or len(named) != 1:
            problems.append(f"data line {row.line}: `{row.path}` got {len(hits)} disallowed diagnostic(s), "
                            f"{len(named)} naming it (expected exactly 1)")
    lo, hi = ranges[CONTROL]
    control = [j for j, (_, ln, _) in enumerate(diags) if lo <= ln <= hi]
    if control:
        problems.append(f"the clean control function got {len(control)} disallowed diagnostic(s)")
    stray = [diags[j] for j in range(len(diags)) if j not in used and j not in control]
    if stray:
        problems.append(f"{len(stray)} disallowed diagnostic(s) outside any canary function: {stray}")
    return problems


def run_clippy(root: Path, lib: str) -> tuple[int, str, str]:
    tmp = Path(tempfile.mkdtemp(prefix="clippy-canary-"))
    try:
        cargo = root / "Cargo.toml"
        edition = re.search(r'^edition\s*=\s*"([^"]+)"', cargo.read_text(encoding="utf-8"), re.M)
        manifest = ("[package]\nname = \"clippy-canary\"\nversion = \"0.0.0\"\n"
                    f"edition = \"{edition.group(1) if edition else '2021'}\"\npublish = false\n\n"
                    "[lib]\npath = \"src/lib.rs\"\n\n[workspace]\n\n"
                    + lints_tables(cargo.read_text(encoding="utf-8")))
        (tmp / "src").mkdir()
        (tmp / "Cargo.toml").write_text(manifest, encoding="utf-8")
        (tmp / "src" / "lib.rs").write_text(lib, encoding="utf-8")
        shutil.copyfile(root / "clippy.toml", tmp / "clippy.toml")
        for name in ("rust-toolchain.toml", "rust-toolchain"):
            if (root / name).exists():
                shutil.copyfile(root / name, tmp / name)
        p = subprocess.run(["cargo", "clippy", "--lib", "--quiet", "--message-format=json", "--", *CLIPPY_FLAGS],
                           cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout, p.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calls", type=Path, default=CALLS)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)

    rows, problems = parse_rows(args.calls.read_text(encoding="utf-8"))
    toml_paths = clippy_toml_paths((args.root / "clippy.toml").read_text(encoding="utf-8"))
    if not problems and rows and any(toml_paths.values()):
        lib, ranges = render_lib(rows)
        code, stdout, stderr = run_clippy(args.root, lib)
        diags = disallowed_diagnostics(stdout)
        problems = check(rows, toml_paths, code, diags, ranges)
        if problems and not diags:
            sys.stderr.write(stderr[-4000:])
    else:
        diags = []
        problems += check(rows, toml_paths, 1, [], {})
    if problems:
        for p in problems:
            print(f"clippy_canary: {p}", file=sys.stderr)
        return 1
    entries = sum(len(v) for v in toml_paths.values())
    print(f"clippy_canary: {len(rows)} banned call(s) planted, {len(diags)} disallowed diagnostic(s), "
          f"control clean, {entries} clippy.toml entries read")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Every type-erasing entry point of src/ has a recorded handshake decision
(ALICE-Law G7).

Entry points scanned:
  * C ABI: every `extern "C" fn` in src/ffi.rs
  * Python: every fn in a `#[pymethods]` impl and every `#[pyfunction]` in
    src/python.rs
  * bytes decoders: every `pub fn from_bytes` / `from_compact_bytes` in src/,
    keyed by file and the type of the enclosing impl

Each must appear in scripts/boundaries.toml as `"<kind>::<name>" = "<decision>"`
where the decision is either
  * `handshake: <test>` - the entry point checks the peer's semantics
    identifier, and `fn <test>` exists in src/ or tests/ (the mismatch test), or
  * `local: <reason>` - no Law-bearing value from another producer crosses it
    (reason at least MIN_REASON characters).

The check fails on an unlisted entry point, a listed one that no longer
exists, a handshake whose test function is missing, a short reason, or when it
found no entry point at all (it compared nothing).

usage: scripts/boundary_guard.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

MIN_REASON = 12
EXTERN_RE = re.compile(r'\bextern\s+"C"\s+fn\s+(\w+)')
FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+(\w+)")
DECODER_RE = re.compile(r"\bpub\s+fn\s+(from_bytes|from_compact_bytes)\b")
IMPL_RE = re.compile(r"^\s*impl\b[^{]*\b(\w+)\s*\{")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(line.split("//", 1)[0] for line in text.split("\n"))


def scan(root: Path) -> tuple[set[str], int]:
    found: set[str] = set()
    ffi = root / "src" / "ffi.rs"
    if ffi.exists():
        for m in EXTERN_RE.finditer(strip_comments(ffi.read_text(encoding="utf-8"))):
            found.add(f"ffi::{m.group(1)}")
    py = root / "src" / "python.rs"
    if py.exists():
        depth = 0
        in_methods = False
        methods_depth = 0
        cls = ""
        pending_methods = pending_function = False
        for line in strip_comments(py.read_text(encoding="utf-8")).split("\n"):
            s = line.strip()
            if s == "#[pymethods]":
                pending_methods = True
            elif s == "#[pyfunction]":
                pending_function = True
            im = IMPL_RE.match(line)
            if im and pending_methods:
                in_methods, methods_depth, cls = True, depth, im.group(1)
                pending_methods = False
            fm = FN_RE.match(line)
            if fm:
                if in_methods and depth == methods_depth + 1:
                    found.add(f"python::{cls}.{fm.group(1)}")
                elif pending_function:
                    found.add(f"python::{fm.group(1)}")
                pending_function = False
            depth += line.count("{") - line.count("}")
            if in_methods and depth <= methods_depth:
                in_methods = False
    for path in sorted((root / "src").rglob("*.rs")):
        rel = path.relative_to(root / "src").as_posix()
        owner = ""
        for line in strip_comments(path.read_text(encoding="utf-8")).split("\n"):
            im = IMPL_RE.match(line)
            if im:
                owner = im.group(1)
            m = DECODER_RE.search(line)
            if m:
                found.add(f"decode::{rel}::{owner}::{m.group(1)}")
    files = len(list((root / "src").rglob("*.rs"))) if (root / "src").exists() else 0
    return found, files


def test_names(root: Path) -> set[str]:
    names: set[str] = set()
    for d in ("src", "tests"):
        for path in (root / d).rglob("*.rs") if (root / d).exists() else []:
            names.update(re.findall(r"\bfn\s+(\w+)", path.read_text(encoding="utf-8")))
    return names


def check(root: Path) -> list[str]:
    listed: dict[str, str] = tomllib.loads(
        (root / "scripts" / "boundaries.toml").read_text(encoding="utf-8")).get("boundary", {})
    found, files = scan(root)
    if not found:
        return [f"no type-erasing entry point found in {files} files under src/ (compared nothing)"]
    tests = test_names(root)
    problems = []
    for key in sorted(found - listed.keys()):
        problems.append(f"{key}: entry point with no handshake decision in scripts/boundaries.toml")
    for key in sorted(listed.keys() - found):
        problems.append(f"{key}: listed but not found (stale entry)")
    for key, decision in sorted(listed.items()):
        kind, _, rest = decision.partition(":")
        rest = rest.strip()
        if kind == "handshake":
            if rest not in tests:
                problems.append(f"{key}: handshake test `fn {rest}` not found in src/ or tests/")
        elif kind == "local":
            if len(rest) < MIN_REASON:
                problems.append(f"{key}: reason shorter than {MIN_REASON} characters")
        else:
            problems.append(f"{key}: decision must start with `handshake:` or `local:`")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    root = Path(ap.parse_args(argv).root)
    problems = check(root)
    for p in problems:
        print(f"boundary-guard: {p}", file=sys.stderr)
    if problems:
        return 1
    found, _ = scan(root)
    listed = tomllib.loads((root / "scripts" / "boundaries.toml").read_text(encoding="utf-8"))["boundary"]
    hs = sum(1 for k in found if listed[k].startswith("handshake"))
    print(f"boundary-guard: {len(found)} entry points: {hs} handshake (each with its mismatch test), "
          f"{len(found) - hs} local, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())

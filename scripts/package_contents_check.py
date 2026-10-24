#!/usr/bin/env python3
"""Check the file list of the published .crate.

`Cargo.toml` names what the package contains with `include` (an allowlist).
This check runs `cargo package --list` and fails when the list

- contains a forbidden path (CI configuration, helper scripts, fuzz targets,
  editor / toolchain settings),
- misses a Cargo target (the library, a test, a bench, an example), as
  reported by `cargo metadata`,
- misses a required file (README, licenses, the C / C# bindings),
- misses a file that a packaged Rust source pulls in with `include!` /
  `include_str!` / `include_bytes!` or `#[path = "..."]` (the crate, or
  its tests, would not compile from crates.io),
- misses a `tests/fixtures/...` file that a packaged source names, or any
  tracked file under a `tests/fixtures/...` directory it names,
- is empty, or no target was compared (the check would compare nothing).

Usage: python scripts/package_contents_check.py
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN_PREFIXES = (
    ".github/",
    ".cargo/",
    "scripts/",
    "fuzz/",
    "target/",
)
FORBIDDEN_FILES = {
    ".gitignore",
    "clippy.toml",
    "deny.toml",
    "rust-toolchain.toml",
    "pyproject.toml",
    "CONTRIBUTING.md",
}
REQUIRED_FILES = (
    "Cargo.toml",
    "README.md",
    "LICENSE-Proprietary",
    "bindings/alice_edge.h",
    "bindings/AliceEdge.cs",
)

EMBED_RE = re.compile(
    r'(?:\binclude(?:_str|_bytes)?!\s*\(\s*"([^"]+)"\s*\))|(?:#\[path\s*=\s*"([^"]+)"\])'
)
# fixtures that tests open at run time (`CARGO_MANIFEST_DIR` + this path)
FIXTURE_RE = re.compile(r'"(tests/fixtures/[^"]*[^"/])"')


def forbidden(path: str) -> bool:
    return path in FORBIDDEN_FILES or path.startswith(FORBIDDEN_PREFIXES)


def embedded_paths(rel: str, text: str) -> list[str]:
    """Package-relative paths that `rel` embeds; a path that leaves the
    package is returned with its leading `../` so that it can never match."""
    out = []
    for m in EMBED_RE.finditer(text):
        target = m.group(1) or m.group(2)
        out.append(posixpath.normpath(posixpath.join(posixpath.dirname(rel), target)))
    return out


def check(
    listing: list[str], targets: list[str], sources: dict[str, str], tracked: list[str]
) -> tuple[list[str], int]:
    """(errors, number of targets compared)

    listing: `cargo package --list` lines, targets: package-relative source
    paths of every Cargo target, sources: text of each packaged `.rs` file,
    tracked: files git tracks (to expand a fixture directory).
    """
    errors = []
    files = {line.strip() for line in listing if line.strip()}
    if not files:
        return ["the package list is empty"], 0
    for path in sorted(files):
        if forbidden(path):
            errors.append(f"{path}: must not be published (forbidden path)")
    for path in REQUIRED_FILES:
        if path not in files:
            errors.append(f"{path}: required file missing from the package")
    compared = 0
    for path in targets:
        compared += 1
        if path not in files:
            errors.append(f"{path}: Cargo target missing from the package")
    if compared == 0:
        errors.append("no Cargo target was compared")
    for rel, text in sorted(sources.items()):
        for dep in embedded_paths(rel, text):
            if dep not in files:
                errors.append(f"{rel}: embeds {dep}, which is not in the package")
        for fixture in FIXTURE_RE.findall(text):
            under = [f for f in tracked if f.startswith(fixture + "/")]
            needed = under if under else [fixture]
            for f in needed:
                if f not in files:
                    errors.append(f"{rel}: reads {f}, which is not in the package")
    return errors, compared


def cargo_targets() -> list[str]:
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    pkg = next(p for p in json.loads(out)["packages"] if p["name"] == "alice-edge")
    base = os.path.dirname(pkg["manifest_path"])
    return sorted(
        os.path.relpath(t["src_path"], base).replace(os.sep, "/") for t in pkg["targets"]
    )


def package_list() -> list[str]:
    out = subprocess.run(
        ["cargo", "package", "--list", "--allow-dirty"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [line.replace("\\", "/") for line in out.splitlines()]


def main() -> int:
    listing = package_list()
    sources = {}
    for rel in listing:
        if rel.endswith(".rs"):
            with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                sources[rel] = f.read()
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    errors, compared = check(listing, cargo_targets(), sources, tracked)
    for e in errors:
        print(f"error: {e}")
    if errors:
        return 1
    print(f"package contents OK: {len(listing)} files, {compared} Cargo targets present, "
          f"{len(sources)} sources scanned for embedded files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

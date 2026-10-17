#!/usr/bin/env python3
"""The local ../ALICE-* checkouts are the commits CI builds against.

Until the versions a manifest needs are on crates.io, it takes them from sibling
checkouts ("../ALICE-Zip"); CI clones each at the full SHA in .github/alice-siblings
("<Repo> <40-hex SHA>" per line, "#" comments) through .github/actions/alice-siblings.
This check makes the local build use the same commits. It fails when a line is
malformed or repeated, when nothing is listed, when a manifest names a sibling path
that is not listed or a listed repository is named by no manifest, when a manifest
path does not resolve to the sibling directory, or when a sibling is missing, at
another commit or has uncommitted changes.

    python3 scripts/alice_siblings_check.py [repo root]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# a POSIX string, not a Path: it appears in messages that must read the same on every OS
LIST_FILE = ".github/alice-siblings"
MANIFESTS = ("Cargo.toml", "fuzz/Cargo.toml")
LINE_RE = re.compile(r"(ALICE-[A-Za-z0-9-]+) ([0-9a-f]{40})")
PATH_RE = re.compile(r'path\s*=\s*"((?:\.\./)+)(ALICE-[A-Za-z0-9-]+)"')


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def read_list(root: Path) -> tuple[dict[str, str], list[str]]:
    listed: dict[str, str] = {}
    problems: list[str] = []
    path = root / LIST_FILE
    if not path.is_file():
        return listed, [f"{LIST_FILE} is missing"]
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = " ".join(raw.split("#", 1)[0].split())
        if not line:
            continue
        m = LINE_RE.fullmatch(line)
        if not m:
            problems.append(f"{LIST_FILE}:{no}: not '<ALICE-Repo> <40-hex SHA>': {raw!r}")
        elif m.group(1) in listed:
            problems.append(f"{LIST_FILE}:{no}: {m.group(1)} is listed twice")
        else:
            listed[m.group(1)] = m.group(2)
    if not listed and not problems:
        problems.append(f"{LIST_FILE} lists no repository (compared nothing)")
    return listed, problems


def check(root: Path) -> list[str]:
    listed, problems = read_list(root)
    if problems:
        return problems

    named: set[str] = set()
    for rel in MANIFESTS:
        manifest = root / rel
        if not manifest.is_file():
            continue
        for m in PATH_RE.finditer(manifest.read_text(encoding="utf-8")):
            repo = m.group(2)
            named.add(repo)
            if (manifest.parent / f"{m.group(1)}{repo}").resolve() != (root / ".." / repo).resolve():
                problems.append(f"{rel} names {m.group(0)}, not the sibling ../{repo}")
            if repo not in listed:
                problems.append(f"{rel} names ../{repo}, which {LIST_FILE} does not list")
    for repo in sorted(set(listed) - named):
        problems.append(f"{LIST_FILE} lists {repo}, which no manifest names as a path")

    for repo, ref in listed.items():
        sibling = (root / ".." / repo).resolve()
        if not (sibling / ".git").exists():
            problems.append(f"../{repo} is not a git checkout; clone {repo} there at {ref}")
            continue
        head = git(sibling, "rev-parse", "HEAD")
        if head != ref:
            problems.append(f"../{repo} is at {head}, {LIST_FILE} says {ref}")
        if git(sibling, "status", "--porcelain", "--untracked-files=no"):
            problems.append(f"../{repo} has uncommitted changes; CI builds the clean commit")
    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    problems = check(root)
    for p in problems:
        print(f"alice_siblings_check: {p}")
    if problems:
        return 1
    listed, _ = read_list(root)
    for repo, ref in listed.items():
        print(f"alice_siblings_check: ../{repo} at {ref}")
    print(f"alice_siblings_check: {len(listed)} sibling(s), each at the commit CI builds against")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""upstream_previous_ids_check.py — another crate's PREVIOUS_*_SEMANTICS_IDS is never read by position

A crate's list of previous semantics identifiers is the crate's own data: on a major fold it goes back to `&[]`,
which is a correct state. A downstream crate that reads `upstream::PREVIOUS_X_SEMANTICS_IDS[0]` (or `.first()`,
`.get(0)`, ...) then panics or loses its value the moment it resolves the new upstream. Read the list as a whole:
iterate it, `contains()`, `len()`, `is_empty()`, pass it on; to name one identifier, use the upstream's named
constant.

This is a lint on a coding pattern: it catches the known positional forms (an index or slice, .get / .first /
.split_first / .last / .nth / .next, through a path, a `use` import or alias, or a `let` binding of the list); it is
not a proof that no code depends on the list's length.

Another crate: the first path segment is not one of the repository's own crates (every [package] name of a
Cargo.toml in the tree, `-` read as `_`; `crate`, `self`, `super`). A crate's own list is checked by its own CI.
Comments and string literals are left out; doc comments are kept (a doc example runs as a doctest).

usage: upstream_previous_ids_check.py [--repo DIR]
exit 0: no finding (the count of files scanned is printed), 1: a finding (path:line and the fix), 2: no .rs file
scanned (nothing compared)
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LIST_NAME = r"PREVIOUS_[A-Z0-9_]*SEMANTICS_IDS"
PATH_LIST_RE = re.compile(r"(?<![A-Za-z0-9_])((?:::)?(?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)+)(" + LIST_NAME + r")\b")
FIX = ("read the list as a whole (iterate it, contains(), len(), is_empty()) or name the identifier you mean with the "
       "upstream's constant: the upstream may empty the list on a major version")


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    text: str


def own_crates(root: Path) -> set[str]:
    """the [package] names of every tracked Cargo.toml, `-` read as `_`"""
    out = set()
    files = subprocess.run(["git", "-C", str(root), "ls-files", "--", "Cargo.toml", "*/Cargo.toml"],
                           capture_output=True, text=True).stdout.split()
    for f in files:
        text = (root / f).read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^\[package\][^\[]*?^name\s*=\s*\"([^\"]+)\"", text, re.M | re.S)
        if m:
            out.add(m.group(1).replace("-", "_"))
    return out


def strip(text: str) -> str:
    """the text with `//` and `/* */` comments and string / char literals blanked (newlines kept); `///` and `//!`
    doc comments are kept, without their marker (a doc example is code)"""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("///", i) or text.startswith("//!", i):
            out.append("   ")
            i += 3
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        if text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
            continue
        m = re.match(r'b?r(#*)"', text[i:])
        if m:
            end = text.find('"' + m.group(1), i + m.end())
            j = n if end < 0 else end + 1 + len(m.group(1))
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
            continue
        if text[i] == '"' or text.startswith('b"', i):
            j = i + (2 if text[i] == "b" else 1)
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(re.sub(r"[^\n]", " ", text[i:j + 1]))
            i = j + 1
            continue
        if text[i] == "'" and re.match(r"'(\\.|[^\\'])'", text[i:]):
            m = re.match(r"'(\\.|[^\\'])'", text[i:])
            out.append(" " * m.end())
            i += m.end()
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


ACCESS = [
    ("index", re.compile(r"^\s*\[")),
    ("get", re.compile(r"^\s*\.\s*get\s*\(")),
    ("first", re.compile(r"^\s*\.\s*(?:split_)?first\s*\(")),
    ("last", re.compile(r"^\s*\.\s*(?:split_)?last\s*\(")),
    ("nth", re.compile(r"^\s*\.\s*iter\s*\(\s*\)\s*(?:\.\s*rev\s*\(\s*\)\s*)?\.\s*nth\s*\(")),
    ("next", re.compile(r"^\s*\.\s*iter\s*\(\s*\)\s*(?:\.\s*rev\s*\(\s*\)\s*)?\.\s*next\s*\(")),
]


def _access(after: str) -> str | None:
    for kind, rx in ACCESS:
        if rx.match(after):
            return kind
    return None


def _foreign(prefix: str, own: set[str]) -> bool:
    first = prefix.lstrip(":").split("::")[0].strip()
    return bool(first) and first not in own and first not in ("crate", "self", "super", "Self")


def scan_text(path: str, text: str, own: set[str]) -> list[Finding]:
    """the positional reads of another crate's list in one file"""
    code = strip(text)
    names: set[str] = set()   # local names that stand for another crate's list
    defined = set(re.findall(r"\b(?:const|static)\s+(" + LIST_NAME + r")\s*:", code))
    for m in re.finditer(r"\buse\s+([^;]+);", code):
        body = re.sub(r"\s+", "", m.group(1))
        root = body.lstrip(":").split("::")[0]
        if not _foreign(root + "::", own):
            continue
        for lm in re.finditer(r"(" + LIST_NAME + r")(?:as([A-Za-z_][A-Za-z0-9_]*))?", body):
            names.add(lm.group(2) or lm.group(1))
        if body.endswith("::*") or "::*," in body or "{*" in body:
            names |= set(re.findall(r"\b(" + LIST_NAME + r")\b", code)) - defined
    # a let binding of another crate's list: `let p = x::LIST;` / `let p: &[..] = &x::LIST;` / of an imported name
    for m in re.finditer(r"\blet\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+?)?=\s*&?\s*([^;]+);", code):
        rhs = m.group(2).strip()
        pm = PATH_LIST_RE.fullmatch(rhs)
        if (pm and _foreign(pm.group(1), own)) or rhs in names:
            names.add(m.group(1))
    out = []
    lines_at = [0] + [i + 1 for i, ch in enumerate(code) if ch == "\n"]

    def line_of(pos: int) -> int:
        import bisect
        return bisect.bisect_right(lines_at, pos)

    for m in PATH_LIST_RE.finditer(code):
        if not _foreign(m.group(1), own):
            continue
        kind = _access(code[m.end():m.end() + 80])
        if kind:
            out.append(Finding(path, line_of(m.start()), kind, (m.group(1) + m.group(2)).replace(" ", "")))
    for name in sorted(names):
        for m in re.finditer(r"(?<![A-Za-z0-9_:])" + re.escape(name) + r"\b", code):
            before = code[max(0, m.start() - 6):m.start()]
            if re.search(r"\b(let|as|use)\s*$", before) or re.search(r"(let\s+(mut\s+)?)$", before):
                continue
            kind = _access(code[m.end():m.end() + 80])
            if kind:
                out.append(Finding(path, line_of(m.start()), kind, name))
    return sorted(out, key=lambda f: (f.line, f.kind))


def render(f: Finding) -> str:
    return f"{f.path}:{f.line}: {f.text} read by position ({f.kind}): {FIX}"


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    root = Path(a.repo).resolve()
    own = own_crates(root)
    files = subprocess.run(["git", "-C", str(root), "ls-files", "--", "*.rs"], capture_output=True,
                           text=True).stdout.split()
    if not files:
        print("upstream-previous-ids: 0 files scanned (no tracked .rs): nothing compared", file=sys.stderr)
        return 2
    found = []
    for f in files:
        p = root / f
        if p.is_file():
            found += scan_text(f, p.read_text(encoding="utf-8", errors="replace"), own)
    for f in found:
        print(f"upstream-previous-ids: {render(f)}", file=sys.stderr)
    print(f"upstream-previous-ids: {len(files)} file{'s' if len(files) != 1 else ''} scanned, {len(found)} "
          f"finding{'s' if len(found) != 1 else ''}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

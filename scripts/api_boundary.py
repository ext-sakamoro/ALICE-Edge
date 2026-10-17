#!/usr/bin/env python3
"""Enumerate the crate's boundary from its public API and compare it with a
reasoned manifest (ALICE-Law G7 / G8).

The crate is documented with rustdoc's JSON output (unstable, enabled with
RUSTC_BOOTSTRAP=1 on the pinned toolchain, the same way as
scripts/clippy_external_coverage.py of ALICE-DB / ALICE-Crypto), with
--document-hidden-items (a #[doc(hidden)] pub item is callable too) and the
features named in the manifest. The JSON format_version must be one this
script was written against (SUPPORTED_FORMATS) or it stops: a new toolchain is
a reason to re-check the walker, not to guess.

Enumerated, from the items of this crate:
  * fn: every function or method whose parameters, return type, generic
    parameters or where clause contain an element type of bytes, text or a
    reader: the primitives u8 / i8 / str / char, or a path ending in one of
    BYTE_PATHS (String, Read, BufRead, Path, File, ...), at any depth (inside
    references, slices, arrays, tuples, Box / Vec / Cow / Option / Result,
    impl Trait / dyn Trait bounds, associated-type constraints such as
    `Iterator<Item = u8>`, and through this crate's type aliases)
  * abi: every function with a non-Rust ABI (extern "C" and the like)
  * impl: every non-blanket trait implementation of a trait in SERDE_TRAITS
    (Serialize, Deserialize: derive or manual) or READ_TRAITS (Read, BufRead,
    FromStr), and of From / TryFrom when the trait's type argument contains a
    byte, text or reader element
  * reexport: every `pub use` of an item outside this crate (the item's own
    signature is not visible here, so the re-export itself needs a decision)

Each must appear in the manifest (scripts/api_boundary.toml) as
`"<kind>::<key>" = "<decision>"`, the decision being `handshake: <test fn>`
(the entry point checks the producer's identifier; the test function must
exist in src/ or tests/) or `local: <reason>` (no Law-bearing value from
another producer crosses it). A second item with the same key gets `#2`, `#3`.

The check fails on an enumerated item that is not listed, a listed item that
was not enumerated (stale), a missing handshake test, a reason shorter than
MIN_REASON, an unsupported format_version, a failed rustdoc, and when it
enumerated nothing (compared nothing).

usage: scripts/api_boundary.py [--root DIR] [--json FILE]
  --json FILE  read an existing rustdoc JSON instead of running rustdoc (tests)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

SUPPORTED_FORMATS = frozenset({60})
MIN_REASON = 12
BYTE_PRIMITIVES = frozenset({"u8", "i8", "str", "char"})
BYTE_PATHS = frozenset({"String", "Read", "BufRead", "Bytes", "BytesMut", "OsStr", "OsString", "Path",
                        "PathBuf", "File", "TcpStream", "UdpSocket", "Stdin"})
SERDE_TRAITS = frozenset({"Serialize", "Deserialize", "Deserializer", "Serializer"})
READ_TRAITS = frozenset({"Read", "BufRead", "FromStr"})
CONVERSION_TRAITS = frozenset({"From", "TryFrom"})


def last(path: str) -> str:
    return path.split("::")[-1]


class Walker:
    def __init__(self, doc: dict):
        if doc.get("format_version") not in SUPPORTED_FORMATS:
            raise SystemExit(f"api_boundary: rustdoc JSON format_version {doc.get('format_version')} "
                             f"is not in {sorted(SUPPORTED_FORMATS)}; check the walker against the "
                             "new format, then add it")
        self.idx = doc["index"]
        self.seen_alias: set[str] = set()

    def item(self, i):
        return self.idx.get(str(i)) if i is not None else None

    def args_types(self, args):
        if not args:
            return
        ab = args.get("angle_bracketed")
        if ab:
            for a in ab.get("args", []):
                if isinstance(a, dict) and "type" in a:
                    yield a["type"]
            for c in ab.get("constraints", []):
                binding = c.get("binding", {})
                if "equality" in binding and isinstance(binding["equality"], dict) \
                        and "type" in binding["equality"]:
                    yield binding["equality"]["type"]
                for b in binding.get("constraint", []) or []:
                    yield {"__bound__": b}
        pa = args.get("parenthesized")
        if pa:
            yield from pa.get("inputs", [])
            if pa.get("output"):
                yield pa["output"]

    def bound_hits(self, bound) -> bool:
        tb = bound.get("trait_bound") if isinstance(bound, dict) else None
        if not tb:
            return False
        tr = tb["trait"]
        if last(tr["path"]) in BYTE_PATHS:
            return True
        return any(self.hits(t) for t in self.args_types(tr.get("args")))

    def hits(self, ty) -> bool:
        """Whether a rustdoc type contains a byte, text or reader element."""
        if ty is None or not isinstance(ty, dict):
            return False
        if "__bound__" in ty:
            return self.bound_hits(ty["__bound__"])
        k, v = next(iter(ty.items()))
        if k == "primitive":
            return v in BYTE_PRIMITIVES
        if k in ("slice",):
            return self.hits(v)
        if k == "array":
            return self.hits(v["type"])
        if k in ("borrowed_ref", "raw_pointer"):
            return self.hits(v["type"])
        if k == "tuple":
            return any(self.hits(t) for t in v)
        if k == "resolved_path":
            if last(v["path"]) in BYTE_PATHS:
                return True
            alias = self.item(v.get("id"))
            if alias and "type_alias" in alias.get("inner", {}) and str(v.get("id")) not in self.seen_alias:
                self.seen_alias.add(str(v.get("id")))
                try:
                    if self.hits(alias["inner"]["type_alias"]["type"]):
                        return True
                finally:
                    self.seen_alias.discard(str(v.get("id")))
            return any(self.hits(t) for t in self.args_types(v.get("args")))
        if k in ("impl_trait", "dyn_trait"):
            bounds = v if k == "impl_trait" else [{"trait_bound": {"trait": t["trait"]}} for t in v["traits"]]
            return any(self.bound_hits(b) for b in bounds)
        if k == "qualified_path":
            return self.hits(v.get("self_type")) or (
                v.get("trait") is not None and last(v["trait"]["path"]) in BYTE_PATHS)
        if k == "function_pointer":
            sig = v["sig"]
            return any(self.hits(t) for _, t in sig["inputs"]) or self.hits(sig.get("output"))
        return False

    def generics_hit(self, g) -> bool:
        for p in g.get("params", []):
            kind = p.get("kind", {})
            t = kind.get("type")
            if t:
                if any(self.bound_hits(b) for b in t.get("bounds", [])) or self.hits(t.get("default")):
                    return True
            c = kind.get("const")
            if c and self.hits(c.get("type")):
                return True
        for w in g.get("where_predicates", []):
            bp = w.get("bound_predicate")
            if bp and (self.hits(bp.get("type")) or any(self.bound_hits(b) for b in bp.get("bounds", []))):
                return True
        return False


def type_name(ty) -> str:
    if isinstance(ty, dict):
        k, v = next(iter(ty.items()))
        if k == "resolved_path":
            return last(v["path"])
        if k == "primitive":
            return v
        if k == "borrowed_ref":
            return type_name(v["type"])
        if k == "generic":
            return v
    return "?"


def enumerate_boundary(doc: dict) -> list[str]:
    w = Walker(doc)
    keys: list[str] = []
    owner: dict[str, str] = {}
    for i, it in w.idx.items():
        if it.get("crate_id") != 0:
            continue
        inner = it.get("inner", {})
        if "impl" in inner:
            imp = inner["impl"]
            if imp.get("blanket_impl") is not None or imp.get("is_synthetic"):
                continue
            for m in imp.get("items", []):
                tr = imp.get("trait")
                owner[str(m)] = (f"<{last(tr['path'])} for {type_name(imp['for'])}>" if tr
                                 else type_name(imp["for"]))
            tr = imp.get("trait")
            if tr:
                name = last(tr["path"])
                conv = name in CONVERSION_TRAITS and any(w.hits(t) for t in w.args_types(tr.get("args")))
                if name in SERDE_TRAITS or name in READ_TRAITS or conv:
                    keys.append(f"impl::{name} for {type_name(imp['for'])}")
    paths = doc.get("paths", {})
    for i, it in w.idx.items():
        if it.get("crate_id") != 0:
            continue
        inner = it.get("inner", {})
        if "function" in inner:
            f = inner["function"]
            if str(i) in owner:
                name = f"{owner[str(i)]}::{it.get('name')}"
            elif str(i) in paths:
                name = "::".join(paths[str(i)]["path"][1:])
            else:
                name = it.get("name") or "?"
            abi = f["header"].get("abi")
            if abi != "Rust":
                keys.append(f"abi::{name}")
                continue
            sig = f["sig"]
            if any(w.hits(t) for _, t in sig["inputs"]) or w.hits(sig.get("output")) \
                    or w.generics_hit(f.get("generics", {})):
                keys.append(f"fn::{name}")
        elif "use" in inner and it.get("visibility") == "public":
            u = inner["use"]
            target = w.item(u.get("id"))
            if target is None or target.get("crate_id") != 0:
                keys.append(f"reexport::{u.get('source')}")
    # number repeated keys in a stable order
    out: list[str] = []
    count: dict[str, int] = {}
    for k in sorted(keys):
        count[k] = count.get(k, 0) + 1
        out.append(k if count[k] == 1 else f"{k}#{count[k]}")
    return out


def rustdoc(root: Path, features: str) -> dict:
    env = dict(os.environ)
    env["RUSTC_BOOTSTRAP"] = "1"
    env["RUSTDOCFLAGS"] = ""
    cmd = ["cargo", "rustdoc", "-q", "--lib"]
    if features:
        cmd += ["--features", features]
    cmd += ["--", "-Z", "unstable-options", "--output-format", "json", "--document-hidden-items"]
    proc = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:])
        raise SystemExit("api_boundary: rustdoc failed")
    meta = json.loads(subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=root,
                                     stdout=subprocess.PIPE, text=True, check=True).stdout)
    target_dir = Path(meta["target_directory"])
    lib_kinds = {"lib", "rlib", "dylib", "cdylib", "staticlib"}
    lib = next(t["name"] for p in meta["packages"] for t in p["targets"] if lib_kinds & set(t["kind"]))
    return json.loads((target_dir / "doc" / f"{lib.replace('-', '_')}.json").read_text(encoding="utf-8"))


def test_names(root: Path) -> set[str]:
    names: set[str] = set()
    for d in ("src", "tests"):
        if (root / d).exists():
            for path in (root / d).rglob("*.rs"):
                names.update(re.findall(r"\bfn\s+(\w+)", path.read_text(encoding="utf-8")))
    return names


def check(root: Path, doc: dict, listed: dict[str, str]) -> tuple[list[str], list[str]]:
    found = enumerate_boundary(doc)
    if not found:
        return ["enumerated no boundary item (compared nothing)"], found
    tests = test_names(root)
    problems = []
    for key in sorted(set(found) - listed.keys()):
        problems.append(f"{key}: not in scripts/api_boundary.toml (decide handshake or local)")
    for key in sorted(listed.keys() - set(found)):
        problems.append(f"{key}: listed but not enumerated (stale entry)")
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
    return problems, found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    cfg = tomllib.loads((root / "scripts" / "api_boundary.toml").read_text(encoding="utf-8"))
    doc = json.loads(Path(a.json).read_text(encoding="utf-8")) if a.json else rustdoc(root, cfg.get("features", ""))
    problems, found = check(root, doc, cfg.get("boundary", {}))
    for p in problems:
        print(f"api_boundary: {p}", file=sys.stderr)
    if problems:
        return 1
    hs = sum(1 for k in found if cfg["boundary"][k].startswith("handshake"))
    kinds = {}
    for k in found:
        kinds[k.split("::", 1)[0]] = kinds.get(k.split("::", 1)[0], 0) + 1
    print(f"api_boundary: {len(found)} items ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}): "
          f"{hs} handshake, {len(found) - hs} local, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())

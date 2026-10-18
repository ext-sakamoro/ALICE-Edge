#!/usr/bin/env python3
"""The published .crate builds on its own, or names exactly what blocks it.

    python3 scripts/package_integrity.py [--expect-in-lock NAME ...]   (CI: security-audit.yml)
    python3 scripts/package_integrity.py --publish-gate                (before a publish)
    python3 scripts/package_integrity.py --member alice-lol [...]      (a workspace member)
    python3 scripts/package_integrity.py --member alice-lol --patches  (the patch arguments only)

Default mode packages this crate (`cargo package --no-verify --allow-dirty`), unpacks the
.crate and builds it as a standalone package twice: with the shipped lock (`--locked`) and
with a fresh resolution. A file the package needs but does not contain, bad metadata or a
dependency that does not resolve fails here and not only for users of the published crate.

A path dependency on a sibling repository listed in .github/alice-siblings ("<Repo> <SHA>",
cloned next to this one) can require a version that is not on crates.io yet. `cargo
package` can never pass for it, so such a dependency is *blocked*: it is packaged and
built with the sibling patched in (`--config patch.crates-io.<name>.path=...`) and
reported as `blocked: ...`. A dependency is blocked only when all of these hold, checked
and not assumed:
  - its manifest entry names `path = "../<Repo>"` for a listed repository,
  - the sibling's own [package] version satisfies the requirement,
  - no published, non-yanked version in the crates.io index satisfies it.
A sibling whose requirement crates.io already satisfies is not patched, so it cannot hide
a failure. Every other packaging or build error fails. No .crate produced fails.

--root is the repository (the git top-level): .github/alice-siblings is read there and a
sibling is <root>/../<Repo>. --member DIR packages the crate in <root>/DIR instead of the
root crate; its path dependencies may climb any number of `../` as long as they land on
<root>/../<Repo>. A path dependency on another package inside the repository (a workspace
member) is blocked the same way: it names a version, the member's own version satisfies
it, and no published version does (the member is about to be published with this crate).
Any other path dependency (outside the repository and not a listed sibling, or a
directory without a package) fails, except a dev-dependency without a version, which
cargo leaves out of the package.

--patches prints the patch arguments of the blocked dependencies, one per line
(`--config=patch.crates-io.<name>.path="<dir>"`), and nothing else on stdout, for a
script that builds the packaged crate itself (`mapfile -t P < <(...)`, then `"${P[@]}"`).

Options for what a crate's package must hold (each runs with the same patches; a value
that starts with "--" is passed as --opt=VALUE):
  --require-file F / --forbid-prefix P   on `cargo package --list` (an empty list fails)
  --doctests-min N [--doctest-args A]    `cargo test --doc` in the unpacked crate, >= N pass
  --test-in-package "ARGS"               `cargo test ARGS` in the unpacked crate, >= 1 pass
  --locked-check-args "ARGS"             the shipped-lock check (default --all-features)
  --check-args "ARGS"                    the fresh-resolution check (default --all-features)
  --expect-in-lock NAME                  a package the fresh resolution must contain

--publish-gate does not package: it fails when any dependency is blocked, so nothing
that only builds against an unpublished sibling can be published.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alice_siblings_check import read_list  # noqa: E402

DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")
SIBLING_RE = re.compile(r"ALICE-[A-Za-z0-9-]+")
VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")
CMP_RE = re.compile(r"^(>=|<=|>|<|=|\^|~)?\s*(\d+(?:\.\d+){0,2}|\*)$")

Version = tuple[int, int, int]


class Problem(Exception):
    pass


def parse_version(text: str) -> Version:
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        raise Problem(f"not a release version: {text!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def comparators(req: str) -> list[tuple[str, Version]]:
    """Cargo requirement -> (op, version) bounds; unsupported syntax is a Problem."""
    out: list[tuple[str, Version]] = []
    parts = [p.strip() for p in req.split(",")]
    if not parts or any(not p for p in parts):
        raise Problem(f"empty part in requirement {req!r}")
    for part in parts:
        m = CMP_RE.fullmatch(part)
        if not m:
            raise Problem(f"unsupported requirement {req!r}")
        op, ver = m.group(1) or "^", m.group(2)
        if ver == "*":
            continue
        nums = [int(x) for x in ver.split(".")]
        given = len(nums)
        major, minor, patch = (nums + [0, 0])[:3]
        low = (major, minor, patch)
        if op in (">=", ">", "<", "<="):
            out.append((op, low))
        elif op == "=":
            if given == 3:
                out.append(("=", low))
            else:
                out.append((">=", low))
                out.append(("<", (major + 1, 0, 0) if given == 1 else (major, minor + 1, 0)))
        elif op == "~":
            out.append((">=", low))
            out.append(("<", (major + 1, 0, 0) if given == 1 else (major, minor + 1, 0)))
        else:  # caret
            out.append((">=", low))
            if major > 0 or given == 1:
                out.append(("<", (major + 1, 0, 0)))
            elif minor > 0 or given == 2:
                out.append(("<", (0, minor + 1, 0)))
            else:
                out.append(("<", (0, 0, patch + 1)))
    return out


def satisfies(version: Version, req: str) -> bool:
    for op, bound in comparators(req):
        ok = {
            ">=": version >= bound,
            ">": version > bound,
            "<": version < bound,
            "<=": version <= bound,
            "=": version == bound,
        }[op]
        if not ok:
            return False
    return True


def registry_versions(name: str) -> list[Version]:
    """Published, non-yanked release versions of `name` from the crates.io sparse index."""
    n = name.lower()
    prefix = {1: "1", 2: "2", 3: f"3/{n[0]}"}.get(len(n), f"{n[:2]}/{n[2:4]}")
    url = f"https://index.crates.io/{prefix}/{n}"
    req = urllib.request.Request(url, headers={"User-Agent": "alice-package-integrity"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode("utf-8").splitlines()
    out = []
    for line in lines:
        entry = json.loads(line)
        if entry.get("yanked") or "-" in entry["vers"] or "+" in entry["vers"]:
            continue
        out.append(parse_version(entry["vers"]))
    return out


def manifest_deps(pkg_dir: Path, top: Path) -> list[tuple[str, str, dict]]:
    """(table, key, spec) of every dependency of pkg_dir/Cargo.toml; `workspace = true`
    entries are completed from top's [workspace.dependencies] (a path there is relative to top)."""
    manifest = tomllib.loads((pkg_dir / "Cargo.toml").read_text(encoding="utf-8"))
    tables = [(t, manifest.get(t, {})) for t in DEP_TABLES]
    for target in manifest.get("target", {}).values():
        tables += [(t, target.get(t, {})) for t in DEP_TABLES]
    inherited: dict = {}
    if any(isinstance(spec, dict) and spec.get("workspace") is True for _, tbl in tables for spec in tbl.values()):
        inherited = tomllib.loads((top / "Cargo.toml").read_text(encoding="utf-8")).get("workspace", {}).get("dependencies", {})
    out = []
    for table, tbl in tables:
        for key, spec in tbl.items():
            if isinstance(spec, dict) and spec.get("workspace") is True:
                base = inherited.get(key)
                if base is None:
                    raise Problem(f"{key}: `workspace = true` but [workspace.dependencies] has no {key}")
                base = {"version": base} if isinstance(base, str) else dict(base)
                if "path" in base:
                    base["path"] = os.path.normpath(top / base["path"])
                spec = {**base, **{k: v for k, v in spec.items() if k != "workspace"}}
            out.append((table, key, spec))
    return out


def sibling_deps(root: Path, member: str = ".") -> list[dict]:
    """Every path dependency of the packaged crate (any dependency table), classified:
    kind "sibling" (<root>/../<Repo>), "member" (a package inside root) or "unresolved"."""
    # lexical paths (no symlink resolution): a sibling may be a link to a checkout elsewhere
    top = Path(os.path.normpath(root.absolute()))
    pkg_dir = Path(os.path.normpath(top / member))
    deps = []
    for table, key, spec in manifest_deps(pkg_dir, top):
        if not isinstance(spec, dict) or "path" not in spec:
            continue
        dep = {"name": spec.get("package", key), "req": spec.get("version"), "table": table}
        target = Path(os.path.normpath(pkg_dir / spec["path"]))
        if target == pkg_dir:
            continue  # the crate itself (a dev-dependency on "." for features)
        if target == top or top in target.parents:
            deps.append({**dep, "kind": "member", "dir": target})
        elif target.parent == top.parent and SIBLING_RE.fullmatch(target.name):
            deps.append({**dep, "kind": "sibling", "repo": target.name})
        else:
            deps.append({**dep, "kind": "unresolved", "path": spec["path"]})
    return deps


def check_requirement(dep: dict, label: str, noun: str, pkg_toml: Path, versions_of) -> dict | None:
    """None when crates.io satisfies dep's requirement; the blocked entry when only the local
    package `label` does; a Problem otherwise."""
    if not pkg_toml.is_file():
        raise Problem(f"{label}/Cargo.toml is missing")
    pkg = tomllib.loads(pkg_toml.read_text(encoding="utf-8")).get("package", {})
    sib_version = parse_version(str(pkg.get("version", "")))
    if pkg.get("name") != dep["name"]:
        raise Problem(f"{label} is package {pkg.get('name')!r}, not {dep['name']}")
    if not satisfies(sib_version, dep["req"]):
        raise Problem(
            f"the {noun} {label} is {'.'.join(map(str, sib_version))}, "
            f"which does not satisfy {dep['req']!r}"
        )
    published = versions_of(dep["name"])
    if any(satisfies(v, dep["req"]) for v in published):
        return None  # resolvable from crates.io: packaged without the sibling
    top = ".".join(map(str, max(published))) if published else "none"
    return {**dep, "sibling": ".".join(map(str, sib_version)), "registry_max": top,
            "path": str(pkg_toml.parent.resolve())}  # the patch names the real directory


def classify(
    root: Path, versions_of: Callable[[str], list[Version]], member: str = "."
) -> tuple[list[dict], list[str]]:
    """(blocked dependencies, problems) for the path dependencies of the packaged crate."""
    listed, problems = read_list(root)
    if problems:
        return [], problems
    blocked: list[dict] = []
    seen: set[tuple[str, str]] = set()
    try:
        deps = sibling_deps(root, member)
    except Problem as e:
        return [], [str(e)]
    for dep in deps:
        # one entry per (name, place): the same crate named in two tables is checked once,
        # a second spec of it with another path (a target table) is checked on its own
        key = (dep["name"], str(dep.get("repo") or dep.get("dir") or dep.get("path")))
        if key in seen:
            continue
        seen.add(key)
        if dep["kind"] == "unresolved":
            if dep["table"] == "dev-dependencies" and not dep["req"]:
                continue  # cargo leaves a version-less path dev-dependency out of the package
            problems.append(f"{dep['name']}: path {dep['path']!r} is neither a listed sibling "
                            f"(<root>/../<Repo>) nor a package in this repository")
            continue
        if dep["kind"] == "member":
            if not dep["req"]:
                if dep["table"] == "dev-dependencies":
                    continue  # left out of the package by cargo
                problems.append(f"{dep['name']}: path dependency without a version cannot be published")
                continue
            rel = dep["dir"].relative_to(os.path.normpath(root.absolute())).as_posix()
            try:
                b = check_requirement(dep, rel, "workspace member", dep["dir"] / "Cargo.toml", versions_of)
            except Problem as e:
                problems.append(f"{dep['name']}: {e}")
                continue
            if b:
                blocked.append({**b, "member": rel})
            continue
        if not dep["req"] and dep["table"] == "dev-dependencies":
            continue  # cargo leaves a version-less path dev-dependency out of the package
        if dep["repo"] not in listed:
            problems.append(f"{dep['name']}: ../{dep['repo']} is not in .github/alice-siblings")
            continue
        if not dep["req"]:
            problems.append(f"{dep['name']}: path dependency without a version cannot be published")
            continue
        try:
            b = check_requirement(dep, f"../{dep['repo']}", "sibling", root / ".." / dep["repo"] / "Cargo.toml", versions_of)
        except Problem as e:
            problems.append(f"{dep['name']}: {e}")
            continue
        if b:
            blocked.append({**b, "sha": listed[dep["repo"]]})
    return blocked, problems


def report_blocked(blocked: list[dict]) -> None:
    for b in blocked:
        if "member" in b:
            print(
                f"blocked: {b['name']} {b['req']} unpublished (crates.io max {b['registry_max']}), "
                f"satisfied only by the workspace member {b['member']} ({b['name']} {b['sibling']}); "
                f"publish wave: {b['name']}@{b['req']}"
            )
            continue
        print(
            f"blocked: {b['name']} {b['req']} unpublished (crates.io max {b['registry_max']}), "
            f"satisfied only by .github/alice-siblings {b['repo']} {b['sha'][:12]} "
            f"({b['name']} {b['sibling']}); publish wave: {b['name']}@{b['req']}"
        )
    print(f"package_integrity: blocked {len(blocked)}")


def patch_args(blocked: list[dict]) -> list[str]:
    args: list[str] = []
    for b in blocked:
        args += ["--config", f"patch.crates-io.{b['name']}.path={json.dumps(b['path'])}"]
    return args


# (exit code, stdout, stderr): stdout alone is parsed where it is data (`cargo package --list`)
Runner = Callable[[list[str], Path], tuple[int, str, str]]


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    env = {**os.environ, "CARGO_TERM_COLOR": "never"}
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


TEST_RESULT_RE = re.compile(r"^test result: (\w+)\. (\d+) passed; (\d+) failed", re.M)


def passed_tests(out: str) -> tuple[int, int]:
    """(passed, failed) summed over every `test result:` line of a cargo test run."""
    passed = failed = 0
    for m in TEST_RESULT_RE.finditer(out):
        passed += int(m.group(2))
        failed += int(m.group(3))
    return passed, failed


def check_list(root: Path, patches: list[str], opts: argparse.Namespace, runner: Runner) -> list[str]:
    """`cargo package --list`: not empty, no forbidden prefix, every required file present."""
    if not opts.require_file and not opts.forbid_prefix:
        return []
    rc, listing, err = runner(["cargo", "package", "--list", "--allow-dirty", *patches], root)
    if rc != 0:
        return [f"cargo package --list failed:\n{listing}{err}"]
    # stdout only: a warning on stderr is not a packaged file
    files = [line.strip() for line in listing.splitlines() if line.strip()]
    if not files:
        return ["cargo package --list listed no file (compared nothing)"]
    problems = [f"the package lacks {f}" for f in opts.require_file if f not in files]
    for prefix in opts.forbid_prefix:
        problems += [f"the package contains {f} (forbidden prefix {prefix})" for f in files if f.startswith(prefix)]
    if not problems:
        print(f"package_integrity: package lists {len(files)} files, {len(opts.require_file)} required "
              f"present, {len(opts.forbid_prefix)} forbidden prefixes absent")
    return problems


def package_and_build(
    root: Path, blocked: list[dict], opts: argparse.Namespace, runner: Runner, target: Path
) -> list[str]:
    """root is the packaged crate's directory (the repository, or the member's directory)"""
    patches = patch_args(blocked)
    problems = check_list(root, patches, opts, runner)
    if problems:
        return problems
    pkg_meta = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["package"]
    crate = target / "package" / f"{pkg_meta['name']}-{pkg_meta['version']}.crate"
    started = time.time()
    rc, stdout, stderr = runner(["cargo", "package", "--no-verify", "--allow-dirty", *patches], root)
    out = stdout + stderr
    if rc != 0:
        return [f"cargo package failed:\n{out}"]
    if not crate.is_file() or crate.stat().st_mtime < started - 1:
        return [f"cargo package wrote no {crate.name} in {crate.parent} (inspected nothing)"]
    print(f"package_integrity: packaged {crate.name} ({crate.stat().st_size} bytes), packages inspected 1")
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(crate) as tar:
            tar.extractall(tmp, filter="data")
        dirs = [d for d in Path(tmp).iterdir() if d.is_dir()]
        if len(dirs) != 1:
            return [f"{crate.name} unpacked to {len(dirs)} directories"]
        pkg = dirs[0]
        with open(pkg / "Cargo.toml", "a", encoding="utf-8") as f:
            f.write("\n[workspace]\n")
        rc, stdout, stderr = runner(["cargo", "check", "--locked", *shlex.split(opts.locked_check_args), *patches], pkg)
        out = stdout + stderr
        if rc != 0:
            return [f"the unpacked crate does not build with its shipped lock:\n{out}"]
        if opts.doctests_min:
            rc, stdout, stderr = runner(["cargo", "test", "--doc", "--locked", *shlex.split(opts.doctest_args), *patches], pkg)
            out = stdout + stderr
            passed, failed = passed_tests(out)
            if rc != 0 or failed:
                return [f"the unpacked crate's doctests fail:\n{out}"]
            if passed < opts.doctests_min:
                return [f"the unpacked crate ran {passed} doctests, expected at least {opts.doctests_min}"]
            print(f"package_integrity: unpacked doctests: {passed} passed")
        for spec in opts.test_in_package:
            rc, stdout, stderr = runner(["cargo", "test", "--locked", *shlex.split(spec), *patches], pkg)
            out = stdout + stderr
            passed, failed = passed_tests(out)
            if rc != 0 or failed:
                return [f"`cargo test {spec}` fails in the unpacked crate:\n{out}"]
            if passed == 0:
                return [f"`cargo test {spec}` ran no test in the unpacked crate (compared nothing)"]
            print(f"package_integrity: unpacked `cargo test {spec}`: {passed} passed")
        (pkg / "Cargo.lock").unlink(missing_ok=True)
        rc, stdout, stderr = runner(["cargo", "check", *shlex.split(opts.check_args), *patches], pkg)
        out = stdout + stderr
        if rc != 0:
            return [f"the unpacked crate does not build with a fresh resolution:\n{out}"]
        lock = (pkg / "Cargo.lock").read_text(encoding="utf-8") if (pkg / "Cargo.lock").exists() else ""
        missing = [n for n in opts.expect_in_lock if f'name = "{n}"' not in lock]
        if missing:
            return [f"the fresh resolution did not resolve {', '.join(missing)}"]
    return []


def main(argv: list[str] | None = None, versions_of=registry_versions, runner: Runner = run_cmd) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="the repository (the git top-level)")
    ap.add_argument("--member", default=".",
                    help="the packaged crate's directory relative to --root (a workspace member)")
    ap.add_argument("--publish-gate", action="store_true")
    ap.add_argument("--patches", action="store_true",
                    help="print the blocked dependencies' patch arguments, one per line, and exit")
    ap.add_argument("--expect-in-lock", action="append", default=[],
                    help="a package the fresh resolution must contain (repeatable)")
    ap.add_argument("--locked-check-args", default="--all-features",
                    help="arguments of the shipped-lock `cargo check --locked` (default --all-features)")
    ap.add_argument("--check-args", default="--all-features",
                    help="arguments of the fresh-resolution `cargo check` (default --all-features)")
    ap.add_argument("--require-file", action="append", default=[],
                    help="a file `cargo package --list` must contain (repeatable)")
    ap.add_argument("--forbid-prefix", action="append", default=[],
                    help="a path prefix no packaged file may start with (repeatable)")
    ap.add_argument("--doctests-min", type=int, default=0,
                    help="run `cargo test --doc` in the unpacked crate and require at least N to pass")
    ap.add_argument("--doctest-args", default="", help="extra arguments of that doctest run")
    ap.add_argument("--test-in-package", action="append", default=[],
                    help="`cargo test` arguments to run in the unpacked crate; at least one test must pass (repeatable)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    pkg_dir = (root / args.member).resolve()
    if not (pkg_dir / "Cargo.toml").is_file() or "package" not in tomllib.loads(
            (pkg_dir / "Cargo.toml").read_text(encoding="utf-8")):
        print(f"package_integrity: {pkg_dir / 'Cargo.toml'} has no [package] (use --member for a workspace)",
              file=sys.stderr if args.patches else sys.stdout)
        return 1
    out = sys.stderr if args.patches else sys.stdout
    try:
        blocked, problems = classify(root, versions_of, args.member)
    except OSError as e:  # the registry index could not be read: never a pass
        print(f"package_integrity: crates.io index unreadable: {e}", file=out)
        return 1
    for p in problems:
        print(f"package_integrity: {p}", file=out)
    if problems:
        return 1
    if args.patches:
        for b in blocked:
            print(f"--config=patch.crates-io.{b['name']}.path={json.dumps(b['path'])}")
        print(f"package_integrity: blocked {len(blocked)}", file=sys.stderr)
        return 0
    report_blocked(blocked)
    if args.publish_gate:
        if blocked:
            print("package_integrity: publish gate: FAIL (a dependency is satisfied only by an unpublished sibling)")
            return 1
        print("package_integrity: publish gate: OK (nothing blocked)")
        return 0
    # a member's package lands in the workspace's target directory
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
    errors = package_and_build(pkg_dir, blocked, args, runner, target)
    for e in errors:
        print(f"package_integrity: {e}")
    if errors:
        return 1
    print(f"package_integrity: OK (packages inspected 1, blocked {len(blocked)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

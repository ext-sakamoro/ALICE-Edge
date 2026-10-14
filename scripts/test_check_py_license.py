#!/usr/bin/env python3
"""Tests of scripts/check_py_license.py: equal licenses pass, a drift fails, and a
missing or unreadable license on either side fails (0 compared is not a pass)."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check_py_license", HERE / "check_py_license.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def run(cargo: str, pyproject: str) -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "Cargo.toml").write_text(cargo, encoding="utf-8")
        (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        mod.ROOT = root
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return mod.main()


CARGO = '[package]\nname = "x"\nlicense = "MIT OR Apache-2.0"\n'
failures = []


def check(name: str, got: int, want: int) -> None:
    if got != want:
        failures.append(f"{name}: exit {got}, expected {want}")


check("equal", run(CARGO, '[project]\nname = "x"\nlicense = "MIT OR Apache-2.0"\n'), 0)
check("drift (one side of the OR dropped)", run(CARGO, '[project]\nlicense = "MIT"\n'), 1)
check("table form is not compared", run(CARGO, '[project]\nlicense = { text = "MIT OR Apache-2.0" }\n'), 1)
check("no license in pyproject", run(CARGO, '[project]\nname = "x"\n'), 1)
check("no license in Cargo.toml", run('[package]\nname = "x"\n', '[project]\nlicense = "MIT OR Apache-2.0"\n'), 1)
check("license under another section", run(CARGO, '[tool.x]\nlicense = "MIT OR Apache-2.0"\n'), 1)

if failures:
    print("\n".join(failures))
    sys.exit(1)
print("check_py_license: 6 of 6 cases as expected")

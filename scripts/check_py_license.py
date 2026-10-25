#!/usr/bin/env python3
"""The Python package license metadata must equal the crate license.

The crate's declared license (`Cargo.toml` `[package] license`) is the source of
truth; the wheel must not drift from it (it once shipped a license string different
from the crate's). Fails if the two differ, and fails
if it cannot compare them (0 compared is not a pass).

usage: scripts/check_py_license.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def license_in(path: Path, section: str) -> str | None:
    """the SPDX string of `license = "…"` inside [section] (PEP 639 / Cargo [package])"""
    in_section = False
    # encoding is explicit: read_text() defaults to the locale encoding, which is
    # cp1252 on the Windows CI runner and fails on any non-ASCII byte in the file
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_section = s == f"[{section}]"
            continue
        if in_section:
            m = re.match(r'license\s*=\s*"([^"]+)"\s*$', s)
            if m:
                return m.group(1)
    return None


def main() -> int:
    cargo = license_in(ROOT / "Cargo.toml", "package")
    pyproject = license_in(ROOT / "pyproject.toml", "project")
    compared = 0
    if cargo is not None and pyproject is not None:
        compared += 1
        if cargo != pyproject:
            sys.stderr.write(
                f"license drift: Cargo.toml {cargo!r} != pyproject.toml {pyproject!r}\n"
                "  the crate license is the source of truth; set pyproject [project] license to it\n"
            )
            return 1
    if compared == 0:
        sys.stderr.write(
            f"no license compared (Cargo.toml={cargo!r}, pyproject.toml={pyproject!r}); "
            "expected a quoted SPDX `license = \"…\"` in [package] and [project]\n"
        )
        return 1
    print(f"license OK: Cargo.toml and pyproject.toml both {cargo!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

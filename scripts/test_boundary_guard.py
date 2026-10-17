#!/usr/bin/env python3
"""Tests of scripts/boundary_guard.py on throwaway trees."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boundary_guard  # noqa: E402

FFI = 'pub extern "C" fn as_x_new(id: *const u8) -> i32 { 0 }\npub extern "C" fn as_x_version() -> u32 { 1 }\n'
PY = ("#[pymethods]\nimpl PyThing {\n    #[new]\n    fn new() -> Self { Self {} }\n"
      "    fn get(&self) -> u8 {\n        if true { 1 } else { 0 }\n    }\n}\n"
      "fn helper() {}\n#[pyfunction]\nfn version() -> u32 { 1 }\n")
DEC = "pub struct Msg;\nimpl Msg {\n    pub fn from_bytes(b: &[u8]) -> Option<Self> { None }\n}\n"
TEST = "#[test]\nfn x_new_refuses_another_identifier() {}\n"
GOOD = {
    "ffi::as_x_new": "handshake: x_new_refuses_another_identifier",
    "ffi::as_x_version": "local: reports a constant of this build",
    "python::PyThing.new": "local: builds a local value from integers",
    "python::PyThing.get": "local: reads a local value of this process",
    "python::version": "local: reports a constant of this build",
    "decode::msg.rs::Msg::from_bytes": "local: framing integers, no Law value",
}


class BoundaryGuardTest(unittest.TestCase):
    def tree(self, listed: dict[str, str], files: dict[str, str] | None = None) -> Path:
        d = Path(tempfile.mkdtemp())
        (d / "src").mkdir()
        (d / "tests").mkdir()
        (d / "scripts").mkdir()
        files = {"src/ffi.rs": FFI, "src/python.rs": PY, "src/msg.rs": DEC, "tests/t.rs": TEST} \
            if files is None else files
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8")
        body = "\n".join(f'"{k}" = "{v}"' for k, v in listed.items())
        (d / "scripts" / "boundaries.toml").write_text("[boundary]\n" + body + "\n", encoding="utf-8")
        return d

    def test_every_entry_point_decided_passes(self):
        self.assertEqual(boundary_guard.check(self.tree(GOOD)), [])

    def test_a_new_ffi_entry_point_without_a_decision_fails(self):
        files = {"src/ffi.rs": FFI + 'pub extern "C" fn as_x_ingest(p: *const u8) {}\n',
                 "src/python.rs": PY, "src/msg.rs": DEC, "tests/t.rs": TEST}
        problems = boundary_guard.check(self.tree(GOOD, files))
        self.assertTrue(any("ffi::as_x_ingest" in p for p in problems), problems)

    def test_python_methods_and_functions_are_found_and_helpers_are_not(self):
        listed = {k: v for k, v in GOOD.items() if not k.startswith("python::")}
        problems = boundary_guard.check(self.tree(listed))
        for key in ("python::PyThing.new", "python::PyThing.get", "python::version"):
            self.assertTrue(any(key in p for p in problems), (key, problems))
        self.assertFalse(any("helper" in p for p in problems), problems)

    def test_a_decoder_without_a_decision_fails(self):
        listed = {k: v for k, v in GOOD.items() if not k.startswith("decode::")}
        problems = boundary_guard.check(self.tree(listed))
        self.assertTrue(any("decode::msg.rs::Msg::from_bytes" in p for p in problems), problems)

    def test_a_handshake_without_its_test_fails(self):
        files = {"src/ffi.rs": FFI, "src/python.rs": PY, "src/msg.rs": DEC, "tests/t.rs": "\n"}
        problems = boundary_guard.check(self.tree(GOOD, files))
        self.assertTrue(any("x_new_refuses_another_identifier" in p for p in problems), problems)

    def test_stale_entry_bad_prefix_and_short_reason_fail(self):
        listed = dict(GOOD)
        listed["ffi::as_x_gone"] = "local: removed some time ago"
        listed["ffi::as_x_version"] = "fine"
        listed["python::version"] = "local: short"
        problems = boundary_guard.check(self.tree(listed))
        self.assertTrue(any("as_x_gone" in p and "stale" in p for p in problems), problems)
        self.assertTrue(any("as_x_version" in p and "must start" in p for p in problems), problems)
        self.assertTrue(any("python::version" in p and "shorter" in p for p in problems), problems)

    def test_no_entry_point_fails(self):
        problems = boundary_guard.check(self.tree({}, {"src/lib.rs": "pub fn f() {}\n"}))
        self.assertTrue(any("compared nothing" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()

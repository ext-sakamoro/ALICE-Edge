#!/usr/bin/env python3
"""Tests for scripts/ffi_status_check.py.

The first case runs the check against this repository. The other cases copy
the three real files into a temporary tree, break exactly one thing, and
assert that the check reports it.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ffi_status_check as fsc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FfiStatusCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        for rel in (fsc.RUST_FILE, fsc.C_FILE, fsc.CS_FILE):
            os.makedirs(os.path.join(self.tmp, os.path.dirname(rel)), exist_ok=True)
            shutil.copyfile(os.path.join(ROOT, rel), os.path.join(self.tmp, rel))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def edit(self, rel, old, new):
        path = os.path.join(self.tmp, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(text.count(old), 1, f"{old!r} must occur once in {rel}")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace(old, new))

    def assertFinding(self, needle):
        errors, _ = fsc.check(self.tmp)
        self.assertTrue(any(needle in e for e in errors), errors)

    def test_repository_is_consistent(self):
        errors, counts = fsc.check(ROOT)
        self.assertEqual(errors, [])
        self.assertEqual(counts, {"rust": 5, "c": 5, "cs": 5, "compared": 5})

    def test_copy_is_consistent(self):
        errors, counts = fsc.check(self.tmp)
        self.assertEqual(errors, [])
        self.assertEqual(counts["compared"], 5)

    def test_swap_in_rust(self):
        self.edit(fsc.RUST_FILE, "ALICE_EDGE_ERR_NULL_POINTER: i32 = -1;", "ALICE_EDGE_ERR_NULL_POINTER: i32 = -2;")
        self.edit(fsc.RUST_FILE, "ALICE_EDGE_ERR_NON_POSITIVE_K: i32 = -2;", "ALICE_EDGE_ERR_NON_POSITIVE_K: i32 = -1;")
        self.assertFinding("ALICE_EDGE_ERR_NULL_POINTER differs")
        self.assertFinding("ALICE_EDGE_ERR_NON_POSITIVE_K differs")

    def test_swap_in_c(self):
        self.edit(fsc.C_FILE, "ALICE_EDGE_ERR_NON_POSITIVE_K (-2)", "ALICE_EDGE_ERR_NON_POSITIVE_K (-3)")
        self.edit(fsc.C_FILE, "ALICE_EDGE_ERR_PANIC          (-3)", "ALICE_EDGE_ERR_PANIC          (-2)")
        self.assertFinding("ALICE_EDGE_ERR_PANIC differs")

    def test_swap_in_cs(self):
        self.edit(fsc.CS_FILE, "public const int Ok = 0;", "public const int Ok = -3;")
        self.edit(fsc.CS_FILE, "public const int Panic = -3;", "public const int Panic = 0;")
        self.assertFinding("ALICE_EDGE_OK differs")
        self.assertFinding("ALICE_EDGE_ERR_PANIC differs")

    def test_missing_entry_in_cs(self):
        self.edit(fsc.CS_FILE, "        public const int Panic = -3;\n", "")
        self.assertFinding("found 4 status codes, expected 5")
        self.assertFinding("AliceEdge.cs: ALICE_EDGE_ERR_PANIC is missing")
        self.assertFinding("compared 4 codes")

    def test_missing_entry_in_c(self):
        self.edit(fsc.C_FILE, "#define ALICE_EDGE_ERR_NULL_POINTER   (-1) // a required pointer was null\n", "")
        self.assertFinding("alice_edge.h: ALICE_EDGE_ERR_NULL_POINTER is missing")

    def test_extra_entry_in_rust(self):
        self.edit(fsc.RUST_FILE, "pub const ALICE_EDGE_OK: i32 = 0;", "pub const ALICE_EDGE_OK: i32 = 0;\npub const ALICE_EDGE_ERR_OTHER: i32 = -5;")
        self.assertFinding("found 6 status codes, expected 5")

    def test_swap_window_too_long_in_rust(self):
        self.edit(fsc.RUST_FILE, "ALICE_EDGE_ERR_PANIC: i32 = -3;", "ALICE_EDGE_ERR_PANIC: i32 = -4;")
        self.edit(fsc.RUST_FILE, "ALICE_EDGE_ERR_WINDOW_TOO_LONG: i32 = -4;", "ALICE_EDGE_ERR_WINDOW_TOO_LONG: i32 = -3;")
        self.assertFinding("ALICE_EDGE_ERR_PANIC differs")
        self.assertFinding("ALICE_EDGE_ERR_WINDOW_TOO_LONG differs")

    def test_swap_window_too_long_in_c(self):
        self.edit(fsc.C_FILE, "ALICE_EDGE_ERR_NULL_POINTER   (-1)", "ALICE_EDGE_ERR_NULL_POINTER   (-4)")
        self.edit(fsc.C_FILE, "ALICE_EDGE_ERR_WINDOW_TOO_LONG (-4)", "ALICE_EDGE_ERR_WINDOW_TOO_LONG (-1)")
        self.assertFinding("ALICE_EDGE_ERR_NULL_POINTER differs")
        self.assertFinding("ALICE_EDGE_ERR_WINDOW_TOO_LONG differs")

    def test_swap_window_too_long_in_cs(self):
        self.edit(fsc.CS_FILE, "public const int NonPositiveK = -2;", "public const int NonPositiveK = -4;")
        self.edit(fsc.CS_FILE, "public const int WindowTooLong = -4;", "public const int WindowTooLong = -2;")
        self.assertFinding("ALICE_EDGE_ERR_NON_POSITIVE_K differs")
        self.assertFinding("ALICE_EDGE_ERR_WINDOW_TOO_LONG differs")

    def test_missing_window_too_long_in_each_file(self):
        removals = (
            (fsc.RUST_FILE, "pub const ALICE_EDGE_ERR_WINDOW_TOO_LONG: i32 = -4;"),
            (fsc.C_FILE, "#define ALICE_EDGE_ERR_WINDOW_TOO_LONG (-4) // len > ALICE_EDGE_ROBUST_MAX_N"),
            (fsc.CS_FILE, "public const int WindowTooLong = -4;"),
        )
        for rel, line in removals:
            with self.subTest(rel=rel):
                # a fresh copy of the three files for each removal
                self.tearDown()
                self.setUp()
                self.edit(rel, line, "")
                self.assertFinding(f"{rel}: found 4 status codes, expected 5")
                self.assertFinding(f"{rel}: ALICE_EDGE_ERR_WINDOW_TOO_LONG is missing")
                self.assertFinding("compared 4 codes")

    def test_limits_class_is_not_a_status(self):
        # AliceLimits.RobustMaxN and ALICE_EDGE_ROBUST_MAX_N sit next to the
        # status codes but are not status codes
        errors, counts = fsc.check(self.tmp)
        self.assertEqual(errors, [])
        self.assertEqual(counts["cs"], 5)
        self.assertEqual(counts["c"], 5)

    def test_renamed_class_compares_nothing(self):
        self.edit(fsc.CS_FILE, "class AliceStatus", "class AliceStatusCodes")
        self.assertFinding("class AliceStatus not found")
        self.assertFinding("found 0 status codes")
        self.assertFinding("compared 0 codes")

    def test_name_mapping(self):
        self.assertEqual(fsc.cs_to_c("Ok"), "OK")
        self.assertEqual(fsc.cs_to_c("NonPositiveK"), "ERR_NON_POSITIVE_K")
        self.assertEqual(fsc.cs_to_c("NullPointer"), "ERR_NULL_POINTER")
        self.assertEqual(fsc.cs_to_c("WindowTooLong"), "ERR_WINDOW_TOO_LONG")


if __name__ == "__main__":
    unittest.main()

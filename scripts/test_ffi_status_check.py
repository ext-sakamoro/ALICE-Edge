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
        self.assertEqual(counts, {"rust": 4, "c": 4, "cs": 4, "compared": 4})

    def test_copy_is_consistent(self):
        errors, counts = fsc.check(self.tmp)
        self.assertEqual(errors, [])
        self.assertEqual(counts["compared"], 4)

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
        self.assertFinding("found 3 status codes, expected 4")
        self.assertFinding("AliceEdge.cs: ALICE_EDGE_ERR_PANIC is missing")
        self.assertFinding("compared 3 codes")

    def test_missing_entry_in_c(self):
        self.edit(fsc.C_FILE, "#define ALICE_EDGE_ERR_NULL_POINTER   (-1) // a required pointer was null\n", "")
        self.assertFinding("alice_edge.h: ALICE_EDGE_ERR_NULL_POINTER is missing")

    def test_extra_entry_in_rust(self):
        self.edit(fsc.RUST_FILE, "pub const ALICE_EDGE_OK: i32 = 0;", "pub const ALICE_EDGE_OK: i32 = 0;\npub const ALICE_EDGE_ERR_OTHER: i32 = -4;")
        self.assertFinding("found 5 status codes, expected 4")

    def test_renamed_class_compares_nothing(self):
        self.edit(fsc.CS_FILE, "class AliceStatus", "class AliceStatusCodes")
        self.assertFinding("class AliceStatus not found")
        self.assertFinding("found 0 status codes")
        self.assertFinding("compared 0 codes")

    def test_name_mapping(self):
        self.assertEqual(fsc.cs_to_c("Ok"), "OK")
        self.assertEqual(fsc.cs_to_c("NonPositiveK"), "ERR_NON_POSITIVE_K")
        self.assertEqual(fsc.cs_to_c("NullPointer"), "ERR_NULL_POINTER")


if __name__ == "__main__":
    unittest.main()

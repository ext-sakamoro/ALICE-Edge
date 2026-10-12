#!/usr/bin/env python3
"""Tests for scripts/package_contents_check.py.

The first case runs the check against this repository. The other cases feed
`check` a listing with exactly one thing broken and assert that it is reported.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import package_contents_check as pcc  # noqa: E402

GOOD = [
    "Cargo.toml",
    "Cargo.toml.orig",
    "README.md",
    "LICENSE",
    "LICENSE-APACHE",
    "bindings/alice_edge.h",
    "bindings/AliceEdge.cs",
    "src/lib.rs",
    "tests/golden.rs",
    "tests/data/table.bin",
]
TARGETS = ["src/lib.rs", "tests/golden.rs"]
SOURCES = {"src/lib.rs": "", "tests/golden.rs": 'const T: &[u8] = include_bytes!("data/table.bin");'}


class PackageContentsCheckTest(unittest.TestCase):
    def test_this_repository_passes(self):
        self.assertEqual(pcc.main(), 0)

    def test_a_good_listing_passes(self):
        self.assertEqual(pcc.check(GOOD, TARGETS, SOURCES), ([], 2))

    def test_forbidden_paths_fail(self):
        for path in (".github/workflows/ci.yml", "scripts/preflight.sh", "fuzz/Cargo.toml",
                     ".cargo/config.toml", "deny.toml", "clippy.toml", "rust-toolchain.toml",
                     ".gitignore", "pyproject.toml", "CONTRIBUTING.md"):
            errors, _ = pcc.check(GOOD + [path], TARGETS, SOURCES)
            self.assertEqual(len(errors), 1, path)
            self.assertIn("forbidden", errors[0])

    def test_a_missing_required_file_fails(self):
        for path in pcc.REQUIRED_FILES:
            errors, _ = pcc.check([p for p in GOOD if p != path], TARGETS, SOURCES)
            self.assertTrue(any("required file" in e and path in e for e in errors), path)

    def test_a_missing_target_fails(self):
        errors, _ = pcc.check(GOOD, TARGETS + ["examples/demo.rs"], SOURCES)
        self.assertEqual(errors, ["examples/demo.rs: Cargo target missing from the package"])

    def test_a_missing_embedded_file_fails(self):
        listing = [p for p in GOOD if p != "tests/data/table.bin"]
        errors, _ = pcc.check(listing, TARGETS, SOURCES)
        self.assertEqual(errors, ["tests/golden.rs: embeds tests/data/table.bin, which is not in the package"])

    def test_an_embed_outside_the_package_fails(self):
        sources = {"src/lib.rs": '#![doc = include_str!("../../README.md")]'}
        errors, _ = pcc.check(GOOD, TARGETS, sources)
        self.assertEqual(len(errors), 1)
        self.assertIn("../README.md", errors[0])

    def test_an_empty_listing_fails(self):
        self.assertEqual(pcc.check([], TARGETS, SOURCES)[0], ["the package list is empty"])

    def test_no_target_compared_fails(self):
        errors, compared = pcc.check(GOOD, [], SOURCES)
        self.assertEqual((errors, compared), (["no Cargo target was compared"], 0))


if __name__ == "__main__":
    unittest.main()

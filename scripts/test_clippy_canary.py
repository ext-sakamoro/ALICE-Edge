#!/usr/bin/env python3
"""Tests for scripts/clippy_canary.py without running cargo: the data file and
clippy.toml parsers, the per-row diagnostic accounting, the control function
and the compared-nothing failures."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clippy_canary as cc  # noqa: E402

TOML = """# comment ]
disallowed-methods = [
    { path = "f64::sin", reason = "r" },
    # a commented entry { path = "f64::cos" }
    { path = "std::time::Instant::now", reason = "r" },
]
disallowed-types = [
    { path = "std::collections::hash_map::RandomState", reason = "r" },
]
"""
DATA = ("# header\n\n"
        "disallowed-methods\tf64::sin\tlet _ = 0.5_f64.sin();\n"
        "disallowed-types\tstd::collections::hash_map::RandomState\tlet _ = RandomState::default();\n")


def diag(lint, line, path):
    kind = "method" if lint.endswith("methods") else "type"
    return (lint, line, f"use of a disallowed {kind} `{path}`")


class Parsers(unittest.TestCase):
    def test_rows_skip_comments_and_blanks(self):
        rows, problems = cc.parse_rows(DATA)
        self.assertEqual(problems, [])
        self.assertEqual([(r.key, r.path) for r in rows],
                         [("disallowed-methods", "f64::sin"),
                          ("disallowed-types", "std::collections::hash_map::RandomState")])
        self.assertEqual(rows[0].line, 3)

    def test_malformed_rows_are_reported(self):
        rows, problems = cc.parse_rows("disallowed-methods\tf64::sin\n"
                                       "disallowed-things\tx\tlet _ = 1;\n")
        self.assertEqual(rows, [])
        self.assertEqual(len(problems), 2, problems)
        self.assertIn("3 tab-separated", problems[0])
        self.assertIn("unknown key", problems[1])

    def test_clippy_toml_entries_by_key(self):
        found = cc.clippy_toml_paths(TOML)
        self.assertEqual(found["disallowed-methods"], {"f64::sin", "std::time::Instant::now"})
        self.assertEqual(found["disallowed-types"], {"std::collections::hash_map::RandomState"})

    def test_lints_tables_are_copied(self):
        cargo = '[package]\nname = "x"\n\n[lints.clippy]\na = "allow"\n\n[dev-dependencies]\nb = "1"\n'
        self.assertEqual(cc.lints_tables(cargo), '[lints.clippy]\na = "allow"\n\n')
        self.assertEqual(cc.lints_tables('[package]\nname = "x"\n'), "")

    def test_render_gives_each_row_its_own_lines(self):
        rows, _ = cc.parse_rows(DATA)
        lib, ranges = cc.render_lib(rows)
        lines = lib.splitlines()
        for i, row in enumerate(rows):
            lo, hi = ranges[f"canary_{i}"]
            self.assertIn(row.stmt, "\n".join(lines[lo - 1:hi]))
        self.assertIn(cc.CONTROL, ranges)

    def test_json_diagnostics_are_filtered_to_disallowed(self):
        out = "\n".join([
            "not json",
            '{"reason":"compiler-artifact"}',
            '{"reason":"compiler-message","message":{"code":{"code":"clippy::disallowed_methods"},'
            '"message":"use of a disallowed method `f64::sin`","spans":[{"is_primary":true,"line_start":3}]}}',
            '{"reason":"compiler-message","message":{"code":{"code":"unused_variables"},'
            '"message":"x","spans":[{"is_primary":true,"line_start":9}]}}',
        ])
        self.assertEqual(cc.disallowed_diagnostics(out),
                         [("clippy::disallowed_methods", 3, "use of a disallowed method `f64::sin`")])


class Check(unittest.TestCase):
    def setUp(self):
        self.rows, _ = cc.parse_rows(DATA)
        self.toml = cc.clippy_toml_paths(TOML)
        _, self.ranges = cc.render_lib(self.rows)
        self.good = [diag("clippy::disallowed_methods", self.ranges["canary_0"][0] + 1, "f64::sin"),
                     diag("clippy::disallowed_types", self.ranges["canary_1"][0] + 1,
                          "std::collections::hash_map::RandomState")]

    def run_check(self, diags=None, code=101, rows=None, toml=None):
        return cc.check(self.rows if rows is None else rows, self.toml if toml is None else toml,
                        code, self.good if diags is None else diags, self.ranges)

    def test_one_diagnostic_per_row_passes(self):
        self.assertEqual(self.run_check(), [])

    def test_a_row_without_its_diagnostic_fails(self):
        p = self.run_check(diags=self.good[:1])
        self.assertEqual(len(p), 1, p)
        self.assertIn("RandomState` got 0", p[0])

    def test_a_diagnostic_naming_another_path_does_not_count(self):
        wrong = [self.good[0], diag("clippy::disallowed_types", self.good[1][1], "other::Type")]
        p = self.run_check(diags=wrong)
        self.assertTrue(any("1 disallowed diagnostic(s), 0 naming it" in x for x in p), p)

    def test_two_diagnostics_in_one_row_fail(self):
        p = self.run_check(diags=self.good + [self.good[0]])
        self.assertTrue(any("f64::sin` got 2" in x for x in p), p)

    def test_clippy_exiting_zero_fails(self):
        p = self.run_check(code=0)
        self.assertTrue(any("exited 0" in x for x in p), p)

    def test_a_diagnostic_in_the_control_fails(self):
        ctl = diag("clippy::disallowed_methods", self.ranges[cc.CONTROL][0] + 1, "f64::sqrt")
        p = self.run_check(diags=self.good + [ctl])
        self.assertTrue(any("control" in x for x in p), p)

    def test_a_stray_diagnostic_fails(self):
        p = self.run_check(diags=self.good + [diag("clippy::disallowed_methods", 1, "f64::cos")])
        self.assertTrue(any("outside any canary" in x for x in p), p)

    def test_a_row_missing_from_clippy_toml_fails_naming_it(self):
        toml = cc.clippy_toml_paths(TOML.replace('{ path = "f64::sin", reason = "r" },', ""))
        p = self.run_check(toml=toml)
        self.assertEqual(len(p), 1, p)
        self.assertIn("`f64::sin` is not listed under disallowed-methods", p[0])

    def test_no_rows_compares_nothing(self):
        p = cc.check([], self.toml, 1, [], {})
        self.assertEqual(p, ["compared nothing: the data file has no rows"])

    def test_empty_clippy_toml_compares_nothing(self):
        p = cc.check(self.rows, cc.clippy_toml_paths(""), 1, [], {})
        self.assertEqual(p, ["compared nothing: clippy.toml has no disallowed-* entries"])


class Main(unittest.TestCase):
    def test_an_empty_data_file_fails_before_cargo(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "calls.tsv"
            f.write_text("# only a comment\n", encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cc.main(["--calls", str(f)])
        self.assertEqual(rc, 1)
        self.assertIn("compared nothing: the data file has no rows", err.getvalue())

    def test_the_repository_rows_are_all_in_clippy_toml(self):
        rows, problems = cc.parse_rows(cc.CALLS.read_text(encoding="utf-8"))
        self.assertEqual(problems, [])
        self.assertGreater(len(rows), 0)
        toml = cc.clippy_toml_paths((cc.ROOT / "clippy.toml").read_text(encoding="utf-8"))
        for row in rows:
            self.assertIn(row.path, toml[row.key], row)


if __name__ == "__main__":
    unittest.main()

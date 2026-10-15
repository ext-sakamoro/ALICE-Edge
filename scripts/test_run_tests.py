#!/usr/bin/env python3
"""Tests for scripts/run_tests.sh: a cargo-shaped log is replayed through the
script with `cat`, and the exit code and messages are checked for a passing
run, a run in which no test ran, a named file that ran none (or was not built
at all), a failing command, and Windows output (backslash paths, CRLF, ANSI
colour). The script itself is shared byte for byte with ALICE-DB and is not
modified here."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_tests.sh"


def find_bash():
    """Git's bash on Windows (CreateProcess and often PATH find System32's WSL
    launcher first), the first `bash` on PATH elsewhere."""
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            for cand in (Path(git).parents[1] / "bin" / "bash.exe", Path(git).parents[1] / "usr" / "bin" / "bash.exe"):
                if cand.exists():
                    return str(cand)
        found = shutil.which("bash")
        return None if found is None or "system32" in found.lower() else found
    return shutil.which("bash")


BASH = find_bash()


def running(name, sep="/"):
    return f"     Running tests{sep}{name}.rs (target/debug/deps/{name}-0123abcd)"


def result(n):
    return f"test result: ok. {n} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"


UNIT = ["     Running unittests src/lib.rs (target/debug/deps/alice_analytics-0123abcd)", result(7)]


class RunTests(unittest.TestCase):
    def setUp(self):
        # not a skip: a run without bash would otherwise report success
        self.assertIsNotNone(BASH, "bash not found: run_tests.sh cannot be exercised")

    def run_log(self, required, lines, *, newline="\n", status=0):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "cargo.log"
            log.write_bytes(newline.join(lines + [""]).encode("utf-8"))
            cmd = f'cat "{log.as_posix()}"; exit {status}'
            p = subprocess.run([BASH, SCRIPT.as_posix(), required, "--", "bash", "-c", cmd],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout, p.stderr

    def test_a_passing_run_with_every_named_file(self):
        rc, out, err = self.run_log("alpha,beta", UNIT + [running("alpha"), result(3), running("beta"), result(2)])
        self.assertEqual(rc, 0, err)
        self.assertIn("run_tests: 12 tests passed in total, 3 in alpha, 2 in beta", out)
        self.assertEqual(err, "")

    def test_no_test_ran_fails(self):
        rc, out, err = self.run_log("alpha", ["     Running unittests src/lib.rs (x)", result(0)])
        self.assertEqual(rc, 1)
        self.assertIn("run_tests: 0 tests passed in total", out)
        self.assertIn("run_tests: no test ran", err)

    def test_an_empty_log_fails(self):
        rc, _, err = self.run_log("alpha", [])
        self.assertEqual(rc, 1)
        self.assertIn("no test ran", err)

    def test_a_named_file_that_ran_zero_fails(self):
        rc, out, err = self.run_log("alpha,beta", UNIT + [running("alpha"), result(3), running("beta"), result(0)])
        self.assertEqual(rc, 1)
        self.assertIn("0 in beta", out)
        self.assertIn("run_tests: beta did not run any test", err)
        self.assertNotIn("alpha did not", err)

    def test_a_named_file_that_was_not_built_fails(self):
        rc, _, err = self.run_log("alpha,beta", UNIT + [running("alpha"), result(3)])
        self.assertEqual(rc, 1)
        self.assertIn("run_tests: beta did not run any test", err)

    def test_a_similar_name_does_not_count(self):
        # `tests/alpha_extra.rs` is not `tests/alpha.rs`
        rc, _, err = self.run_log("alpha", UNIT + [running("alpha_extra"), result(4)])
        self.assertEqual(rc, 1)
        self.assertIn("alpha did not run any test", err)

    def test_a_failing_command_fails(self):
        rc, _, _ = self.run_log("alpha", UNIT + [running("alpha"), result(3)], status=101)
        self.assertNotEqual(rc, 0)

    def test_windows_output_is_counted(self):
        esc = "\x1b"
        lines = UNIT + [f"     {esc}[1m{esc}[32mRunning{esc}[0m tests\\alpha.rs (target\\debug\\deps\\alpha-0.exe)",
                        result(5)]
        rc, out, err = self.run_log("alpha", lines, newline="\r\n")
        self.assertEqual(rc, 0, err)
        self.assertIn("12 tests passed in total, 5 in alpha", out)


if __name__ == "__main__":
    unittest.main()

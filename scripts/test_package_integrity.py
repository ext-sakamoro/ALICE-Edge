#!/usr/bin/env python3
"""Tests for scripts/package_integrity.py (run: python3 scripts/test_package_integrity.py).

The registry and cargo are injected: `versions_of` stands for the crates.io index and the
runner for cargo, so each case fixes exactly what is published and what cargo reports.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_integrity as pi  # noqa: E402

SHA = "72cd19faf37760e228ba6155fd3bdb5a0a85bcfa"


class Requirements(unittest.TestCase):
    def test_range_and_caret_forms(self) -> None:
        self.assertTrue(pi.satisfies((0, 11, 0), ">=0.11, <0.12"))
        self.assertFalse(pi.satisfies((0, 10, 0), ">=0.11, <0.12"))
        self.assertFalse(pi.satisfies((0, 12, 0), ">=0.11, <0.12"))
        self.assertTrue(pi.satisfies((0, 10, 3), "0.10"))
        self.assertFalse(pi.satisfies((0, 11, 0), "0.10"))
        self.assertTrue(pi.satisfies((7, 4, 0), "7"))
        self.assertFalse(pi.satisfies((8, 0, 0), ">=7, <8"))
        self.assertTrue(pi.satisfies((0, 0, 3), "^0.0.3"))
        self.assertFalse(pi.satisfies((0, 0, 4), "^0.0.3"))
        self.assertTrue(pi.satisfies((1, 2, 9), "~1.2"))
        self.assertFalse(pi.satisfies((1, 3, 0), "~1.2"))
        self.assertTrue(pi.satisfies((0, 5, 1), "=0.5.1"))
        self.assertFalse(pi.satisfies((0, 5, 2), "=0.5.1"))

    def test_unsupported_syntax_is_a_problem(self) -> None:
        for bad in ("0.11-beta", ">= ", "0.11,", "latest"):
            with self.assertRaises(pi.Problem, msg=bad):
                pi.comparators(bad)


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "ALICE-Crate"
        (self.root / ".github").mkdir(parents=True)
        (self.root / ".github" / "alice-siblings").write_text(f"ALICE-Zip {SHA}\n")
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "alice-crate"\nversion = "0.6.0"\n\n[dependencies]\n'
            'alice-zip = { path = "../ALICE-Zip", version = ">=0.11, <0.12", optional = true }\n'
        )
        (base / "ALICE-Zip").mkdir()
        (base / "ALICE-Zip" / "Cargo.toml").write_text('[package]\nname = "alice-zip"\nversion = "0.11.0"\n')
        self.target = base / "target"
        self.calls: list[list[str]] = []
        self.published = {"alice-zip": [(0, 10, 0), (0, 9, 0)]}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def versions_of(self, name: str) -> list:
        return self.published[name]

    def write_crate(self) -> None:
        out = self.target / "package"
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out / "alice-crate-0.6.0.crate", "w:gz") as tar:
            data = b'[package]\nname = "alice-crate"\nversion = "0.6.0"\n'
            info = tarfile.TarInfo("alice-crate-0.6.0/Cargo.toml")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    def runner(
        self,
        package_error: str | None = None,
        build_error: str | None = None,
        files: str = "Cargo.toml\nREADME.md\nsrc/lib.rs\n",
        doc_out: str = "test result: ok. 2 passed; 0 failed; 0 ignored",
        test_out: str = "test result: ok. 3 passed; 0 failed; 0 ignored",
        list_stderr: str = "",
    ):
        def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
            self.calls.append(cmd)
            if cmd[1] == "package" and "--list" in cmd:
                return 0, files, list_stderr
            if cmd[1] == "test":
                out = doc_out if "--doc" in cmd else test_out
                return (1 if " 0 failed" not in out else 0), out, ""
            if cmd[1] == "package":
                if package_error:
                    return 101, "", package_error
                self.write_crate()
                return 0, "", "Packaged"
            if build_error:
                return 101, "", build_error
            if "--locked" not in cmd:
                (cwd / "Cargo.lock").write_text('[[package]]\nname = "alice-det-math"\n')
            return 0, "", "Finished"

        return run

    def main(self, *argv: str, runner=None) -> tuple[int, str]:
        buf = io.StringIO()
        os.environ["CARGO_TARGET_DIR"] = str(self.target)
        with redirect_stdout(buf):
            rc = pi.main(["--root", str(self.root), *argv], versions_of=self.versions_of, runner=runner or self.runner())
        return rc, buf.getvalue()


class PackageIntegrity(Fixture):
    def test_unpublished_sibling_is_blocked_patched_and_green(self) -> None:
        rc, out = self.main("--expect-in-lock", "alice-det-math")
        self.assertEqual(rc, 0, out)
        self.assertIn("blocked: alice-zip >=0.11, <0.12 unpublished (crates.io max 0.10.0)", out)
        self.assertIn("blocked 1", out)
        self.assertIn("packages inspected 1", out)
        self.assertTrue(all(any("patch.crates-io.alice-zip.path=" in a for a in c) for c in self.calls), self.calls)

    def test_non_sibling_packaging_error_stays_red(self) -> None:
        rc, out = self.main(runner=self.runner(package_error="error: failed to read `src/missing.rs`"))
        self.assertEqual(rc, 1, out)
        self.assertIn("cargo package failed", out)
        rc, out = self.main(runner=self.runner(build_error="error[E0432]: unresolved import"))
        self.assertEqual(rc, 1, out)

    def test_sibling_on_the_registry_cannot_hide_a_failure(self) -> None:
        self.published["alice-zip"] = [(0, 11, 0), (0, 10, 0)]
        failure = "failed to select a version for the requirement `alice-zip`"
        rc, out = self.main(runner=self.runner(package_error=failure))
        self.assertEqual(rc, 1, out)
        self.assertIn("blocked 0", out)
        self.assertFalse(any("patch.crates-io" in a for c in self.calls for a in c), self.calls)

    def test_publish_gate_is_red_while_anything_is_blocked(self) -> None:
        rc, out = self.main("--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("publish gate: FAIL", out)
        self.assertEqual(self.calls, [])
        self.published["alice-zip"] = [(0, 11, 0)]
        rc, out = self.main("--publish-gate")
        self.assertEqual(rc, 0, out)
        self.assertIn("publish gate: OK", out)

    def test_no_crate_produced_fails(self) -> None:
        def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
            return 0, "", "Packaged"

        rc, out = self.main(runner=run)
        self.assertEqual(rc, 1, out)
        self.assertIn("inspected nothing", out)

    def test_sibling_version_must_satisfy_the_requirement(self) -> None:
        (self.root.parent / "ALICE-Zip" / "Cargo.toml").write_text('[package]\nname = "alice-zip"\nversion = "0.10.0"\n')
        rc, out = self.main()
        self.assertEqual(rc, 1, out)
        self.assertIn("does not satisfy", out)

    def test_unlisted_sibling_path_fails(self) -> None:
        (self.root / ".github" / "alice-siblings").write_text(f"ALICE-DB {SHA}\n")
        rc, out = self.main()
        self.assertEqual(rc, 1, out)
        self.assertIn("not in .github/alice-siblings", out)

    def test_expected_lock_entry_missing_fails(self) -> None:
        rc, out = self.main("--expect-in-lock", "alice-not-there")
        self.assertEqual(rc, 1, out)
        self.assertIn("did not resolve alice-not-there", out)

    def test_unreadable_index_fails(self) -> None:
        def down(name: str) -> list:
            raise OSError("network down")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pi.main(["--root", str(self.root)], versions_of=down, runner=self.runner())
        self.assertEqual(rc, 1, buf.getvalue())
        self.assertIn("index unreadable", buf.getvalue())



class PackageContents(Fixture):
    def test_required_files_and_prefixes(self) -> None:
        rc, out = self.main("--require-file", "README.md", "--forbid-prefix", ".github/")
        self.assertEqual(rc, 0, out)
        self.assertIn("1 required present, 1 forbidden prefixes absent", out)
        rc, out = self.main("--require-file", "README_JP.md")
        self.assertEqual(rc, 1, out)
        self.assertIn("the package lacks README_JP.md", out)
        rc, out = self.main("--forbid-prefix", "scripts/", runner=self.runner(files="Cargo.toml\nscripts/x.py\n"))
        self.assertEqual(rc, 1, out)
        self.assertIn("forbidden prefix scripts/", out)

    def test_empty_package_list_fails(self) -> None:
        rc, out = self.main("--require-file", "README.md", runner=self.runner(files=""))
        self.assertEqual(rc, 1, out)
        self.assertIn("listed no file", out)

    def test_stderr_lines_are_not_packaged_files(self) -> None:
        # a cargo warning on stderr must not count as a file nor hide an empty list
        warn = "warning: manifest has no documentation\nnote: see more\n"
        rc, out = self.main("--forbid-prefix", ".github/", runner=self.runner(files="", list_stderr=warn))
        self.assertEqual(rc, 1, out)
        self.assertIn("listed no file", out)
        rc, out = self.main("--forbid-prefix", "warning", runner=self.runner(list_stderr=warn))
        self.assertEqual(rc, 0, out)
        self.assertIn("package lists 3 files", out)

    def test_doctests_minimum(self) -> None:
        rc, out = self.main("--doctests-min", "1")
        self.assertEqual(rc, 0, out)
        self.assertIn("unpacked doctests: 2 passed", out)
        rc, out = self.main("--doctests-min", "3")
        self.assertEqual(rc, 1, out)
        self.assertIn("ran 2 doctests, expected at least 3", out)
        rc, out = self.main("--doctests-min", "1", runner=self.runner(doc_out="test result: FAILED. 1 passed; 1 failed;"))
        self.assertEqual(rc, 1, out)

    def test_tests_in_package_must_run_and_pass(self) -> None:
        spec = "--features open-file-audit --test residual_arithmetic"
        rc, out = self.main(f"--test-in-package={spec}")
        self.assertEqual(rc, 0, out)
        self.assertIn("3 passed", out)
        self.assertTrue(any(c[:3] == ["cargo", "test", "--locked"] and "residual_arithmetic" in c for c in self.calls))
        rc, out = self.main(f"--test-in-package={spec}", runner=self.runner(test_out="test result: ok. 0 passed; 0 failed;"))
        self.assertEqual(rc, 1, out)
        self.assertIn("ran no test", out)
        rc, out = self.main(f"--test-in-package={spec}", runner=self.runner(test_out="test result: FAILED. 2 passed; 1 failed;"))
        self.assertEqual(rc, 1, out)

    def test_every_cargo_call_carries_the_sibling_patch(self) -> None:
        rc, out = self.main("--require-file", "README.md", "--doctests-min", "1", "--test-in-package=--lib",
                            "--check-args=--lib --all-features", "--locked-check-args=--all-features --all-targets")
        self.assertEqual(rc, 0, out)
        self.assertGreaterEqual(len(self.calls), 6)
        self.assertTrue(all(any("patch.crates-io.alice-zip.path=" in a for a in c) for c in self.calls), self.calls)
        self.assertIn(["--lib", "--all-features"], [c[2:4] for c in self.calls if c[1] == "check" and "--locked" not in c])
        self.assertIn(["--locked", "--all-features", "--all-targets"], [c[2:5] for c in self.calls if c[1] == "check" and "--locked" in c])

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tests for scripts/alice_siblings_check.py (run: python3 scripts/test_alice_siblings_check.py)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alice_siblings_check as c  # noqa: E402

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def run(cwd: Path, *args: str) -> str:
    return subprocess.run(
        list(args), cwd=cwd, check=True, capture_output=True, text=True, env=ENV
    ).stdout.strip()


def repo_with_two_commits(path: Path) -> tuple[str, str]:
    path.mkdir()
    run(path, "git", "init", "-q")
    run(path, "git", "config", "maintenance.auto", "false")
    run(path, "git", "config", "gc.auto", "0")
    (path / "a.txt").write_text("a\n")
    run(path, "git", "add", "a.txt")
    run(path, "git", "commit", "-qm", "one")
    first = run(path, "git", "rev-parse", "HEAD")
    (path / "a.txt").write_text("b\n")
    run(path, "git", "commit", "-qam", "two")
    return first, run(path, "git", "rev-parse", "HEAD")


class SiblingsCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "ALICE-Crate"
        self.root.mkdir()
        (self.root / ".github").mkdir()
        self.zip_first, self.zip = repo_with_two_commits(base / "ALICE-Zip")
        _, self.db = repo_with_two_commits(base / "ALICE-DB")
        (self.root / "Cargo.toml").write_text(
            "[dependencies]\n"
            'alice-zip = { path = "../ALICE-Zip", version = ">=0.11, <0.12" }\n'
            'alice-db = { path = "../ALICE-DB", version = ">=0.6, <0.7" }\n'
        )
        self.write(f"# pinned\nALICE-Zip {self.zip}\nALICE-DB {self.db}  # db\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, text: str) -> None:
        (self.root / ".github" / "alice-siblings").write_text(text)

    def problems(self) -> list[str]:
        return c.check(self.root)

    def assertProblem(self, needle: str) -> None:
        got = self.problems()
        self.assertTrue(any(needle in p for p in got), got)

    def test_matching_heads_pass(self) -> None:
        self.assertEqual(self.problems(), [])
        self.assertEqual(c.main(["x", str(self.root)]), 0)

    def test_other_commit_fails(self) -> None:
        self.write(f"ALICE-Zip {self.zip_first}\nALICE-DB {self.db}\n")
        self.assertProblem("../ALICE-Zip is at")

    def test_short_or_branch_ref_fails(self) -> None:
        for bad in (self.zip[:12], "main"):
            self.write(f"ALICE-Zip {bad}\nALICE-DB {self.db}\n")
            self.assertProblem("not '<ALICE-Repo> <40-hex SHA>'")

    def test_repeated_repository_fails(self) -> None:
        self.write(f"ALICE-Zip {self.zip}\nALICE-Zip {self.zip}\nALICE-DB {self.db}\n")
        self.assertProblem("listed twice")

    def test_empty_list_fails(self) -> None:
        self.write("# nothing\n\n")
        self.assertProblem("compared nothing")

    def test_unlisted_manifest_path_fails(self) -> None:
        self.write(f"ALICE-Zip {self.zip}\n")
        self.assertProblem("names ../ALICE-DB, which")

    def test_listed_but_unnamed_fails(self) -> None:
        (self.root / "Cargo.toml").write_text('[dependencies]\nalice-zip = { path = "../ALICE-Zip" }\n')
        self.assertProblem("lists ALICE-DB, which no manifest names")

    def test_fuzz_manifest_must_point_at_the_same_sibling(self) -> None:
        (self.root / "fuzz").mkdir()
        (self.root / "fuzz" / "Cargo.toml").write_text('[dependencies]\nalice-zip = { path = "../ALICE-Zip" }\n')
        self.assertProblem("not the sibling ../ALICE-Zip")
        (self.root / "fuzz" / "Cargo.toml").write_text('[dependencies]\nalice-zip = { path = "../../ALICE-Zip" }\n')
        self.assertEqual(self.problems(), [])

    def test_missing_sibling_fails(self) -> None:
        shutil.move(str(self.root.parent / "ALICE-DB"), str(self.root.parent / "elsewhere"))
        self.assertProblem("../ALICE-DB is not a git checkout")

    def test_dirty_sibling_fails(self) -> None:
        (self.root.parent / "ALICE-Zip" / "a.txt").write_text("dirty\n")
        self.assertProblem("uncommitted")

    def test_missing_list_fails(self) -> None:
        (self.root / ".github" / "alice-siblings").unlink()
        self.assertEqual(self.problems(), [".github/alice-siblings is missing"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tests of scripts/api_boundary.py on hand-built rustdoc JSON (format 60).

The shapes are the ones rustdoc emits for the forms listed in each test name;
the end-to-end run against the real crate is the CI step itself.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_boundary  # noqa: E402

U8 = {"primitive": "u8"}
U32 = {"primitive": "u32"}


def ref(t):
    return {"borrowed_ref": {"lifetime": None, "is_mutable": False, "type": t}}


def path(name, *args, constraints=()):
    return {"resolved_path": {"path": name, "id": None, "args": {"angle_bracketed": {
        "args": [{"type": a} for a in args], "constraints": list(constraints)}}}}


def bound(trait):
    return {"trait_bound": {"trait": trait["resolved_path"], "generic_params": [], "modifier": "none"}}


def fn(name, inputs=(), output=None, generics=None, abi="Rust"):
    return {"crate_id": 0, "name": name, "visibility": "public", "inner": {"function": {
        "sig": {"inputs": [[f"p{i}", t] for i, t in enumerate(inputs)], "output": output, "is_c_variadic": False},
        "generics": generics or {"params": [], "where_predicates": []},
        "header": {"is_const": False, "is_unsafe": False, "is_async": False, "abi": abi}}}}


def doc(items: dict, version: int = 60) -> dict:
    return {"format_version": version, "root": "0", "index": {str(k): v for k, v in items.items()}, "paths": {}}


class EnumerateTest(unittest.TestCase):
    def keys(self, items):
        return api_boundary.enumerate_boundary(doc(items))

    def test_integer_only_fn_is_not_listed(self):
        self.assertEqual(self.keys({1: fn("plain", [U32], U32)}), [])

    def test_byte_shapes_in_parameters_are_listed(self):
        shapes = {
            "slice": ref({"slice": U8}),
            "array": {"array": {"type": U8, "len": "8"}},
            "vec": path("Vec", U8),
            "boxed": path("Box", {"slice": U8}),
            "cow": path("Cow", {"slice": U8}),
            "str": ref({"primitive": "str"}),
            "string": path("String"),
            "chars": ref({"slice": {"primitive": "char"}}),
            "as_ref": {"impl_trait": [bound(path("AsRef", {"slice": U8}))]},
            "iterator": {"impl_trait": [bound(path("Iterator", constraints=[
                {"name": "Item", "args": None, "binding": {"equality": {"type": U8}}}]))]},
            "dyn_read": ref({"dyn_trait": {"traits": [{"trait": path("Read")["resolved_path"],
                                                        "generic_params": []}], "lifetime": None}}),
            "tuple": {"tuple": [U32, ref({"slice": U8})]},
        }
        for name, t in shapes.items():
            self.assertEqual(self.keys({1: fn(name, [t])}), [f"fn::{name}"], name)

    def test_byte_return_type_is_listed(self):
        self.assertEqual(self.keys({1: fn("to_bytes", [], path("Vec", U8))}), ["fn::to_bytes"])

    def test_generic_bound_and_where_clause_are_listed(self):
        g = {"params": [{"name": "B", "kind": {"type": {"bounds": [bound(path("AsRef", {"slice": U8}))],
                                                        "default": None, "is_synthetic": False}}}],
             "where_predicates": []}
        w = {"params": [{"name": "B", "kind": {"type": {"bounds": [], "default": None,
                                                        "is_synthetic": False}}}],
             "where_predicates": [{"bound_predicate": {"type": {"generic": "B"},
                                                       "bounds": [bound(path("BufRead"))],
                                                       "generic_params": []}}]}
        self.assertEqual(self.keys({1: fn("g", [{"generic": "B"}], None, g)}), ["fn::g"])
        self.assertEqual(self.keys({1: fn("w", [{"generic": "B"}], None, w)}), ["fn::w"])

    def test_type_alias_is_followed(self):
        alias = {"crate_id": 0, "name": "Blob", "inner": {"type_alias": {"type": path("Vec", U8),
                                                                          "generics": {}}}}
        arg = {"resolved_path": {"path": "Blob", "id": 2, "args": None}}
        self.assertEqual(self.keys({1: fn("takes_alias", [arg]), 2: alias}), ["fn::takes_alias"])

    def test_extern_abi_is_always_listed(self):
        self.assertEqual(self.keys({1: fn("as_x", [U32], U32, abi={"C": {"unwind": False}})}), ["abi::as_x"])

    def test_serde_and_conversion_impls_are_listed_and_blanket_impls_are_not(self):
        def impl(trait, for_, blanket=None, items=()):
            return {"crate_id": 0, "inner": {"impl": {"trait": trait["resolved_path"], "for": for_,
                                                      "items": list(items), "blanket_impl": blanket,
                                                      "is_synthetic": False}}}
        t = path("Thing")
        items = {
            1: impl(path("Deserialize"), t),
            2: impl(path("Serialize"), t, blanket={"generic": "T"}),
            3: impl(path("TryFrom", ref({"slice": U8})), t),
            4: impl(path("From", U32), t),
            5: impl(path("FromStr"), t),
        }
        self.assertEqual(self.keys(items), ["impl::Deserialize for Thing", "impl::FromStr for Thing",
                                            "impl::TryFrom for Thing"])

    def test_reexport_of_an_outside_item_is_listed(self):
        use = {"crate_id": 0, "visibility": "public", "inner": {"use": {"source": "core::str::from_utf8",
                                                                          "name": "from_utf8", "id": 99,
                                                                          "is_glob": False}}}
        self.assertEqual(self.keys({1: use}), ["reexport::core::str::from_utf8"])

    def test_repeated_keys_are_numbered(self):
        self.assertEqual(self.keys({1: fn("new", [U8]), 2: fn("new", [U8])}), ["fn::new", "fn::new#2"])

    def test_unsupported_format_stops(self):
        with self.assertRaises(SystemExit):
            api_boundary.enumerate_boundary(doc({}, version=61))


class CheckTest(unittest.TestCase):
    def root(self, test_src: str = "") -> Path:
        d = Path(tempfile.mkdtemp())
        (d / "tests").mkdir()
        (d / "tests" / "t.rs").write_text(test_src, encoding="utf-8")
        return d

    def test_listed_items_pass_and_drift_fails(self):
        d = doc({1: fn("read_it", [ref({"slice": U8})]), 2: fn("send", [path("String")])})
        root = self.root("fn read_it_refuses_another_identifier() {}\n")
        ok = {"fn::read_it": "handshake: read_it_refuses_another_identifier",
              "fn::send": "local: a label chosen by the caller"}
        self.assertEqual(api_boundary.check(root, d, ok)[0], [])
        missing = dict(ok)
        del missing["fn::send"]
        self.assertTrue(any("fn::send" in p and "not in" in p for p in api_boundary.check(root, d, missing)[0]))
        stale = dict(ok, **{"fn::gone": "local: was removed some time ago"})
        self.assertTrue(any("stale" in p for p in api_boundary.check(root, d, stale)[0]))
        no_test = dict(ok, **{"fn::read_it": "handshake: no_such_test"})
        self.assertTrue(any("no_such_test" in p for p in api_boundary.check(root, d, no_test)[0]))
        bad = dict(ok, **{"fn::send": "short"})
        self.assertTrue(any("must start" in p for p in api_boundary.check(root, d, bad)[0]))

    def test_nothing_enumerated_fails(self):
        problems, _ = api_boundary.check(self.root(), doc({1: fn("plain", [U32])}), {})
        self.assertEqual(problems, ["enumerated no boundary item (compared nothing)"])


if __name__ == "__main__":
    unittest.main()

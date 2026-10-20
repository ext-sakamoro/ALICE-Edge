#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push`: every command below is
# the one .github/workflows/ci.yml, security-audit.yml or semver-checks.yml runs, with the same
# arguments. A step this script does not cover is a step that can only fail
# remotely, so a step added to a workflow is added here in the same commit.
#
# usage: scripts/preflight.sh [--quick]
#   (none)   every gate: static checks, docs lint, clippy, no_std / thumbv7em
#            builds, rustdoc, examples, the full test suites (no_std, std,
#            law, analytic oracles, determinism goldens, panic contract in
#            both profiles, full feature set, edge-pipeline), MSRV, feature
#            powerset, fuzz build, the security jobs (cargo audit / deny /
#            machete) and the semver gate
#   --quick  static checks, docs lint, clippy, no_std / thumbv7em builds,
#            rustdoc, `cargo test --lib`, the determinism goldens and the
#            debug-profile panic contract; skips the other test suites, the
#            release-profile panic contract, MSRV, feature powerset, fuzz
#            build, the security jobs and the semver gate
#
# The python job runs in the full mode only (it needs network for pip).
#
# Not reproduced here (CI only): the ubuntu-24.04-arm / windows-latest legs of
# the test matrix, sensors-hw clippy off Linux, coverage,
# package-integrity and the time-boxed fuzz runs.
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
case "${1:-}" in
  --quick) quick=1 ;;
  "") ;;
  *) echo "usage: scripts/preflight.sh [--quick]" >&2; exit 2 ;;
esac
# docs.rs feature set (Cargo.toml [package.metadata.docs.rs]) = ci.yml FULL_FEATURES
FULL='std,zip,law,codec,db,ml,sdf,depth-camera,asp,sensors,mqtt,dashboard,ffi,serde-raw'
MSRV='1.87'

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate root invalidates only this crate's fingerprints.
relint() { touch src/lib.rs; }
add_target() { rustup target list --installed | grep -qx "$1" || rustup target add "$1"; }

need actionlint "brew install actionlint"
need python3 "python 3.11+ (tomllib)"

step "ci.yml / actionlint: workflow YAML"
actionlint .github/workflows/*.yml

step "ci.yml / actionlint: workflow-timeouts (every job has timeout-minutes)"
step "eol.yml / line endings (tests, then check)"
# text files are LF in the index and the working tree (.github/workflows/eol.yml, shared by the ALICE crates)
python3 scripts/test_check_eol.py
python3 scripts/check_eol.py
python3 scripts/test_workflow_timeouts.py
python3 scripts/workflow_timeouts.py
python3 scripts/test_tracked_binaries.py
python3 scripts/tracked_binaries.py

step "ci.yml / fmt: cargo fmt --check"
cargo fmt -- --check

step "ci.yml / docs-lint: tests + public documents / CHANGELOG structure"
python3 scripts/test_docs_lint.py
python3 scripts/docs_lint.py --check

step "ci.yml / docs-lint: ALICE siblings (each ../ALICE-* is the commit in .github/alice-siblings)"
python3 scripts/test_alice_siblings_check.py
python3 scripts/alice_siblings_check.py

step "security-audit.yml / package-integrity: tests (the packaging run itself is CI only)"
python3 scripts/test_package_integrity.py

step "ci.yml / python: pyproject license equals the crate license"
python3 scripts/test_check_py_license.py
python3 scripts/check_py_license.py

step "ci.yml / docs-lint: FFI status codes agree in Rust / C / C#"
python3 scripts/test_ffi_status_check.py
python3 scripts/ffi_status_check.py

step "ci.yml / docs-lint: package contents (include allowlist)"
python3 scripts/test_package_contents_check.py
python3 scripts/package_contents_check.py

step "ci.yml / docs-lint: entropy guard (no OS entropy / wall clock in production code)"
python3 scripts/test_entropy_guard.py
python3 scripts/entropy_guard.py

step "ci.yml / clippy: Identifier moves (tests, then against the published release)"
python3 scripts/test_id_move_check.py
python3 scripts/id_move_check.py

step "ci.yml / clippy: API boundary and boundary guard (G7 / G8)"
python3 scripts/test_api_boundary.py
python3 scripts/api_boundary.py
python3 scripts/test_boundary_guard.py
python3 scripts/boundary_guard.py

step "ci.yml / docs-lint: Order guard (G13, interim text layer, tests, then src/)"
python3 scripts/test_order_guard.py
python3 scripts/order_guard.py

step "ci.yml / docs-lint: libm dependency guard (test, then check)"
python3 scripts/test_libm_dependency_guard.py
python3 scripts/libm_dependency_guard.py

step "ci.yml / docs-lint: no direct rkyv (manifests and Rust sources)"
python3 scripts/test_no_direct_rkyv.py
python3 scripts/no_direct_rkyv.py
python3 scripts/test_build_config_guard.py
python3 scripts/build_config_guard.py
python3 scripts/test_build_flags_check.py

step "ci.yml / test + g3-ir.yml: no negated fma result (G3, syntax tree, then the IR findings on this host)"
cargo fmt --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml -- --check
cargo test --quiet --locked --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml
cargo run --quiet --locked --release --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml -- --root .
# the floors are for the CI runners' targets (g3-ir.yml); here only the findings on this host
python3 scripts/test_fneg_fma_ir_check.py
python3 scripts/fneg_fma_ir_check.py --package alice-edge --features "std,zip,law,codec,db,ml,sdf,depth-camera,asp,sensors,mqtt,dashboard,ffi,serde-raw,edge-pipeline" --findings-only

step "ci.yml / docs-lint: changed_paths tests + sync (the changes jobs of g3-ir / fuzz / security-audit / semver-checks)"
python3 scripts/test_changed_paths.py
python3 scripts/changed_paths.py sync

step "ci.yml / docs-lint: run_tests.sh (zero-test guard) tests"
python3 scripts/test_run_tests.py

step "ci.yml / docs-lint: Shared files equal ALICE-Law's table"
python3 scripts/test_shared_files.py
scripts/shared_files.sh

step "ci.yml / clippy: clippy canary data (generator tests, generated from clippy.toml, no drift)"
python3 scripts/test_gen_clippy_canary_calls.py
python3 scripts/gen_clippy_canary_calls.py --check

step "ci.yml / clippy: clippy canary tests (parsers, accounting, one end-to-end run)"
python3 scripts/test_clippy_canary.py

step "law-readme-block.yml: README ALICE-Law block (tests, then the check the job runs, at the block's ALICE-Law pin)"
python3 scripts/test_law_readme_block_check.py
python3 scripts/law_readme_block_check.py

step "security-audit.yml / stub-guard"
scripts/stub_guard.sh

step "ci.yml / clippy: no_std, no_std + law, std, full feature set, + edge-pipeline, pyo3"
relint
cargo clippy --lib --no-default-features -- -D warnings
relint
cargo clippy --lib --test edge_law --examples --no-default-features --features "law" -- -D warnings
relint
cargo clippy --lib --all-targets --features "std" -- -D warnings
relint
cargo clippy --lib --all-targets --features "$FULL" -- -D warnings
relint
cargo clippy --lib --all-targets --features "$FULL,edge-pipeline" -- -D warnings
relint
cargo clippy --lib --features "pyo3" -- -D warnings

step "x86_64 no_std / std clippy (CI runs on x86_64; is_x86_feature_detected! is std-only)"
add_target x86_64-unknown-linux-gnu
relint
cargo clippy --lib --no-default-features --target x86_64-unknown-linux-gnu -- -D warnings
relint
cargo clippy --lib --no-default-features --features "ffi" --target x86_64-unknown-linux-gnu -- -D warnings
relint
cargo clippy --lib --all-targets --features "std" --target x86_64-unknown-linux-gnu -- -D warnings

step "ci.yml / no-std-cross: no_std check + thumbv7em build (alone, ffi, law + ffi) + clippy"
cargo check --lib --no-default-features
add_target thumbv7em-none-eabihf
cargo build --lib --target thumbv7em-none-eabihf --no-default-features
cargo build --lib --target thumbv7em-none-eabihf --no-default-features --features "ffi"
cargo build --lib --target thumbv7em-none-eabihf --no-default-features --features "law,ffi"
relint
cargo clippy --lib --target thumbv7em-none-eabihf --no-default-features --features "law,ffi" -- -D warnings

step "ci.yml / test: sensors-hw clippy (Linux + libudev only)"
if [[ "$(uname -s)" == "Linux" ]]; then
  relint
  cargo clippy --lib --features "std,sensors-hw" -- -D warnings
else
  echo "skip: sensors-hw needs Linux (rppal + libudev); the ubuntu CI job runs it" >&2
fi

step "ci.yml / doc: rustdoc -D warnings (std + docs.rs feature set)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "std"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$FULL"

step "ci.yml / test: examples build (sensors + dashboard + mqtt)"
cargo build --examples --features "std,sensors,dashboard,mqtt"

step "public-text.yml / public-text: Public text (ALICE-Law checker at the pin; tree, then the commits)"
python3 scripts/test_public_text.py
scripts/public_text.sh

if [[ $quick -eq 1 ]]; then
  step "cargo test --lib (quick: no_std, std, law)"
  cargo test --lib --no-default-features
  cargo test --lib --features "std"
  cargo test --lib --features "law"
  # The determinism goldens and the panic contract run in milliseconds and
  # cover the two properties that are easiest to break without noticing
  # (cross-platform bit-exactness, degenerate-input behaviour), so --quick
  # includes them. The remaining integration suites do not run here.
  step "quick: determinism goldens (must run tests) + panic contract (std)"
  scripts/run_tests.sh determinism_golden -- cargo test --test determinism_golden --no-default-features
  scripts/run_tests.sh determinism_golden -- cargo test --test determinism_golden --features "std"
  scripts/run_tests.sh determinism_golden -- cargo test --test determinism_golden --features "law"
  scripts/run_tests.sh determinism_golden,float_edge_golden,edge_semantics_id -- cargo test --test determinism_golden --test float_edge_golden --test edge_semantics_id --features "std,law"
  scripts/run_tests.sh determinism_golden,float_edge_golden -- cargo test --release --test determinism_golden --test float_edge_golden --features "std,law"
  cargo test --test panic_contract --features "std"
  echo; echo "preflight --quick OK (the other test suites, the release-profile panic contract, MSRV, powerset, fuzz build and security jobs are skipped)"; exit 0
fi

step "ci.yml / clippy: clippy canary (every planted banned call is flagged)"
python3 scripts/clippy_canary.py

step "ci.yml / test: float environment (debug and release)"
scripts/run_tests.sh float_environment=2 -- cargo test --test float_environment --no-default-features
scripts/run_tests.sh float_environment=2 -- cargo test --release --test float_environment --no-default-features

step "ci.yml / build-flags: effective rustc flags (fresh target directory)"
python3 scripts/build_flags_check.py -- cargo check --release --lib --no-default-features
python3 scripts/build_flags_check.py -- cargo check --release --lib --features std
python3 scripts/build_flags_check.py -- cargo check --release --lib --features "$FULL"

step "ci.yml / test: no_std core, std, doc tests"
cargo check --lib --no-default-features
cargo test --lib --no-default-features
cargo test --tests --no-run --no-default-features
cargo build --lib --features "std"
cargo test --lib --features "std"
cargo test --doc --features "std"

step "ci.yml / test: law (lib, doc, tests/edge_law.rs, example run)"
cargo test --lib --features "law"
cargo test --doc --features "law"
cargo test --test edge_law --features "law"
cargo run --example edge_law --features "law"

step "ci.yml / test: analytic oracles (tests/analytic_oracle.rs, std + sdf)"
cargo test --test analytic_oracle --features "std,sdf"

step "ci.yml / test: robust fit contract (tests/robust_contract.rs, std)"
cargo test --test robust_contract --features "std"

step "ci.yml / test: db bridge errors (tests/db_bridge_errors.rs, db)"
cargo test --test db_bridge_errors --features "db"

step "ci.yml / test: determinism goldens (no features, std, law, full; each must run tests)"
scripts/run_tests.sh determinism_golden -- cargo test --test determinism_golden --no-default-features
scripts/run_tests.sh determinism_golden -- cargo test --test determinism_golden --features "std"
scripts/run_tests.sh determinism_golden -- cargo test --test determinism_golden --features "law"
scripts/run_tests.sh determinism_golden -- cargo test --test determinism_golden --features "$FULL"
scripts/run_tests.sh determinism_golden,float_edge_golden,edge_semantics_id -- cargo test --test determinism_golden --test float_edge_golden --test edge_semantics_id --features "std,law"
scripts/run_tests.sh determinism_golden,float_edge_golden -- cargo test --release --test determinism_golden --test float_edge_golden --features "std,law"
scripts/run_tests.sh determinism_golden -- cargo test --release --test determinism_golden --features "$FULL"

step "ci.yml / test: panic contract (std + full, debug and release)"
cargo test --test panic_contract --features "std"
cargo test --release --test panic_contract --features "std"
cargo test --test panic_contract --features "$FULL"
cargo test --release --test panic_contract --features "$FULL"

step "ci.yml / test: full feature set, edge-pipeline"
cargo build --lib --features "$FULL"
cargo test --lib --features "$FULL"
cargo test --lib --features "edge-pipeline"
scripts/run_tests.sh minmax_ties=5 -- cargo test --test minmax_ties --features "std,law,ml,sdf,edge-pipeline"

step "ci.yml / msrv: rust-version = $MSRV"
if rustup toolchain list | grep -q "^$MSRV"; then
  cargo "+$MSRV" check --lib --no-default-features
  cargo "+$MSRV" check --lib --no-default-features --features "law"
  cargo "+$MSRV" check --lib --features "$FULL"
else
  echo "toolchain $MSRV not installed (rustup toolchain install $MSRV --profile minimal)" >&2
  exit 1
fi

step "ci.yml / feature-powerset: cargo hack, depth 2"
need cargo-hack "cargo install cargo-hack --locked"
cargo hack check --lib --feature-powerset --depth 2 \
  --exclude-features pyo3,sensors-hw,edge-pipeline --no-dev-deps

step "fuzz.yml: fuzz targets build (nightly + cargo-fuzz)"
need cargo-fuzz "cargo install cargo-fuzz --locked"
(cd fuzz && cargo +nightly fuzz build)

step "security-audit.yml: cargo audit / cargo deny / cargo machete"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-machete "cargo install cargo-machete --locked"
# same ignore list as security-audit.yml (mirrors deny.toml [advisories].ignore)
cargo audit --db "${CARGO_TARGET_DIR:-target}/advisory-db" --deny yanked \
  --ignore RUSTSEC-2026-0049 \
  --ignore RUSTSEC-2026-0098 \
  --ignore RUSTSEC-2026-0099 \
  --ignore RUSTSEC-2026-0104 \
  --ignore RUSTSEC-2026-0235 \
  --ignore RUSTSEC-2025-0141 \
  --ignore RUSTSEC-2025-0134
cargo deny --features "$FULL" check all
cargo machete

step "semver-checks.yml: semver gate tests + cargo semver-checks against the latest crates.io release"
need cargo-semver-checks "cargo install cargo-semver-checks --version 0.50.0 --locked"
python3 scripts/test_semver_gate.py
python3 scripts/semver_gate.py --config .github/semver-gate.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver"

step "ci.yml / python: maturin build + import + binding tests + example (throwaway venv)"
venv="${CARGO_TARGET_DIR:-target}/preflight-python"
python3 -m venv "$venv"
"$venv/bin/python" -m pip install -q --upgrade pip 'maturin>=1.5,<2.0' numpy pytest
"$venv/bin/maturin" build --out "$venv/dist"
"$venv/bin/pip" install -q --force-reinstall --no-deps "$venv"/dist/alice_edge-"$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"-*.whl
"$venv/bin/python" -c "import alice_edge"
collected=$("$venv/bin/python" -m pytest python/tests --collect-only -q 2>/dev/null | grep -c '::' || true)
[[ "$collected" -gt 0 ]] || { echo "no python tests collected" >&2; exit 1; }
"$venv/bin/python" -m pytest python/tests -q
"$venv/bin/python" examples/python_batch.py >/dev/null

echo; echo "preflight OK"

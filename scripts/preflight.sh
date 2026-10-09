#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push`: every command below is
# the one .github/workflows/ci.yml or security-audit.yml runs, with the same
# arguments. A step this script does not cover is a step that can only fail
# remotely, so a step added to a workflow is added here in the same commit.
#
# usage: scripts/preflight.sh [--quick]
#   (none)   every gate: static checks, docs lint, clippy, no_std / thumbv7em
#            builds, rustdoc, examples, the full test suites (no_std, std,
#            law, analytic oracles, determinism goldens, panic contract in
#            both profiles, full feature set, edge-pipeline), MSRV, feature
#            powerset, fuzz build and the security jobs (cargo audit / deny /
#            machete)
#   --quick  static checks, docs lint, clippy, no_std / thumbv7em builds,
#            rustdoc, `cargo test --lib`, the determinism goldens and the
#            debug-profile panic contract; skips the other test suites, the
#            release-profile panic contract, MSRV, feature powerset, fuzz
#            build and the security jobs
#
# Not reproduced here (CI only): the ubuntu-24.04-arm / windows-latest legs of
# the test matrix, sensors-hw clippy off Linux, coverage, semver-checks,
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
FULL='std,zip,law,codec,db,ml,sdf,depth-camera,asp,sensors,mqtt,dashboard,ffi'
MSRV='1.87'

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate root invalidates only this crate's fingerprints.
relint() { touch src/lib.rs; }
add_target() { rustup target list --installed | grep -qx "$1" || rustup target add "$1"; }

need actionlint "brew install actionlint"
need python3 "python 3.9+"

step "ci.yml / actionlint: workflow YAML"
actionlint .github/workflows/*.yml

step "ci.yml / fmt: cargo fmt --check"
cargo fmt -- --check

step "ci.yml / docs-lint: tests + public documents / CHANGELOG structure"
python3 scripts/test_docs_lint.py
python3 scripts/docs_lint.py --check

step "security-audit.yml / stub-guard"
scripts/stub_guard.sh

step "ci.yml / clippy: no_std, no_std + law, std, full feature set"
relint
cargo clippy --lib --no-default-features -- -D warnings
relint
cargo clippy --lib --test edge_law --examples --no-default-features --features "law" -- -D warnings
relint
cargo clippy --lib --all-targets --features "std" -- -D warnings
relint
cargo clippy --lib --all-targets --features "$FULL" -- -D warnings

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

if [[ $quick -eq 1 ]]; then
  step "cargo test --lib (quick: no_std, std, law)"
  cargo test --lib --no-default-features
  cargo test --lib --features "std"
  cargo test --lib --features "law"
  # The determinism goldens and the panic contract run in milliseconds and
  # cover the two properties that are easiest to break without noticing
  # (cross-platform bit-exactness, degenerate-input behaviour), so --quick
  # includes them. The remaining integration suites do not run here.
  step "quick: determinism goldens + panic contract (std)"
  cargo test --test determinism_golden --no-default-features
  cargo test --test determinism_golden --features "std"
  cargo test --test determinism_golden --features "law"
  cargo test --test panic_contract --features "std"
  echo; echo "preflight --quick OK (the other test suites, the release-profile panic contract, MSRV, powerset, fuzz build and security jobs are skipped)"; exit 0
fi

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

step "ci.yml / test: determinism goldens (no features, std, full)"
cargo test --test determinism_golden --no-default-features
cargo test --test determinism_golden --features "std"
cargo test --test determinism_golden --features "law"
cargo test --test determinism_golden --features "$FULL"

step "ci.yml / test: panic contract (std + full, debug and release)"
cargo test --test panic_contract --features "std"
cargo test --release --test panic_contract --features "std"
cargo test --test panic_contract --features "$FULL"
cargo test --release --test panic_contract --features "$FULL"

step "ci.yml / test: full feature set, edge-pipeline"
cargo build --lib --features "$FULL"
cargo test --lib --features "$FULL"
cargo test --lib --features "edge-pipeline"

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

echo; echo "preflight OK"

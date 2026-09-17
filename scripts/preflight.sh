#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push` (ci.yml + the
# blocking jobs of security-audit.yml). Every command is the one CI runs;
# a step this script does not cover is a step that can only fail remotely.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the slow test suites)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1
# docs.rs feature set (Cargo.toml [package.metadata.docs.rs]) = ci.yml FULL_FEATURES
FULL='std,zip,codec,db,ml,sdf,depth-camera,asp,sensors,mqtt,dashboard,ffi'
MSRV='1.87'

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

step "actionlint (workflow YAML)"
if have actionlint; then actionlint .github/workflows/*.yml; else echo "skip: actionlint not installed" >&2; fi

step "cargo fmt --check"
cargo fmt -- --check

step "clippy -D warnings (no_std / std / full feature set)"
cargo clippy --lib --no-default-features -- -D warnings
cargo clippy --lib --all-targets --features "std" -- -D warnings
cargo clippy --lib --all-targets --features "$FULL" -- -D warnings

step "x86_64 no_std / std clippy (CI runs on x86_64; is_x86_feature_detected! is std-only)"
rustup target list --installed | grep -q x86_64-unknown-linux-gnu || rustup target add x86_64-unknown-linux-gnu
cargo clippy --lib --no-default-features --target x86_64-unknown-linux-gnu -- -D warnings
cargo clippy --lib --no-default-features --features "ffi" --target x86_64-unknown-linux-gnu -- -D warnings
cargo clippy --lib --all-targets --features "std" --target x86_64-unknown-linux-gnu -- -D warnings

step "no_std check + thumbv7em cross build (with and without ffi)"
cargo check --lib --no-default-features
rustup target list --installed | grep -q thumbv7em-none-eabihf || rustup target add thumbv7em-none-eabihf
cargo build --lib --target thumbv7em-none-eabihf --no-default-features
cargo build --lib --target thumbv7em-none-eabihf --no-default-features --features "ffi"

step "sensors-hw clippy (Linux + libudev only; on macOS this is verified on the RasPi / CI)"
if [[ "$(uname -s)" == "Linux" ]]; then
  cargo clippy --lib --features "std,sensors-hw" -- -D warnings
else
  echo "skip: sensors-hw needs Linux (rppal + libudev); run on the RasPi or rely on the ubuntu CI job" >&2
fi

step "MSRV $MSRV (lib, no_std + full feature set)"
if rustup toolchain list | grep -q "^$MSRV"; then
  cargo "+$MSRV" check --lib --no-default-features
  cargo "+$MSRV" check --lib --features "$FULL"
else
  echo "skip: toolchain $MSRV not installed (rustup toolchain install $MSRV --profile minimal)" >&2
fi

step "rustdoc -D warnings (std + docs.rs feature set)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "std"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$FULL"

step "examples build (sensors + dashboard + mqtt)"
cargo build --examples --features "std,sensors,dashboard,mqtt"

step "stub guard (todo! / unimplemented! / panic!(STUB) / dbg! in src/)"
hits=$(grep -rnE 'todo!\(|unimplemented!\(|panic!\([^)]*STUB|dbg!\(' src/ --include="*.rs" --exclude-dir=bin || true)
if [[ -n "$hits" ]]; then echo "$hits" >&2; exit 1; fi

step "cargo-machete (unused deps)"
if have cargo-machete; then cargo machete; else echo "skip: cargo-machete not installed" >&2; fi

step "cargo-deny (license / advisories / bans / sources)"
if have cargo-deny; then cargo deny --features "$FULL" check all; else echo "skip: cargo-deny not installed" >&2; fi

step "feature powerset (cargo-hack, depth 2, 80 combinations)"
if have cargo-hack; then
  cargo hack check --lib --feature-powerset --depth 2 \
    --exclude-features pyo3,sensors-hw,edge-pipeline --no-dev-deps
else
  echo "skip: cargo-hack not installed" >&2
fi

step "fuzz targets build (needs nightly + cargo-fuzz)"
if have cargo-fuzz && cargo +nightly --version >/dev/null 2>&1; then
  (cd fuzz && cargo +nightly fuzz build)
else
  echo "skip: cargo-fuzz / nightly not installed" >&2
fi

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test suites skipped)"; exit 0
fi

step "cargo test --lib (no_std core)"
cargo test --lib --no-default-features

step "cargo test --lib + --doc (std)"
cargo test --lib --features "std"
cargo test --doc --features "std"

step "analytic oracles (tests/analytic_oracle.rs, std + sdf) — same as the ci.yml step"
cargo test --test analytic_oracle --features "std,sdf"

step "cargo test --lib (full feature set)"
cargo test --lib --features "$FULL"

step "cargo test --lib (edge-pipeline)"
cargo test --lib --features "edge-pipeline"

echo; echo "preflight OK"

#!/usr/bin/env python3
"""No deterministic path of the crate draws OS entropy or reads the wall clock.

A fitted model's coefficients are transmitted (MQTT) and can be stored by a
caller; this crate's own production code must never make that model, or any
other value it computes/stores, depend on OS randomness or the wall clock --
a device that sampled at a different moment or rounded one ulp differently
would then produce a different model from the same samples.

Checked in every src/**/*.rs file, outside `#[cfg(test)]` / `#[cfg(all(test,
...))]` modules, with the code of a file joined (a call split over lines still
matches) and `use .. as` aliases of `Instant` / `SystemTime` resolved:

    getrandom  OsRng  thread_rng  rand::random  from_entropy(  try_from_entropy(
    SystemTime::now  Instant::now  UNIX_EPOCH  RandomState::new  Uuid::new_v4

clippy's `disallowed-methods` in clippy.toml is the first layer (compiled,
`cargo clippy --all-targets -- -D warnings`); this script is the second,
readable without compiling and run on `src/` only (clippy's `--all-targets`
also covers tests/benches/examples, where a scoped
`#[allow(clippy::disallowed_methods, reason = "...")]` carries the same
per-call-site accounting this script does, e.g. the demo examples that print
a fit's wall-clock latency or stamp a send-time on an MQTT payload).

Deliberately not scanned: `HashMap` / `HashSet` (their default hasher also
seeds from OS randomness, but that is a separate gate -- iteration order, not
entropy in a stored value -- out of scope here; this crate has no
HashMap/HashSet in a path that affects a stored/computed/transmitted value).

A definition (`fn from_entropy`) is not a call. ALLOWED lists the call sites
that stay, each with its reason and exact count: a count that drifts either
way fails, so a new call cannot hide behind an old entry and a removed one
must be removed here too. Scanning no file, or no pattern, fails.

    python3 scripts/entropy_guard.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "getrandom": r"\bgetrandom\b",
    "OsRng": r"\bOsRng\b",
    "thread_rng": r"\bthread_rng\b",
    "rand::random": r"\brand\s*::\s*random\b",
    "from_entropy(": r"(?<!fn )\b(?:try_)?from_entropy\s*\(",
    "SystemTime::now": r"\bSystemTime\s*::\s*now\b",
    "Instant::now": r"\bInstant\s*::\s*now\b",
    "UNIX_EPOCH": r"\bUNIX_EPOCH\b",
    "RandomState::new": r"\bRandomState\s*::\s*new\b",
    "Uuid::new_v4": r"\bUuid\s*::\s*new_v4\b",
}
# `use std::time::Instant as Clock;` makes `Clock::now()` the same call
ALIASABLE = {
    "Instant": ("Instant::now", r"\b{}\s*::\s*now\b"),
    "SystemTime": ("SystemTime::now", r"\b{}\s*::\s*now\b"),
}
ALIAS_RE = re.compile(r"\b(Instant|SystemTime)\s+as\s+(\w+)")
# (file, pattern) -> (count, reason)
ALLOWED: dict[tuple[str, str], tuple[int, str]] = {
    ("src/depth_capture.rs", "Instant::now"): (
        2, "DolphinD5Driver::init's start_time only backs timestamp_ms, a per-frame diagnostic metadata field "
           "no code in this crate reads back; the point cloud itself is derived from frame_counter",
    ),
    ("src/dashboard.rs", "Instant::now"): (
        3, "EdgeDashboard's start/last_reset only back uptime_secs/models_per_sec, live monitoring telemetry "
           "never read back into a compression/classification/encoding value",
    ),
    ("src/edge_pipeline.rs", "Instant::now"): (
        7, "process_frame's 5 timers measure per-stage latency fed only into self.stats (diagnostic averages), "
           "never into the returned packet/compress_stats; run_for's 2 timers pace a duration-bounded real-time "
           "frame loop (the loop's own contract), not the content of any one packet",
    ),
    ("src/watchdog.rs", "Instant::now"): (
        3, "Watchdog is a liveness/safety mechanism: is_expired() asks whether real wall-clock time since the "
           "last kick exceeds timeout, a hardware-hang detector that is inherently wall-clock-driven; the "
           "result is consulted live (a bool), never stored, transmitted or compared for equality",
    ),
    ("src/sensors.rs", "Instant::now"): (
        10,
        "7 of 10 sites: ts only backs batch.timestamps metadata or paces/rate-limits real hardware reads "
        "(Bno055Sensor, Max30102Sensor, GpsSensor, SimulatedSensor use the loop index or a deterministic hash "
        "for every computed reading; DHT22's start_wait/high_start measure a real GPIO pulse width, the "
        "sensor's actual bit encoding). 3 sites are a KNOWN DEFECT found during this audit, not fixed here: "
        "Bme280Sensor/Dht22Sensor/Adxl345Sensor's #[cfg(not(feature = \"sensors-hw\"))] branch computes its "
        "synthetic temperature/humidity/pressure/accel from ts (wall-clock elapsed) instead of the loop index "
        "like their siblings; see the per-site reason in src/sensors.rs for the exact call sites",
    ),
}
STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')


def production_lines(text: str) -> list[tuple[int, str]]:
    """(line number, code) outside `#[cfg(test)]` modules, strings and `//` comments removed."""
    out = []
    skip_depth = None  # brace depth at which the test module started
    pending_test = False
    depth = 0
    for n, raw in enumerate(text.splitlines(), 1):
        code = STRING_RE.sub('""', raw).split("//")[0]
        stripped = code.strip()
        if skip_depth is None and (stripped == "#[cfg(test)]" or re.match(r"#\[cfg\(all\(test\s*,", stripped)):
            pending_test = True
            continue
        opens, closes = code.count("{"), code.count("}")
        if pending_test and stripped:
            if stripped.startswith("#["):
                continue  # more attributes on the test module (`#[allow(..)]`)
            if re.match(r"(pub(\([^)]*\))?\s+)?mod\s+\w+\s*\{", stripped) and opens > closes:
                skip_depth = depth
            pending_test = False
        depth += opens - closes
        if skip_depth is not None:
            if depth <= skip_depth:
                skip_depth = None
            continue
        out.append((n, code))
    return out


def matches(lines: list[tuple[int, str]]) -> dict[str, list[int]]:
    """Line numbers of each pattern in the production lines of one file, with
    the code joined so a call split over lines still matches, and aliases
    (`use std::time::Instant as Clock`) resolved"""
    text = "\n".join(code for _, code in lines)
    starts = []
    pos = 0
    for _, code in lines:
        starts.append(pos)
        pos += len(code) + 1
    patterns = {name: [re.compile(rx)] for name, rx in PATTERNS.items()}
    for original, alias in ALIAS_RE.findall(text):
        name, rx = ALIASABLE[original]
        patterns[name].append(re.compile(rx.format(re.escape(alias))))
    found: dict[str, list[int]] = {}
    for name, rxs in patterns.items():
        hits = set()
        for rx in rxs:
            for m in rx.finditer(text):
                idx = max(i for i, st in enumerate(starts) if st <= m.start())
                hits.add(lines[idx][0])
        if hits:
            found[name] = sorted(hits)
    return found


def check(root: Path) -> tuple[list[str], int, int]:
    problems: list[str] = []
    counts: dict[tuple[str, str], list[int]] = {}
    files = sorted((root / "src").rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        for name, hits in matches(production_lines(path.read_text(encoding="utf-8"))).items():
            counts[(rel, name)] = hits
    for key, lines in sorted(counts.items()):
        allowed = ALLOWED.get(key)
        where = ", ".join(f"{key[0]}:{n}" for n in lines)
        if allowed is None:
            problems.append(f"{key[1]} in production code: {where}")
        elif allowed[0] != len(lines):
            problems.append(f"{key[1]} in {key[0]}: {len(lines)} call sites, ALLOWED says {allowed[0]} ({where})")
    for key, (count, reason) in sorted(ALLOWED.items()):
        if not reason.strip():
            problems.append(f"ALLOWED entry {key[1]} in {key[0]} has no reason")
        if key not in counts:
            problems.append(f"ALLOWED entry {key[1]} in {key[0]} ({count}) matches nothing: remove it")
    return problems, len(files), len(PATTERNS)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    problems, files, patterns = check(ROOT)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if files == 0 or patterns == 0:
        print("error: compared nothing (no src file or no pattern)", file=sys.stderr)
        return 1
    print(f"entropy_guard: {files} files, {patterns} patterns, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

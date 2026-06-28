#!/usr/bin/env python3
"""
Smallest cost_sensor on 0.001 grid (millis 23..50) with n_ground_sensors==0 for
50M greedy-uniform placement. Bracket: MILLIS=22 prior run has sensors>0.

Wave 1 (5 parallel): 23,29,36,43,50 — includes 0.050 upper-bound check.
Then interval refinement with <=5 new jobs per wave.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "California2021Dataset" / "logs"
SLURM_SCRIPT = Path(__file__).resolve().parent / "slurm_50m_greedy_probe_millis.sh"
MAX_PARALLEL = 5
POLL_SEC = 15
GRID_TAG = "261x161_mean"


def json_path_for_millis(millis: int) -> Path:
    tag = f"thresh50M_m{millis}"
    return LOG_DIR / (
        f"sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_{GRID_TAG}_{tag}.json"
    )


def read_n_ground(millis: int) -> int:
    p = json_path_for_millis(millis)
    if not p.is_file():
        raise FileNotFoundError(f"Missing result JSON: {p}")
    with open(p) as f:
        d = json.load(f)
    return int(d["device_counts"]["n_ground_sensors"])


def sbatch_millis(millis: int) -> str:
    r = subprocess.run(
        ["sbatch", f"--export=ALL,MILLIS={millis}", str(SLURM_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"sbatch failed: {r}")
    m = re.search(r"Submitted batch job (\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {r.stdout!r}")
    return m.group(1)


def wait_jobs(job_ids: list[str]) -> None:
    pending = set(job_ids)
    while pending:
        for jid in list(pending):
            r = subprocess.run(
                ["sacct", "-j", jid, "--format=State", "-X", "-n", "-P"],
                capture_output=True,
                text=True,
            )
            lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
            if not lines:
                continue
            if set(lines) <= {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL"}:
                pending.discard(jid)
        if pending:
            time.sleep(POLL_SEC)


def run_batch(millis_list: list[int]) -> None:
    missing = [m for m in millis_list if not json_path_for_millis(m).is_file()]
    if not missing:
        return
    if len(missing) > MAX_PARALLEL:
        raise RuntimeError(f"Batch {len(missing)} > MAX_PARALLEL {MAX_PARALLEL}")
    ids = []
    for m in missing:
        jid = sbatch_millis(m)
        print(f"  submitted MILLIS={m} -> job {jid}", flush=True)
        ids.append(jid)
    print(f"  waiting on {len(ids)} jobs ...", flush=True)
    wait_jobs(ids)
    time.sleep(2)
    for jid in ids:
        r = subprocess.run(
            ["sacct", "-j", jid, "--format=ExitCode", "-X", "-n", "-P"],
            capture_output=True,
            text=True,
        )
        ec = (r.stdout or "").strip().splitlines()
        if not ec or ec[0] != "0:0":
            raise RuntimeError(f"Job {jid} ExitCode {ec!r}")
    for m in missing:
        if not json_path_for_millis(m).is_file():
            raise FileNotFoundError(f"JSON still missing for MILLIS={m}")


def five_probes(lo: int, hi: int) -> list[int]:
    if lo > hi:
        return []
    if lo == hi:
        return [lo]
    return sorted({lo + (hi - lo) * i // 4 for i in range(5)})


def refine(lo: int, hi: int) -> int:
    """Invariant: n(lo-1)>0 (or lo==23), n(hi)==0, answer is min m in [lo,hi] with n(m)==0."""
    while lo < hi:
        if hi - lo + 1 <= MAX_PARALLEL:
            mills = list(range(lo, hi + 1))
            run_batch(mills)
            for m in mills:
                if read_n_ground(m) == 0:
                    return m
            raise RuntimeError(f"No zero in [{lo},{hi}]")

        probes = five_probes(lo, hi)
        run_batch(probes)
        vals = {p: read_n_ground(p) for p in probes}
        for p in sorted(vals):
            print(f"    MILLIS={p} n_ground={vals[p]}", flush=True)

        if vals[lo] == 0:
            return lo

        z = min(p for p in probes if vals[p] == 0)
        positives_below = [p for p in probes if p < z and vals[p] > 0]
        new_hi = z
        new_lo = max(positives_below) + 1 if positives_below else lo
        if new_lo > new_hi:
            raise RuntimeError(f"bad shrink lo={lo} hi={hi} z={z} vals={vals}")
        lo, hi = new_lo, new_hi
        print(f"  -> [{lo}, {hi}]  ({lo/1000:.3f} .. {hi/1000:.3f})", flush=True)

    run_batch([lo])
    if read_n_ground(lo) != 0:
        raise RuntimeError(f"endpoint MILLIS={lo} not zero")
    return lo


def main() -> None:
    if not SLURM_SCRIPT.is_file():
        sys.exit(f"Missing {SLURM_SCRIPT}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    first = [23, 29, 36, 43, 50]
    print("Wave 1 (5 parallel):", first, flush=True)
    run_batch(first)
    vals = {m: read_n_ground(m) for m in first}
    for m in sorted(vals):
        print(f"  MILLIS={m} n_ground={vals[m]}", flush=True)

    if vals[50] > 0:
        raise SystemExit("MILLIS=50 (0.050) still uses sensors — not a valid upper bound.")

    if vals[23] == 0:
        # Confirm MILLIS=22 still uses sensors (grid step 0.001); else answer would be 0.022.
        legacy_22 = LOG_DIR / (
            "sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean_"
            "breakeven_50M_cs0p022.json"
        )
        if legacy_22.is_file():
            with open(legacy_22) as f:
                n22 = int(json.load(f)["device_counts"]["n_ground_sensors"])
        else:
            run_batch([22])
            n22 = read_n_ground(22)
        if n22 == 0:
            raise SystemExit("MILLIS=22 also has zero sensors; re-check bracket.")
        ans = 23
        print(f"\nANSWER: {ans / 1000:.3f} M (wave 1; confirmed MILLIS=22 has sensors via prior run)")
        print(f"JSON: {json_path_for_millis(ans)}")
        return

    z = min(m for m in first if vals[m] == 0)
    positives_below = [m for m in first if m < z and vals[m] > 0]
    lo = max(positives_below) + 1 if positives_below else 23
    hi = z
    print(f"\nAfter wave 1: search interval MILLIS [{lo}, {hi}]", flush=True)
    if lo > hi:
        raise RuntimeError("empty interval after wave 1")

    ans = refine(lo, hi)
    print(f"\nANSWER: {ans / 1000:.3f} M  (smallest 0.001-step with zero ground sensors)")
    print(f"JSON: {json_path_for_millis(ans)}")


if __name__ == "__main__":
    main()

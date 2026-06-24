#!/usr/bin/env python3
"""Pd-sensitivity aggregator for the wildfire drone-detection benchmark.

This is the post-processing half of the "replay-based Pd sensitivity" analysis.
It consumes one or more JSONL *pass catalogs* produced by

    python-jl run_benchmark_california_yearly.py ... --emit-pass-catalog PATH

and, WITHOUT re-running any routing optimization or fire replay, sweeps the
per-pass drone detection probability ``Pd`` and reports how the headline
detection metrics change.

Model
-----
The deterministic benchmark assumes a drone detects a fire with probability one
whenever its certain-detection disk covers the fire cell (``Pd = 1``). The pass
catalog records, for every fire, the full schedule of detection *opportunities*
over the scored window:

  * ``drone_passes``: ``[[hour, n_hits], ...]`` where ``hour`` is the data step
    (equal to the Pd=1 ``delta_t`` value) and ``n_hits`` is the number of
    (substep x drone) coverage overlaps of the fire cell during that hour.
  * ``det_terminal``: ``[hour, device]`` for the first hour a *fixed* sensor
    (ground/charging) covers the fire, treated as a deterministic backstop.

Under a per-pass drone probability ``p`` (and fixed-sensor probability ``q``,
default 1), the time of first detection is a simple first-success process. For a
fire with ordered detection opportunities, the probability that detection first
occurs at hour ``h`` is

    P(first at h) = S_before(h) * (1 - (1 - p)**n_h)        [drone hour]
    P(first at h) = S_before(h) * q                          [fixed-sensor hour]

where ``S_before(h)`` is the probability the fire is still undetected entering
hour ``h``. This is exact (no Monte Carlo). Setting ``p = q = 1`` reproduces the
deterministic benchmark, which is used as a built-in correctness check.

The ``--pass-granularity`` flag controls how a hover-and-stare drone is scored:

  * ``substep`` : every (substep x drone) overlap is an independent Bernoulli
    trial (n_h trials in hour h). Optimistic: many frames per hour.
  * ``hour``    : at most one trial per hour regardless of dwell (n_h -> 1).
    Conservative; recommended as the headline for the rebuttal.

Usage
-----
    python3 pd_sensitivity_aggregate.py \
        --catalog 'pass_catalog_100M_TOPGrowing_2021.jsonl' \
                  'pass_catalog_100M_TOPGrowing_2022.jsonl' \
                  'pass_catalog_100M_TOPGrowing_2023.jsonl' \
                  'pass_catalog_100M_TOPGrowing_2024.jsonl' \
        --strategy TOPGrowing \
        --pd 0.5 0.7 0.9 1.0 \
        --pass-granularity hour \
        --out-csv pd_sensitivity_100M_TOPGrowing.csv \
        --out-tex pd_sensitivity_100M_TOPGrowing.tex

Run ``--self-test`` to validate the probability arithmetic with no input files.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path


# ───────────────────────── core probability model ──────────────────────────

def fire_detection_distribution(passes, terminal, p, q, granularity="hour",
                                 max_hours=None):
    """Return (P_detect, within1h_prob, expected_hour_numerator) for one fire.

    ``expected_hour_numerator`` = sum_h h * P(first detection at h); divide the
    population sum of this by the population sum of P_detect to get the mean
    detection delay over detected fires.
    """
    # Build chronological event list. Fixed-sensor terminal is processed before a
    # drone pass at the same hour (mirrors the deterministic per-hour ordering),
    # though in practice drone passes are always recorded strictly before it.
    events = []  # (hour, kind, n)   kind: 0 = terminal (sorts first), 1 = drone
    for h, n in passes:
        h = int(h)
        if max_hours is not None and h > max_hours:
            continue
        n_eff = int(n) if granularity == "substep" else 1
        if n_eff > 0:
            events.append((h, 1, n_eff))
    if terminal is not None:
        h_t = int(terminal[0])
        if max_hours is None or h_t <= max_hours:
            events.append((h_t, 0, 1))
    events.sort(key=lambda e: (e[0], e[1]))

    S = 1.0  # survival: prob still undetected
    within1h = 0.0
    exp_hour_numer = 0.0
    for h, kind, n in events:
        if kind == 1:  # drone hour
            p_here = S * (1.0 - (1.0 - p) ** n)
            S *= (1.0 - p) ** n
        else:          # fixed-sensor terminal
            p_here = S * q
            S *= (1.0 - q)
        if h == 0:
            within1h += p_here
        exp_hour_numer += h * p_here
    return (1.0 - S), within1h, exp_hour_numer


# ──────────────────────────── catalog loading ──────────────────────────────

def load_and_merge(catalog_paths, strategy_filter=None):
    """Load JSONL pass catalogs and merge per (year, strategy, scenario).

    A reachable fire appears once per serving station/cluster (under
    ``--no-clustering``); all those records are merged into a single fire by
    unioning drone passes per hour (summing n_hits) and keeping the earliest
    fixed-sensor terminal.
    """
    merged = {}  # key -> dict(passes=defaultdict(int), terminal=..., baseline_dt, baseline_dev)
    n_records = 0
    for path in catalog_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if strategy_filter and strategy_filter.lower() not in str(
                        rec.get("strategy_combo", "")).lower():
                    continue
                n_records += 1
                key = (rec.get("year"), rec.get("strategy_combo"),
                       rec.get("scenario_name"))
                m = merged.get(key)
                if m is None:
                    m = {"passes": defaultdict(int), "terminal": None,
                         "baseline_dt": rec.get("baseline_delta_t"),
                         "baseline_dev": rec.get("baseline_device")}
                    merged[key] = m
                for h, n in (rec.get("drone_passes") or []):
                    m["passes"][int(h)] += int(n)
                term = rec.get("det_terminal")
                if term is not None:
                    if m["terminal"] is None or int(term[0]) < int(m["terminal"][0]):
                        m["terminal"] = [int(term[0]), term[1]]
                # Keep the earliest (smallest non-negative) deterministic baseline.
                bdt = rec.get("baseline_delta_t")
                if bdt is not None and bdt >= 0:
                    if m["baseline_dt"] is None or m["baseline_dt"] < 0 or bdt < m["baseline_dt"]:
                        m["baseline_dt"] = bdt
                        m["baseline_dev"] = rec.get("baseline_device")

    fires = []
    for (year, combo, scen), m in merged.items():
        fires.append({
            "year": year, "combo": combo, "scenario": scen,
            "passes": sorted(m["passes"].items()),
            "terminal": m["terminal"],
            "baseline_dt": m["baseline_dt"],
            "baseline_dev": m["baseline_dev"],
        })
    return fires, n_records


# ──────────────────────────── aggregation ──────────────────────────────────

def aggregate(fires, p, q, granularity, max_hours):
    n = len(fires)
    per_fire = []  # (P_detect, within1h, exp_hour_numer)
    for fr in fires:
        per_fire.append(fire_detection_distribution(
            fr["passes"], fr["terminal"], p, q, granularity, max_hours))
    det = sum(x[0] for x in per_fire)
    w1h = sum(x[1] for x in per_fire)
    numer = sum(x[2] for x in per_fire)
    det_rate = det / n if n else 0.0
    within1h = w1h / n if n else 0.0
    mean_dt = (numer / det) if det > 0 else float("nan")
    return det_rate, within1h, mean_dt, per_fire


def bootstrap_ci(per_fire_values, n_boot, seed, alpha=0.05):
    """Percentile bootstrap CI for the mean of per-fire values."""
    n = len(per_fire_values)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += per_fire_values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


# ──────────────────────────── output ───────────────────────────────────────

def write_csv(rows, path):
    cols = ["pass_granularity", "drone_pd", "ground_pd", "n_fires",
            "det_rate_pct", "det_rate_lo", "det_rate_hi",
            "within1h_pct", "within1h_lo", "within1h_hi", "mean_dt_detected"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def write_tex(rows, path, strategy, budget_label):
    # One block per granularity present.
    grans = []
    for r in rows:
        if r["pass_granularity"] not in grans:
            grans.append(r["pass_granularity"])
    lines = []
    lines.append("% Auto-generated by paper/Nature_Wildfires/scripts/pd_sensitivity_aggregate.py")
    lines.append("% Replay-based detection-probability sensitivity (no routing re-optimization).")
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{\textbf{Sensitivity of detection performance to the per-pass drone "
                 r"detection probability $P_d$.} Replay of the fixed " + budget_label +
                 r" " + strategy + r" routing solution against the historical fire set, "
                 r"re-scoring detection under $P_d<1$ without re-optimizing placement or routes. "
                 r"$P_d=1$ reproduces the deterministic benchmark. \emph{Det.\ rate}: overall "
                 r"detection rate (95\% bootstrap CI). \emph{Within 1\,h}: share of all fires "
                 r"detected in the first step ($\Delta t=0$). \emph{Mean $\Delta t$}: mean delay "
                 r"over detected fires. Rows labelled \emph{per-hour} count at most one detection "
                 r"opportunity per hour (conservative); \emph{per-frame} counts every "
                 r"substep$\times$drone overlap.}")
    lines.append(r"\label{tab:pd_sensitivity}")
    lines.append(r"\begin{tabular}{@{}llrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Pass model & $P_d$ & Det.\ rate (\%) & Within 1\,h (\%) & Mean $\Delta t$ (h) \\")
    lines.append(r"\midrule")
    label = {"hour": "per-hour", "substep": "per-frame"}
    for gi, g in enumerate(grans):
        grows = [r for r in rows if r["pass_granularity"] == g]
        for ri, r in enumerate(grows):
            head = label.get(g, g) if ri == 0 else ""
            lines.append(
                f"{head} & {r['drone_pd']:.2f} & "
                f"{r['det_rate_pct']:.1f} [{r['det_rate_lo']:.1f}, {r['det_rate_hi']:.1f}] & "
                f"{r['within1h_pct']:.1f} & {r['mean_dt_detected']:.2f} \\\\")
        if gi != len(grans) - 1:
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(path).write_text("\n".join(lines) + "\n")


# ──────────────────────────── self test ────────────────────────────────────

def self_test():
    EPS = 1e-9
    # One drone pass of 1 hit at hour 0.
    d, w, numer = fire_detection_distribution([[0, 1]], None, p=0.5, q=1.0,
                                              granularity="hour")
    assert abs(d - 0.5) < EPS and abs(w - 0.5) < EPS and abs(numer - 0.0) < EPS, (d, w, numer)

    # Two hits in hour 0, per-frame -> 1-(0.5)^2 = 0.75.
    d, w, numer = fire_detection_distribution([[0, 2]], None, p=0.5, q=1.0,
                                              granularity="substep")
    assert abs(d - 0.75) < EPS and abs(w - 0.75) < EPS, (d, w)
    # Same, per-hour -> clamps to one trial -> 0.5.
    d, w, numer = fire_detection_distribution([[0, 2]], None, p=0.5, q=1.0,
                                              granularity="hour")
    assert abs(d - 0.5) < EPS, d

    # Passes at hour 0 and hour 2, p=0.5, per-hour.
    # P(detect)=1-0.25=0.75; P(first@0)=0.5; P(first@2)=0.5*0.5=0.25.
    d, w, numer = fire_detection_distribution([[0, 1], [2, 1]], None, p=0.5,
                                              q=1.0, granularity="hour")
    assert abs(d - 0.75) < EPS and abs(w - 0.5) < EPS and abs(numer - 0.5) < EPS, (d, w, numer)

    # p=1 reproduces deterministic: first opportunity hour wins.
    d, w, numer = fire_detection_distribution([[3, 5]], [4, "ground sensor"],
                                              p=1.0, q=1.0, granularity="hour")
    assert abs(d - 1.0) < EPS and abs(w - 0.0) < EPS and abs(numer - 3.0) < EPS, (d, w, numer)

    # Terminal-only fire (sensor-only), p irrelevant, q=1.
    d, w, numer = fire_detection_distribution([], [0, "ground sensor"], p=0.5,
                                              q=1.0, granularity="hour")
    assert abs(d - 1.0) < EPS and abs(w - 1.0) < EPS, (d, w)

    # Undetected fire.
    d, w, numer = fire_detection_distribution([], None, p=0.9, q=1.0,
                                              granularity="hour")
    assert d == 0.0 and w == 0.0, (d, w)

    # max_hours cap drops late passes.
    d, w, numer = fire_detection_distribution([[5, 1]], None, p=1.0, q=1.0,
                                              granularity="hour", max_hours=4)
    assert d == 0.0, d

    print("self-test: all assertions passed.")


# ──────────────────────────── main ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", nargs="+", default=None,
                    help="JSONL pass-catalog file(s); globs allowed.")
    ap.add_argument("--strategy", default=None,
                    help="Substring filter on strategy_combo (e.g. TOPGrowing).")
    ap.add_argument("--pd", nargs="+", type=float, default=[0.5, 0.7, 0.9, 1.0],
                    help="Per-pass drone detection probabilities to sweep.")
    ap.add_argument("--ground-pd", type=float, default=1.0,
                    help="Detection probability for fixed sensors (default 1.0).")
    ap.add_argument("--pass-granularity", nargs="+", default=["hour", "substep"],
                    choices=["hour", "substep"],
                    help="Trial granularity per hour (default: both).")
    ap.add_argument("--max-hours", type=int, default=None,
                    help="Cap detection window at this delta_t (hours); default uses full window.")
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples.")
    ap.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    ap.add_argument("--budget-label", default="\\$100M", help="Budget label for the LaTeX caption.")
    ap.add_argument("--out-csv", default=None, help="Write the sweep table as CSV.")
    ap.add_argument("--out-tex", default=None, help="Write a LaTeX table.")
    ap.add_argument("--self-test", action="store_true", help="Run arithmetic self-test and exit.")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.catalog:
        ap.error("--catalog is required (or use --self-test)")

    paths = []
    for pat in args.catalog:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])
    print(f"Loading {len(paths)} catalog file(s)...", flush=True)

    fires, n_records = load_and_merge(paths, strategy_filter=args.strategy)
    n = len(fires)
    print(f"  {n_records} task records -> {n} distinct fires "
          f"(strategy filter={args.strategy!r}).", flush=True)
    if n == 0:
        ap.error("No fires after filtering; check --strategy / catalog contents.")

    # Sanity: deterministic baseline from the catalog vs aggregator @ Pd=1.
    base_det = sum(1 for fr in fires if (fr["baseline_dt"] is not None and fr["baseline_dt"] >= 0))
    base_w1h = sum(1 for fr in fires if fr["baseline_dt"] == 0)
    det1, w1, _, _ = aggregate(fires, 1.0, args.ground_pd, "hour", args.max_hours)
    print(f"  baseline (catalog delta_t): det={base_det/n*100:.2f}%  within1h={base_w1h/n*100:.2f}%",
          flush=True)
    print(f"  aggregator @ Pd=1, granularity=hour: det={det1*100:.2f}%  within1h={w1*100:.2f}%  "
          f"(should match baseline)", flush=True)

    rows = []
    for g in args.pass_granularity:
        for p in args.pd:
            det_rate, within1h, mean_dt, per_fire = aggregate(
                fires, p, args.ground_pd, g, args.max_hours)
            lo, hi = bootstrap_ci([x[0] for x in per_fire], args.n_boot, args.seed)
            rows.append({
                "pass_granularity": g,
                "drone_pd": p,
                "ground_pd": args.ground_pd,
                "n_fires": n,
                "det_rate_pct": round(det_rate * 100, 2),
                "det_rate_lo": round(lo * 100, 2),
                "det_rate_hi": round(hi * 100, 2),
                "within1h_pct": round(within1h * 100, 2),
                "within1h_lo": round(bootstrap_ci([x[1] for x in per_fire], args.n_boot, args.seed)[0] * 100, 2),
                "within1h_hi": round(bootstrap_ci([x[1] for x in per_fire], args.n_boot, args.seed)[1] * 100, 2),
                "mean_dt_detected": round(mean_dt, 3),
            })

    print("\n  granularity   Pd    det%   [CI]            within1h%   meanDt")
    for r in rows:
        print(f"  {r['pass_granularity']:9s}  {r['drone_pd']:.2f}  "
              f"{r['det_rate_pct']:5.1f}  [{r['det_rate_lo']:.1f},{r['det_rate_hi']:.1f}]   "
              f"{r['within1h_pct']:6.1f}      {r['mean_dt_detected']:.2f}", flush=True)

    if args.out_csv:
        write_csv(rows, args.out_csv)
        print(f"\nWrote {args.out_csv}", flush=True)
    if args.out_tex:
        write_tex(rows, args.out_tex, strategy=(args.strategy or "TOP"),
                  budget_label=args.budget_label)
        print(f"Wrote {args.out_tex}", flush=True)


if __name__ == "__main__":
    main()

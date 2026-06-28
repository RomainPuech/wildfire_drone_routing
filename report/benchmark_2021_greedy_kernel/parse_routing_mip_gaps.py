#!/usr/bin/env python3
r"""
Parse Gurobi optimality-gap lines from yearly routing Slurm logs.

Matches Julia output from drone_routing_opt.jl (MaxCov) and
drone_routing_opt_linear.jl (LinearMinTime):
  Gurobi optimality gap: <pct>% (status: <STATUS>)

TOPGrowing does not emit these lines in the current pipeline.

Default log paths match job array 4486528 (MaxCov/TOP array) and 4496394
(Linear) from the Apr 2026 greedy-uniform runs. Override with env
WF_ROUTING_LOG_DIR or positional directory.

Usage (repo root):
  python3 report/benchmark_2021_greedy_kernel/parse_routing_mip_gaps.py
  python3 report/benchmark_2021_greedy_kernel/parse_routing_mip_gaps.py /path/to/logs
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

LINE_RE = re.compile(
    r"Gurobi optimality gap:\s*([0-9.]+)%\s*\(status:\s*([^)]+)\)"
)

# (label, filename)
DEFAULT_FILES = [
    ("20M MaxCov", "wf_greedy_route-4486528_0.out"),
    ("100M MaxCov", "wf_greedy_route-4486528_2.out"),
    ("500M MaxCov", "wf_greedy_route-4486528_4.out"),
    ("20M LinearMinTime", "wf_greedy_route_lin-4496394_0.out"),
    ("100M LinearMinTime", "wf_greedy_route_lin-4496394_1.out"),
    ("500M LinearMinTime", "wf_greedy_route_lin-4496394_2.out"),
]


def parse_log(path: Path) -> tuple[list[float], list[str]]:
    if not path.is_file():
        return [], []
    text = path.read_text(errors="replace")
    gaps: list[float] = []
    statuses: list[str] = []
    for m in LINE_RE.finditer(text):
        gaps.append(float(m.group(1)))
        statuses.append(m.group(2).strip())
    return gaps, statuses


def summarize(gaps: list[float], statuses: list[str]) -> dict:
    if not gaps:
        return {"n": 0}
    ctr = Counter(statuses)
    return {
        "n": len(gaps),
        "mean_pct": round(statistics.mean(gaps), 4),
        "median_pct": round(statistics.median(gaps), 4),
        "min_pct": round(min(gaps), 4),
        "max_pct": round(max(gaps), 4),
        "status_OPTIMAL": ctr.get("OPTIMAL", 0),
        "status_TIME_LIMIT": sum(v for k, v in ctr.items() if "TIME_LIMIT" in k),
        "gap_eq_zero": sum(1 for g in gaps if g == 0.0),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "logs"

    print(f"log_dir={log_dir}\n")
    for label, name in DEFAULT_FILES:
        path = log_dir / name
        gaps, st = parse_log(path)
        s = summarize(gaps, st)
        print(f"{label} ({name})")
        if s["n"] == 0:
            print("  (no Gurobi gap lines)\n")
            continue
        print(
            f"  n={s['n']}  mean={s['mean_pct']}%  median={s['median_pct']}%  "
            f"min={s['min_pct']}%  max={s['max_pct']}%"
        )
        print(
            f"  OPTIMAL={s['status_OPTIMAL']}  TIME_LIMIT={s['status_TIME_LIMIT']}  "
            f"gap==0%: {s['gap_eq_zero']}"
        )
        print()


if __name__ == "__main__":
    main()

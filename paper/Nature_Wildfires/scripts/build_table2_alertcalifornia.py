#!/usr/bin/env python3
"""Build Table 2 (ALERTCalifornia detection rates) from per-fire CSVs.

Reads paper/final_report/csv/alertcalifornia_<Y>_<R>km.csv for every
year in {2021, 2022, 2023, 2024} and radius in {5, 10, 20, 32, 50} km,
computes Wilson 95% confidence intervals, and writes:
  paper/Nature_Wildfires/table2_alertcalifornia.tex

Run from the project root:
    python paper/Nature_Wildfires/scripts/build_table2_alertcalifornia.py
"""

from __future__ import annotations

import csv
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CSV_DIR = REPO / "paper" / "final_report" / "csv"
OUT_DIR = REPO / "paper" / "Nature_Wildfires"

YEARS = [2021, 2022, 2023, 2024]
RADII = [5, 10, 20, 32, 50]


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * sqrt((p * (1 - p) / n + z**2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_csv(year: int, radius: int) -> tuple[int, int] | None:
    """Return (n_fires, n_detected) or None if file is missing."""
    path = CSV_DIR / f"alertcalifornia_{year}_{radius}km.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    det = sum(
        1 for row in rows if row["detected"].strip().lower() in ("1", "true", "t")
    )
    return n, det


def _pct(v: float) -> str:
    """Format a percentage to 1 decimal, but render exactly 100.0 as ``100`` to save width."""
    return "100" if abs(v - 100.0) < 0.05 else f"{v:.1f}"


def format_cell(n: int, det: int) -> str:
    p = det / n if n else 0.0
    lo, hi = wilson_ci(p, n)
    return f"{_pct(p * 100)}\\% [{_pct(lo * 100)}, {_pct(hi * 100)}]"


def build_table() -> None:
    data: dict[tuple[int, int], tuple[int, int] | None] = {}
    for y in YEARS:
        for r in RADII:
            data[(y, r)] = load_csv(y, r)

    # Per-year fire count (constant across radii) for the header label.
    year_n: dict[int, int | None] = {}
    for y in YEARS:
        n_y: int | None = None
        for r in RADII:
            d = data.get((y, r))
            if d is not None:
                n_y = d[0]
                break
        year_n[y] = n_y
    year_cols = " & ".join(
        f"\\textbf{{{y}}}" + (f" (n={year_n[y]})" if year_n[y] is not None else "")
        for y in YEARS
    )
    lines = [
        "\\begin{table}[!t]",
        (
            "\\caption{\\textbf{ALERTCalifornia camera-network detection per year.}"
            " For each detection radius and year, we report the share of California"
            " wildfire ignitions whose location falls within the radius of at least"
            " one camera, with 95\\% Wilson confidence intervals."
            " Computed on the full California fire datasets"
            " (one column per year; 2024 is fully out-of-sample).}"
        ),
        "\\label{tab:alertcalifornia}",
        "{\\small",
        "\\begin{tabular}{@{}l" + "c" * len(YEARS) + "@{}}",
        "\\toprule",
        f"\\textbf{{Radius}} & {year_cols} \\\\",
        "\\midrule",
    ]

    for r in RADII:
        cells = [f"{r}\\,km"]
        for y in YEARS:
            d = data[(y, r)]
            if d is None:
                cells.append("---")
            else:
                n, det = d
                cells.append(format_cell(n, det))
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    out_path = OUT_DIR / "table2_alertcalifornia.tex"
    out_path.write_text(tex)
    print(f"Wrote {out_path} ({len(lines)} lines)")

    # Summary to stdout for STATUS.md
    print("\n--- 10 km detection rates ---")
    for y in YEARS:
        d = data[(y, 10)]
        if d is not None:
            n, det = d
            print(f"  {y}: {det}/{n} = {100 * det / n:.1f}%")


if __name__ == "__main__":
    build_table()

"""Compute Pyrologix predictive quality metrics for California ignition years 2021-2024.

Run from the repo root:
    python -u paper/Nature_Wildfires/scripts/compute_pyrologix_quality.py

Outputs:
    paper/Nature_Wildfires/appendix_pyrologix_quality.md  (JSON audit log)
    paper/Nature_Wildfires/appendix_pyrologix_quality.tex (LaTeX table snippet)

Use --force to recompute even if the JSON log already exists.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
YEARS = [2021, 2022, 2023, 2024]
OUT_DIR = REPO / "paper" / "Nature_Wildfires"


def load_risk_and_mask() -> tuple[np.ndarray, np.ndarray]:
    mask = np.load(REPO / "California2021Dataset" / "mask.npy")
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    risk = np.load(REPO / "California2021Dataset" / "static_risk_pyrologix.npy")
    if risk.ndim == 3:
        risk = risk[0]
    risk = risk.astype(float) / 255.0  # rescale to [0, 1]
    return risk, mask


def compute_metrics(risk: np.ndarray, mask: np.ndarray, years: list[int]) -> tuple[float, list[dict]]:
    V = mask.astype(bool)
    bg_median = float(np.median(risk[V]))

    rows: list[dict] = []
    for y in years:
        ds = REPO / f"California{y}Dataset" / "scenarii"
        files = sorted(ds.glob("*.npy"))
        if not files:
            rows.append({"year": y, "n": 0, "skip": True, "bg_median": bg_median})
            continue
        pts = np.stack([np.load(p) for p in files])   # (N, 2) int32 (row, col)
        vals = risk[pts[:, 0], pts[:, 1]].astype(float)
        fire_median = float(np.median(vals))
        pct_above = float(100.0 * np.mean(vals > bg_median))
        rows.append({
            "year": y,
            "n": int(len(files)),
            "fire_median": round(fire_median, 4),
            "bg_median": round(bg_median, 4),
            "ratio": round(fire_median / bg_median, 4) if bg_median > 0 else float("nan"),
            "pct_above_bg": round(pct_above, 2),
            "improvement_pp": round(pct_above - 50.0, 2),
        })
    return bg_median, rows


def build_latex_table(rows: list[dict], bg_median: float) -> str:
    def fmt(x: float, p: int = 2) -> str:
        return f"{x:.{p}f}" if isinstance(x, float) else str(x)

    tex_rows = []
    for r in rows:
        if r.get("skip"):
            tex_rows.append(
                f"{r['year']} & 0 & {fmt(bg_median)} & --- & --- & --- \\\\"
            )
        else:
            tex_rows.append(
                f"{r['year']} & {r['n']} & {fmt(r['fire_median'])} & {fmt(bg_median)} & "
                f"{fmt(r['ratio'], 3)} & {fmt(r['pct_above_bg'], 1)}\\% & "
                f"{fmt(r['improvement_pp'], 1)} \\\\"
            )

    caption = (
        "\\textbf{Pyrologix predictive quality for California ignition locations 2021--2024.} "
        "The background median is computed over all valid California cells (Pyrologix is static so the "
        "background is year-independent). For each year we report the median Pyrologix value at ignition "
        "cells, its ratio to the background, the share of ignitions whose cell value exceeds the "
        "background median, and the improvement over the random-baseline 50\\%."
    )
    header = (
        "\\textbf{Year} & \\textbf{N fires} & \\textbf{Fire median} & \\textbf{Bg median} & "
        "\\textbf{Ratio} & \\textbf{\\% above bg} & \\textbf{Improvement (pp)} \\\\"
    )

    return (
        "\\begin{table}[!t]\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:pyrologix_quality}\n"
        "{\\small\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(tex_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute even if JSON log exists.")
    args = parser.parse_args()

    json_path = OUT_DIR / "appendix_pyrologix_quality.md"
    tex_path = OUT_DIR / "appendix_pyrologix_quality.tex"

    if json_path.exists() and not args.force:
        print(f"JSON log already exists at {json_path}. Use --force to recompute.")
        rows = json.loads(json_path.read_text())
        bg_median = next(r["bg_median"] for r in rows if not r.get("skip"))
    else:
        risk, mask = load_risk_and_mask()
        print(f"Risk shape: {risk.shape}, dtype: {risk.dtype}, range: [{risk.min()}, {risk.max()}]")
        print(f"Mask shape: {mask.shape}, valid cells: {mask.sum()}")
        bg_median, rows = compute_metrics(risk, mask, YEARS)
        json_path.write_text(json.dumps(rows, indent=2))
        print(f"JSON log written to {json_path}")

    print(f"\nbg_median = {bg_median:.4f}")
    for r in rows:
        print(r)

    tex = build_latex_table(rows, bg_median)
    tex_path.write_text(tex)
    print(f"\nLaTeX table written to {tex_path}")


if __name__ == "__main__":
    main()

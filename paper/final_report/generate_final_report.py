#!/usr/bin/env python3
"""Generate final_report.md from placement JSONs and bundled CSVs.

**Figure 4** (§1.2): four **operational-scale** placement maps — 4a=20M, 4b=50M, 4c=100M, 4d=500M. See ``docs/FIGURE4.md``.

**Figure 3** (Nature ``frontier.png``): detection frontier only — ``paper/Nature_Wildfires/make_figure3_frontier.py`` calls ``compute_frontier_detection_curves()`` so curves match this report's §1.1 + §2 CSVs.

§1.1 placement table:
  * Sensor allocation JSONs are read only from ``placement_data/logs/`` next to this
    script (``paper/final_report/placement_data/logs/``) — not from
    ``California2021Dataset/logs/``, so the report can be regenerated without that tree.
  * The 100-fire benchmark subsample uses ``config_california_2021.json`` and
    ``scenarii/*.npy`` from ``California2021Dataset/`` at the repo root when present;
    if that directory is missing, the copies under ``placement_data/`` are used.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
# Path from repo root, for docs (e.g. paper/final_report)
OUT_REL = OUT.relative_to(ROOT).as_posix()
CSV_DIR = OUT / "csv"
IMG_OUT = OUT / "images"

# Routing results are POOLED across all benchmark years (2021–2024): every (budget,
# strategy) cell concatenates the per-year per-fire CSVs and computes detection rate +
# Wilson CI over the full pooled set of fires (not averaged over per-year rates).
ROUTING_YEARS: list[int] = [2021, 2022, 2023, 2024]


def _routing_year_paths(budget: int, strat: str) -> list[Path]:
    """Per-year CSV paths for one (budget, strategy) cell, pooled across years.

    * 20/50/100M — 20260520_062339 all-fires export.
    * 75M        — 20260522_170454 all-fires export.
    * 500M       — eps1 pruned placement, 20260523_212551, with the T20
      operational-cell ground-sensor detection fix applied (``_sensorfix``).
    """
    if budget == 75:
        stamp = "20260522_170454"
    else:
        stamp = "20260520_062339"
    if budget == 500:
        return [
            CSV_DIR / f"benchmark_results_yearly_500M_{y}_{strat}_20260523_212551_sensorfix.csv"
            for y in ROUTING_YEARS
        ]
    return [
        CSV_DIR / f"benchmark_results_yearly_{budget}M_{y}_{strat}_{stamp}.csv"
        for y in ROUTING_YEARS
    ]


# (budget, strategy, mode-label, [per-year CSV paths]) — same §2 cells as final_report.md.
ROUTING_CSV_SPECS: list[tuple[int, str, str, list[Path]]] = [
    (b, s, "pooled 2021–2024", _routing_year_paths(b, s))
    for b in (20, 50, 75, 100, 500)
    for s in ("TOPGrowing", "MaxCov", "LinearMinTime")
]
# Self-contained placement JSONs (do not read California2021Dataset/logs/)
BUNDLE = OUT / "placement_data"
BUNDLE_LOGS = BUNDLE / "logs"


def resolve_dataset_root_for_benchmark() -> Path:
    """Config + scenarii for the fixed 100-fire benchmark (seed 42).

    Prefer the canonical ``California2021Dataset`` at the repository root. If absent,
    use the bundled ``placement_data/`` copy (offline / archive reproduction).
    """
    canonical = ROOT / "California2021Dataset"
    if (canonical / "config_california_2021.json").is_file() and (canonical / "scenarii").is_dir():
        return canonical
    if (BUNDLE / "config_california_2021.json").is_file() and (BUNDLE / "scenarii").is_dir():
        return BUNDLE
    raise FileNotFoundError(
        "Need either California2021Dataset/ or placement_data/ with "
        "config_california_2021.json and scenarii/ at:\n"
        f"  {canonical}\n  or {BUNDLE}"
    )

# --- Wilson score 95% CI for binomial proportion ---
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    rad = z * math.sqrt((p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - rad), min(1.0, center + rad)


def t_ci_95(xs: list[float]) -> tuple[float, float] | None:
    n = len(xs)
    if n < 2:
        return None
    m = mean(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    se = s / math.sqrt(n)
    # approximate t critical for 95% (df=n-1)
    tcrit_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042, 60: 2.000, 90: 1.987, 120: 1.980}
    df = n - 1
    t = 1.96 if df >= 120 else tcrit_table.get(df, 2.0)
    return m - t * se, m + t * se


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_rows_multi(paths: list[Path]) -> list[dict[str, Any]]:
    """Concatenate per-fire rows across all per-year CSVs for one routing cell.

    Pooling at the row level means detection rate and its Wilson interval are
    computed over the full multi-year fire population (each fire = one trial),
    rather than averaging per-year percentages.
    """
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(
                "Routing data: missing per-year CSV (refresh "
                f"``paper/final_report/csv/``): {p}"
            )
        rows.extend(load_rows(p))
    return rows


def is_detected(row: dict[str, Any]) -> bool:
    d = row["device"].strip().lower()
    try:
        dt = float(row["delta_t"])
    except ValueError:
        return False
    return d != "undetected" and dt >= 0


def routing_block(rows: list[dict[str, Any]], n_disc: int) -> dict[str, Any]:
    n = len(rows)
    det = [r for r in rows if is_detected(r)]
    k = len(det)
    reachable_rows = [r for r in rows if r["cluster"].strip().lower() != "none"]
    nr = len(reachable_rows)
    det_r = [r for r in reachable_rows if is_detected(r)]
    kr = len(det_r)
    dts = [float(r["delta_t"]) for r in det]
    ci_p = wilson_ci(k, n)
    # If n_disc < kr (e.g. benchmark 100-fire count vs all-fires CSV), fall back to nr
    n_disc_eff = n_disc if (n_disc and n_disc >= kr) else nr
    ci_disc = wilson_ci(kr, n_disc_eff) if n_disc_eff > 0 else None
    ci_dt = t_ci_95(dts) if dts else None
    # Within-1h: fires with delta_t == 0, denominator = all rows (n)
    k_dt0 = sum(1 for r in det if float(r["delta_t"]) == 0)
    ci_within1h = wilson_ci(k_dt0, n) if n > 0 else None
    return {
        "n": n,
        "k": k,
        "p": k / n if n else 0.0,
        "ci_p": ci_p,
        "kr": kr,
        "n_disc": n_disc_eff,
        "ci_disc": ci_disc,
        "mean_dt": mean(dts) if dts else float("nan"),
        "med_dt": median(dts) if dts else float("nan"),
        "max_dt": max(dts) if dts else float("nan"),
        "ci_dt": ci_dt,
        "k_dt0": k_dt0,
        "within1h_p": k_dt0 / n if n else 0.0,
        "within1h_ci": ci_within1h,
    }


def fmt_pct_interval(ci: tuple[float, float] | None, p: float) -> str:
    if ci is None:
        return f"{100 * p:.1f}% (n too small for CI)"
    return f"{100 * p:.1f}% [{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"


def fmt_dt_interval(ci: tuple[float, float] | None, m: float) -> str:
    if ci is None or math.isnan(m):
        return "—"
    return f"{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def placement_table() -> list[dict[str, Any]]:
    """Return placement summary rows; prefer live computation via visualize script."""
    try:
        import numpy as np

        spec = importlib.util.spec_from_file_location("viz2021", ROOT / "visualize_sensor_placement_2021.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)

        dataset_root = resolve_dataset_root_for_benchmark()
        with open(dataset_root / "config_california_2021.json") as f:
            config = json.load(f)
        scenarii = sorted((dataset_root / "scenarii").glob("*.npy"))
        valid = [
            sf
            for sf in scenarii
            if all(f"{k}_{sf.stem.replace('_scenario1', '')}" in config for k in ("offset", "date", "time"))
        ]
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(valid), size=100, replace=False))
        benchmark = [valid[i] for i in idx]
        fire_rows, fire_cols = [], []
        for sf in benchmark:
            pt = np.load(str(sf))
            fire_rows.append(int(pt[0]))
            fire_cols.append(int(pt[1]))

        rows_out: list[dict] = []
        configs = [
            (20, "sensor_alloc_GaussianBudget20M_StationMaxGreedyUniform_261x161_mean.json", "StationMaxGreedyUniform"),
            (50, "sensor_alloc_GaussianBudget50M_StationMaxGreedyUniform_261x161_mean.json", "StationMaxGreedyUniform"),
            (75, "sensor_alloc_GaussianBudget75M_StationMaxGreedyUniform_261x161_mean.json", "StationMaxGreedyUniform"),
            (100, "sensor_alloc_GaussianBudget100M_StationMaxGreedyUniform_261x161_mean.json", "StationMaxGreedyUniform"),
            (
                500,
                "sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_fullpool_eps1_6h_pruned.json",
                "StationMaxUniformFixedDrones (full pool, ε=1, warm-start; post-hoc prune)",
            ),
        ]
        for budget, fname, label in configs:
            jpath = BUNDLE_LOGS / fname
            if not jpath.is_file():
                raise FileNotFoundError(
                    f"Missing bundled placement JSON (expected under {BUNDLE_LOGS}): {fname}"
                )
            with jpath.open() as f:
                d = json.load(f)
            ground_opt = {tuple(x) for x in d["ground_sensor_locations"]}
            charging = [tuple(x) for x in d["charging_station_locations"]]
            drones = d["drones_per_charging_station"]
            dc = d["device_counts"]
            clusters = mod.compute_clusters(charging, drones)
            dg, disc, nd = mod.classify_fires(fire_rows, fire_cols, clusters, ground_opt, 5)
            n_disc = len(dg) + len(disc)
            pct_benchmark = 100.0 * n_disc / 100.0
            ng, ns, ndev = dc["n_ground_sensors"], dc["n_charging_stations"], dc["n_drones"]
            spent = ng * 0.1 + ns * 0.15 + ndev * 0.05
            rows_out.append(
                {
                    "budget": budget,
                    "placement_label": label,
                    "json": jpath.name,
                    "n_ground": ng,
                    "n_station": ns,
                    "n_drones": ndev,
                    "n_clusters": len(clusters),
                    "discoverable": n_disc,
                    "pct_benchmark": pct_benchmark,
                    "spent_m": spent,
                    "cap_m": float(d.get("budget_millions", budget)),
                }
            )
        return rows_out
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "placement_table() could not compute placement counts from the committed "
            "inputs (placement JSONs under paper/final_report/placement_data/logs/ and "
            "the California2021Dataset scenarii). Fix the underlying error above rather "
            "than relying on pre-baked numbers.\n"
            f"  underlying error: {exc!r}"
        ) from exc


def compute_frontier_detection_curves() -> dict[str, list[float]]:
    """
    Detection frontier inputs aligned with ``final_report.md`` §1.1 (placement reachable)
    and §2 (overall detection % from the same CSVs as the routing table).

    Returns:

    * ``budgets_m`` — cap labels (0, 20, 50, 100, 500) in MUSD (for reference).
    * ``spent_millions`` — realized device spend from §1.1 (0 at origin; equals cap for
      greedy-uniform rows; 500M uses pruned layout spend, e.g. **172.6** M$).
    * Percent series keyed by cap but sampled at the same points as ``spent_millions``.
    * ``top_mean_dt_hours``, ``top_mean_dt_ci_lo``, ``top_mean_dt_ci_hi`` — parallel to
      ``spent_millions``: mean ``delta_t`` over **detected** fires for TOPGrowing, with
      Student--t 95% interval when ``routing_block`` provides ``ci_dt``; ``nan`` at spend 0
      and when no interval exists (aligned with §2 table).
    """
    placements = placement_table()
    disc_map = {int(p["budget"]): int(p["discoverable"]) for p in placements}
    spent_map = {int(p["budget"]): float(p["spent_m"]) for p in placements}
    budgets = sorted(disc_map.keys())
    by_strat: dict[str, dict[int, float]] = {
        "MaxCov": {},
        "TOPGrowing": {},
        "LinearMinTime": {},
    }
    top_dt_by_budget: dict[int, dict[str, Any]] = {}
    for budget, strat, _mode, paths in ROUTING_CSV_SPECS:
        rows = load_rows_multi(paths)
        nd = disc_map[budget]
        st = routing_block(rows, nd)
        by_strat[strat][budget] = 100.0 * st["p"]
        if strat == "TOPGrowing":
            top_dt_by_budget[budget] = {
                "mean_dt": st["mean_dt"],
                "ci_dt": st["ci_dt"],
                "within1h_p": st["within1h_p"],
                "within1h_ci": st["within1h_ci"],
            }

    def stack(strat: str) -> list[float]:
        return [0.0] + [by_strat[strat][b] for b in budgets]

    reachable = [0.0] + [100.0 * disc_map[b] / 100.0 for b in budgets]
    budgets_m = [0.0] + [float(b) for b in budgets]
    spent_millions = [0.0] + [spent_map[b] for b in budgets]

    top_mean_dt_hours: list[float] = [float("nan")]
    top_mean_dt_ci_lo: list[float] = [float("nan")]
    top_mean_dt_ci_hi: list[float] = [float("nan")]
    top_within1h_pct: list[float] = [0.0]
    top_within1h_ci_lo: list[float] = [float("nan")]
    top_within1h_ci_hi: list[float] = [float("nan")]
    for b in budgets:
        rec = top_dt_by_budget[b]
        m = float(rec["mean_dt"])
        top_mean_dt_hours.append(m)
        ci = rec["ci_dt"]
        if ci is not None and not math.isnan(m):
            top_mean_dt_ci_lo.append(float(ci[0]))
            top_mean_dt_ci_hi.append(float(ci[1]))
        else:
            top_mean_dt_ci_lo.append(float("nan"))
            top_mean_dt_ci_hi.append(float("nan"))
        w = float(rec["within1h_p"])
        top_within1h_pct.append(100.0 * w)
        wci = rec["within1h_ci"]
        if wci is not None:
            top_within1h_ci_lo.append(100.0 * float(wci[0]))
            top_within1h_ci_hi.append(100.0 * float(wci[1]))
        else:
            top_within1h_ci_lo.append(float("nan"))
            top_within1h_ci_hi.append(float("nan"))

    return {
        "budgets_m": budgets_m,
        "spent_millions": spent_millions,
        "placement_reachable_pct": reachable,
        "maxcov_pct": stack("MaxCov"),
        "top_pct": stack("TOPGrowing"),
        "linear_pct": stack("LinearMinTime"),
        "top_mean_dt_hours": top_mean_dt_hours,
        "top_mean_dt_ci_lo": top_mean_dt_ci_lo,
        "top_mean_dt_ci_hi": top_mean_dt_ci_hi,
        "top_within1h_pct": top_within1h_pct,
        "top_within1h_ci_lo": top_within1h_ci_lo,
        "top_within1h_ci_hi": top_within1h_ci_hi,
    }


# ALERTCalifornia baseline — pre-computed CSV files (one per radius sweep)
ALERTCA_RADII: list[tuple[float, str]] = [
    (5.0,  "csv/alertcalifornia_baseline_5km.csv"),
    (10.0, "csv/alertcalifornia_baseline_10km.csv"),
    (20.0, "csv/alertcalifornia_baseline_20km.csv"),
    (32.0, "csv/alertcalifornia_baseline_32km.csv"),   # ≈ 20 miles, actual zone radius
    (50.0, "csv/alertcalifornia_baseline_50km.csv"),
]
# Canonical coverage figure (matches the actual ~20-mile zone radius)
ALERTCA_CANONICAL_KM = 32.0
ALERTCA_CANONICAL_IMG = "images/alertcalifornia_coverage_32km.png"


def alertcalifornia_baseline_data() -> list[dict[str, Any]]:
    """Read pre-computed per-fire CSVs and return one row per radius sweep.

    Each row: radius_km, n_detected, n_total, pct, ci_lo, ci_hi.
    Raises FileNotFoundError when a CSV is absent.
    """
    results = []
    for radius_km, rel_path in ALERTCA_RADII:
        csv_path = OUT / rel_path
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing ALERTCalifornia baseline CSV: {csv_path}\n"
                f"Run: python code/benchmark_alertcalifornia.py --radius {radius_km:g} "
                f"--out {rel_path}"
            )
        rows = load_rows(csv_path)
        n = len(rows)
        k = sum(int(r["detected"]) for r in rows)
        p = k / n if n else 0.0
        ci = wilson_ci(k, n)
        results.append(
            {
                "radius_km": radius_km,
                "n_detected": k,
                "n_total": n,
                "pct": 100.0 * p,
                "ci_lo": 100.0 * ci[0] if ci else float("nan"),
                "ci_hi": 100.0 * ci[1] if ci else float("nan"),
            }
        )
    return results


def copy_placement_figures() -> None:
    """Copy PNGs into final_report/images/ when they exist under report/."""
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    mapping = [
        (
            ROOT / "report" / "california_2021_sensor_clusters_opt_greedy_uniform_20M.png",
            IMG_OUT / "california_2021_sensor_clusters_opt_greedy_uniform_20M.png",
        ),
        (
            ROOT / "report" / "california_2021_sensor_clusters_opt_greedy_uniform_50M.png",
            IMG_OUT / "california_2021_sensor_clusters_opt_greedy_uniform_50M.png",
        ),
        (
            ROOT / "report" / "california_2021_sensor_clusters_opt_greedy_uniform_100M.png",
            IMG_OUT / "california_2021_sensor_clusters_opt_greedy_uniform_100M.png",
        ),
        (
            ROOT / "report" / "california_2021_sensor_clusters_opt_uniform_fixed_500M_fullpool_eps1_6h_pruned.png",
            IMG_OUT / "california_2021_sensor_clusters_opt_uniform_fixed_500M_fullpool_eps1_6h_pruned.png",
        ),
    ]
    for src, dst in mapping:
        if src.is_file():
            shutil.copy2(src, dst)


def main() -> None:
    copy_placement_figures()
    placements = placement_table()
    disc_map = {p["budget"]: p["discoverable"] for p in placements}

    lines: list[str] = []
    lines.append("# California 2021 — Final benchmark report (placement + routing)\n")
    lines.append("\n**Generated:** this file is produced by `generate_final_report.py` in this folder.\n")
    lines.append("\n**Scope:** greedy-uniform (or uniform-fixed full-pool) **placement** at **20M, 50M, 100M, 500M**; **yearly routing** on the same 100-fire benchmark subset (`RANDOM_SEED = 42`).\n")

    # --- Placement ---
    lines.append("\n## 1. Placement (main)\n")
    lines.append("\n### 1.1 Summary table\n")
    lines.append("\n**Discoverable fires** = ground-detectable ∪ drone-reachable on the benchmark 100 fires, using the same geometry as `visualize_sensor_placement_2021.py` (one-way Chebyshev reach `floor(7/2)=3` operational cells from a charging station; ground sensors on the pooled operational grid).\n")
    lines.append("\n**Benchmark placement coverage** = discoverable fires / 100 (share of the benchmark sample that placement geometry can ever expose to drones or passive sensors).\n")
    lines.append(
        "\n**§1.1 data sources:** Sensor allocation JSONs are read only from **`placement_data/logs/`** here (not from `California2021Dataset/logs/`). "
        "The 100-fire subsample uses `config_california_2021.json` and `scenarii/*.npy` from **`California2021Dataset/`** at the repository root when present; if that tree is missing, the bundled copies under **`placement_data/`** are used.\n"
    )
    lines.append(
        "\n**Device spend (M$)** uses the standard unit prices: sensor **$0.1M**, station **$0.15M**, drone **$0.05M** each "
        "(equivalently **$0.5M** per open charging station with **7** fixed drones: $0.15+7×$0.05, as in the Julia MIP). "
        "Spend is **not** required to equal the budget cap when the placement objective uses ε-regularization (500M row).\n"
    )
    lines.append(
        "\n| Budget (cap) | Placement strategy | Ground sensors | Stations | Drones | Clusters | Discoverable / 100 | Benchmark coverage | Device spend (M$) | Cap (M$) |\n"
    )
    lines.append(
        "|--------------|--------------------|----------------|----------|--------|----------|--------------------|--------------------|-------------------|----------|\n"
    )
    for p in placements:
        lines.append(
            f"| **{p['budget']}M** | {p['placement_label']} | {p['n_ground']} | {p['n_station']} | {p['n_drones']} | {p['n_clusters']} | **{p['discoverable']}** | **{p['pct_benchmark']:.1f}%** | **{p['spent_m']:.2f}** | **{p['cap_m']:.0f}** |\n"
        )
    lines.append(
        "\n*500M **solver** placement is **uniform-fixed full-pool** with **ε=1** (`fullpool_eps1_6h`, 6 h Gurobi, warm-started from ε=0.5): **359** stations, **2513** drones, **~182.6 M$** device spend. "
        "This report’s **§1.1 table and Figure 4d** use the **post-hoc pruned** file `sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_fullpool_eps1_6h_pruned.json` (**20** redundant stations removed, **no** loss of benchmark coverage): **339** stations, **2373** drones, **172.60 M$** spend "
        "(**3.1 M$** sensors + **169.5 M$** stations with drones, 339×(0.15+7×0.05 M$)). **~327 M$** of the **500 M$** cap remains unused by design. "
        "The older `ws500f20` placement and `StationMaxGreedyUniform` 500M JSON (zero stations) are **not** used here.*\n"
    )

    lines.append(
        "\n### 1.2 Placement maps (operational scale, 5 km/cell; pooled Pyrologix + clusters + benchmark fires)\n"
    )
    lines.append(
        "\n**Figure 4** refers collectively to the four maps below (this subsection). "
        "**Figure 4a** = 20M, **4b** = 50M, **4c** = 100M, **4d** = 500M. "
        f"Authoring glossary: `{OUT_REL}/docs/FIGURE4.md`.\n"
    )
    imgs = [
        (20, "a", "images/california_2021_sensor_clusters_opt_greedy_uniform_20M.png"),
        (50, "b", "images/california_2021_sensor_clusters_opt_greedy_uniform_50M.png"),
        (100, "c", "images/california_2021_sensor_clusters_opt_greedy_uniform_100M.png"),
        (500, "d", "images/california_2021_sensor_clusters_opt_uniform_fixed_500M_fullpool_eps1_6h_pruned.png"),
    ]
    for b, panel, rel in imgs:
        lines.append(f"\n#### {b}M (Figure 4{panel})\n")
        lines.append(f"\n![Figure 4{panel} — {b}M placement, operational scale]({rel})\n")

    # --- Routing ---
    lines.append("\n## 2. Routing (main)\n")
    lines.append("\n**Detection** (per CSV row): `device != undetected` and `delta_t >= 0`.\n")
    lines.append("\n**Overall detection %** is `detected / n_rows` with **Wilson 95% score interval** (two-sided).\n")
    lines.append("\n**Among placement-discoverable** uses the **placement** discoverable count from §1.1 as denominator (same 100 fires); numerator = detected rows whose fire is placement-discoverable — implemented as detections among rows with `cluster != none` (routing assignment), which matches the April 2026 benchmark convention when placement is consistent.\n")
    lines.append("\n**Mean Δt** is the mean of `delta_t` over **detected** fires only, with **Student-t 95% interval** when ≥2 detects.\n")
    lines.append("\n| Budget | Strategy | Routing mode | n rows | Overall detection | Among discoverable | Mean Δt (detected) | Median Δt | Max Δt |\n")
    lines.append("|--------|----------|--------------|--------|-------------------|--------------------|--------------------|-----------|--------|\n")

    for budget, strat, mode, paths in ROUTING_CSV_SPECS:
        rows = load_rows_multi(paths)
        nd = disc_map[budget]
        st = routing_block(rows, nd)
        overall = fmt_pct_interval(st["ci_p"], st["p"])
        p_disc = st["kr"] / nd if nd else 0.0
        among = fmt_pct_interval(st["ci_disc"], p_disc) if st["ci_disc"] else "—"
        dt = fmt_dt_interval(st["ci_dt"], st["mean_dt"]) if st["ci_dt"] else f"{st['mean_dt']:.3f}" if not math.isnan(st["mean_dt"]) else "—"
        lines.append(
            f"| **{budget}M** | {strat} | {mode} | {st['n']} | {overall} | {among} | {dt} | {st['med_dt']:.3f} | {st['max_dt']:.3f} |\n"
        )

    lines.append("\n**Notes:**\n")
    lines.append("\n- **100M TOPGrowing** rows in the merged CSV: **98** (two scenarios missing vs MaxCov); Wilson intervals use **n=98** for overall detection.\n")
    lines.append(
        "\n- **500M** routing (20260523 eps1 rerun): all three strategies use the **pruned** "
        "`..._fullpool_eps1_6h_pruned.json` placement (**339** stations). TOPGrowing rows were "
        "merged with T19 mop-up CSVs (Julia return-path fix) to reach full yearly fire counts.\n"
    )
    lines.append("\n- **20M / 100M TOPGrowing** come from the **merged** Apr 11–13 bundle (`default` routing driver: reeval 5, horizon 10, 120 s routing cap in the yearly driver, **no** `_final_nature` suffix). **50M TOP** is a standalone default run; **500M** MaxCov, LinearMinTime, and TOPGrowing use **`_final_nature`** with the **pruned** ε=1 placement file (`..._fullpool_eps1_6h_pruned.json`).\n")

    # --- ALERTCalifornia baseline ---
    lines.append("\n## 3. ALERTCalifornia Camera Baseline\n")
    lines.append(
        "\nThe ALERTCalifornia network provides a reference static-sensor baseline: "
        "699 unique camera sites distributed across California, each monitoring a fixed "
        "sector.  A fire is counted as **detected** if and only if its ignition point falls "
        "within a given radius of at least one camera site (Δt = 0 by assumption).  "
        "The sector polygons in `cameras.json` correspond to a uniform zone radius of "
        "**~20 miles (≈ 32 km)**; the table below sweeps four radii to characterise "
        "how detection rate varies with assumed coverage reach.\n"
    )
    lines.append(
        "\n**Data source:** `code/dataset_creation/nature_dataset_creation/data/cameras.json` "
        "(699 unique sites after deduplication by lat/lon).  "
        "**Benchmark fires:** same 100-fire subset as §1–§2 (seed = 42).  "
        "**Wilson 95% score interval** shown in brackets.\n"
    )
    lines.append(
        "\n**Benchmark script:** `code/benchmark_alertcalifornia.py --radius <km>`  "
        "**Coverage plot:** `code/plot_alertcalifornia_coverage.py --radius <km>`\n"
    )

    lines.append("\n### 3.1 Detection rate by assumed coverage radius\n")
    lines.append("\n| Radius | Detected / 100 | Detection rate | 95% CI |\n")
    lines.append("|--------|---------------|----------------|--------|\n")
    try:
        alertca_rows = alertcalifornia_baseline_data()
        for r in alertca_rows:
            radius_label = f"**{r['radius_km']:g} km**"
            if r["radius_km"] == ALERTCA_CANONICAL_KM:
                radius_label += " *(actual)*"
            lines.append(
                f"| {radius_label} | {r['n_detected']} / {r['n_total']} "
                f"| **{r['pct']:.1f}%** "
                f"| [{r['ci_lo']:.1f}%, {r['ci_hi']:.1f}%] |\n"
            )
    except FileNotFoundError as exc:
        lines.append(f"\n*Table unavailable: {exc}*\n")

    lines.append("\n### 3.2 Coverage map (32 km ≈ 20 miles — actual zone radius)\n")
    lines.append(
        "\nBlue circles show the 32 km detection radius around each of the 699 unique "
        "ALERTCalifornia camera sites, overlaid on the masked Pyrologix ignition-probability "
        "map.  Green dots are detected benchmark fires; orange dots are undetected.\n"
    )
    if (OUT / ALERTCA_CANONICAL_IMG).is_file():
        lines.append(f"\n![ALERTCalifornia coverage at 32 km]({ALERTCA_CANONICAL_IMG})\n")
    else:
        lines.append(
            f"\n*Figure not found at `{ALERTCA_CANONICAL_IMG}`. "
            "Run `python code/plot_alertcalifornia_coverage.py --radius 32 "
            f"--out {OUT_REL}/{ALERTCA_CANONICAL_IMG}` to generate it.*\n"
        )

    # Appendix
    lines.append("\n---\n")
    lines.append("\n## Appendix A — Data files in this report folder\n")
    lines.append("\n- **Routing results:** CSVs under `csv/` (listed below).\n")
    lines.append("\n- **Placement plot inputs:** directory `placement_data/` (see `placement_data/MANIFEST.txt` and Appendix C).\n")
    lines.append("\n### CSV files (`csv/`)\n")
    for p in sorted(CSV_DIR.glob("*.csv")):
        lines.append(f"\n- `csv/{p.name}` (full path: `{p.relative_to(ROOT)}`)\n")

    lines.append("\n## Appendix B — Parameters\n")
    lines.append("\n### Placement\n")
    lines.append(
        f"\n| Budget | JSON (read only from `{OUT_REL}/placement_data/logs/` — not `California2021Dataset/logs/`) |\n"
    )
    lines.append("|--------|---------------------------------------------|\n")
    for p in placements:
        lines.append(f"| {p['budget']}M | `{p['json']}` |\n")
    lines.append(
        "\n- **`generate_final_report.py`** never reads `California2021Dataset/logs/`; refresh the bundle by copying solver outputs into **`placement_data/logs/`** if filenames match.\n"
    )
    lines.append("\n- **500M** solve: `report/benchmark_2021_greedy_kernel/supercloud_500M_placement_uniform_fixed_eps1_6h.sh` (ε=1, warm-start from `fullpool_eps05_6h`) → solver output `sensor_alloc_GaussianBudget500M_StationMaxUniformFixedDrones_261x161_mean_fullpool_eps1_6h.json`. Post-hoc pruned layout: `..._fullpool_eps1_6h_pruned.json`. Older script `supercloud_500M_placement_uniform_fixed_ws500filt20_fullpool_6h.sh` corresponds to the superseded `ws500f20` placement.\n")
    lines.append("\n- **500M routing** (`final_nature` for MaxCov, LinearMinTime, TOPGrowing): `report/benchmark_2021_greedy_kernel/supercloud_500M_fnature_routing_maxcov_linear.sh` and `supercloud_500M_fnature_routing_topgrowing.sh` used `--sensor-placement-file` → **solver** `..._fullpool_eps1_6h.json` (**359** stations). For geometry aligned with §1.1 / Figure 4d, point `--sensor-placement-file` at **`..._fullpool_eps1_6h_pruned.json`** (**339** stations) and re-run.\n")

    lines.append("\n### Routing — `final_nature` (MaxCov / LinearMinTime on 20–100M and 500M MaxCov / LinearMinTime / TOPGrowing)\n")
    lines.append("\n- `--no-clustering`\n")
    lines.append("\n- `--reevaluation-step 7`\n")
    lines.append("\n- `--optimization-horizon 7`\n")
    lines.append("\n- `--routing-time-limit 300` (seconds per Gurobi solve for MILP strategies)\n")
    lines.append("\n- `--detection-horizon-data-steps 6`\n")
    lines.append("\n- `--combo-name-suffix _final_nature` and `--routing-log-tag final_nature`\n")

    lines.append("\n### Routing — default TOPGrowing (20M / 100M merged bundle; 50M standalone)\n")
    lines.append("\n- `--no-clustering` in `supercloud_3_greedy_uniform_routing_array.sh` family\n")
    lines.append("\n- Default re-evaluation (**5** operational substeps) and horizon (**10**)\n")
    lines.append("\n- **TOP / PSO** inner cap: **120 s** wall time per PSO call (`max_time` default in `julia/TOP.jl` → `solve_PSO_TOP_multiple_depots`); not the same mechanism as Gurobi `TIME_LIMIT` on MaxCov.\n")

    lines.append("\n### Statistical methods\n")
    lines.append("\n- **Wilson score** interval for binomial proportions (overall detection; detection conditional on discoverable count).\n")
    lines.append("\n- **Student-t** interval for the mean of `delta_t` on detected fires (when *n* ≥ 2).\n")

    lines.append("\n## Appendix C — Figures in this folder\n")
    lines.append(
        "\n**Figure 4** (§1.2): the **four** operational-scale placement PNGs under `images/` "
        "(`california_2021_sensor_clusters_opt_*.png`) — "
        "**4a** 20M, **4b** 50M, **4c** 100M, **4d** 500M (see §1.2 and `docs/FIGURE4.md`). "
        "Collectively they are the final-report **Figure 4**; a lettered panel always maps to the same budget as in §1.2.\n"
    )
    lines.append(
        "\nAll **inputs** to re-create Figure 4 offline are under **`placement_data/`** "
        "(Pyrologix raster, mask, `config_california_2021.json`, full `scenarii/` set, and the four `logs/sensor_alloc_*.json` files). "
        "Step-by-step instructions: **`docs/REPRODUCE_PLOTS.md`**. One-shot shell helper: **`docs/reproduce_placement_plots.sh`** (run from repository root).\n"
    )
    lines.append(
        "\n**Figure 5** (§3.2): `images/alertcalifornia_coverage_32km.png` — "
        "ALERTCalifornia 32 km coverage map on the 100-fire benchmark. "
        "Regenerate with `python code/plot_alertcalifornia_coverage.py --radius 32 "
        f"--out {OUT_REL}/{ALERTCA_CANONICAL_IMG}`.\n"
    )
    lines.append(
        f"\nThis generator also copies matching PNGs from `report/` into `images/` when they exist there. "
        f"To refresh all tables in this Markdown file, run from the repo root: `python3 {OUT_REL}/generate_final_report.py`.\n"
    )

    out_md = OUT / "final_report.md"
    out_md.write_text("".join(lines))
    print(f"Wrote {out_md.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()

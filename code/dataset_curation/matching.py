"""Pure, deterministic scenario-matching logic.

All grid coordinates in this package are ``(row, column)``.  This fixes the
mixed row/column convention in the historical notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FireRecord:
    fire_id: str
    row: int
    col: int
    discovery_date: date | None = None


@dataclass(frozen=True)
class ScenarioRecord:
    scenario_id: str
    ignition_cells: tuple[tuple[int, int], ...]
    start_date: date | None = None


@dataclass(frozen=True)
class Match:
    fire_id: str
    scenario_id: str
    fire_row: int
    fire_col: int
    ignition_row: int
    ignition_col: int
    chebyshev_cells: int
    manhattan_cells: int
    day_difference: int | None


def chebyshev_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return L-infinity distance between two ``(row, col)`` cells."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _seeded_tie(seed: int, fire_id: str, scenario_id: str) -> str:
    # A digest makes tie-breaking stable across Python versions and hash seeds.
    value = f"{seed}\0{fire_id}\0{scenario_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def match_scenarios(
    fires: Sequence[FireRecord],
    scenarios: Sequence[ScenarioRecord],
    *,
    max_distance: int = 5,
    date_aware: bool = False,
    max_day_difference: int = 1,
    seed: int = 0,
    excluded_scenarios: Iterable[str] = (),
) -> tuple[list[Match], list[FireRecord]]:
    """Greedily match fires without scenario reuse.

    Input order does not affect results.  Date-aware matching prioritizes day
    difference, then Chebyshev and Manhattan distance.  Space-only matching
    prioritizes squared Euclidean distance, matching the recovered "closest"
    behavior. Remaining ties use a seeded stable digest and scenario ID.
    """
    used = set(excluded_scenarios)
    matched: list[Match] = []
    unmatched: list[FireRecord] = []
    ordered_fires = sorted(
        fires,
        key=lambda f: (
            f.discovery_date or date.min,
            str(f.fire_id),
            f.row,
            f.col,
        ),
    )
    ordered_scenarios = sorted(scenarios, key=lambda s: s.scenario_id)

    for fire in ordered_fires:
        candidates: list[tuple[tuple[object, ...], ScenarioRecord, tuple[int, int], int | None]] = []
        for scenario in ordered_scenarios:
            if scenario.scenario_id in used:
                continue
            if date_aware:
                if fire.discovery_date is None or scenario.start_date is None:
                    continue
                day_difference = abs((scenario.start_date - fire.discovery_date).days)
                if day_difference > max_day_difference:
                    continue
            else:
                day_difference = None

            for cell in sorted(set(scenario.ignition_cells)):
                chebyshev = chebyshev_distance((fire.row, fire.col), cell)
                if chebyshev > max_distance:
                    continue
                dr, dc = cell[0] - fire.row, cell[1] - fire.col
                manhattan = abs(dr) + abs(dc)
                if date_aware:
                    score: tuple[object, ...] = (
                        day_difference,
                        chebyshev,
                        manhattan,
                        _seeded_tie(seed, fire.fire_id, scenario.scenario_id),
                        scenario.scenario_id,
                        cell,
                    )
                else:
                    score = (
                        dr * dr + dc * dc,
                        chebyshev,
                        manhattan,
                        _seeded_tie(seed, fire.fire_id, scenario.scenario_id),
                        scenario.scenario_id,
                        cell,
                    )
                candidates.append((score, scenario, cell, day_difference))

        if not candidates:
            unmatched.append(fire)
            continue
        _, scenario, cell, day_difference = min(candidates, key=lambda item: item[0])
        used.add(scenario.scenario_id)
        matched.append(
            Match(
                fire_id=fire.fire_id,
                scenario_id=scenario.scenario_id,
                fire_row=fire.row,
                fire_col=fire.col,
                ignition_row=cell[0],
                ignition_col=cell[1],
                chebyshev_cells=chebyshev_distance((fire.row, fire.col), cell),
                manhattan_cells=abs(cell[0] - fire.row) + abs(cell[1] - fire.col),
                day_difference=day_difference,
            )
        )
    return matched, unmatched

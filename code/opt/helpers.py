"""Helper functions for Python optimization models.

Port of ``julia/helper_functions.jl``.  All coordinates are **0-based**.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Burn-map I/O
# ---------------------------------------------------------------------------

def load_burn_map(filename: str) -> np.ndarray:
    """Load a burn map ``.npy`` file.  Returns shape ``(T, N, M)``."""
    arr = np.load(filename)
    if not filename.endswith(".npy") and not filename.endswith(".npz"):
        arr = np.load(filename + ".npy")
    return arr


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def linf_distance(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    """L-infinity (Chebyshev) distance."""
    return max(abs(ai - bi) for ai, bi in zip(a, b))


# ---------------------------------------------------------------------------
# Neighbour helpers
# ---------------------------------------------------------------------------

def linf_neighbors_and_point(
    point: Tuple[int, int],
    feasible: Optional[Set[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Return all L-inf neighbours of *point* **including itself**, filtered by *feasible*."""
    x, y = point
    result: List[Tuple[int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            p = (x + dx, y + dy)
            if feasible is None or p in feasible:
                result.append(p)
    return result


# ---------------------------------------------------------------------------
# Drone reachability
# ---------------------------------------------------------------------------

def get_drone_gridpoints(
    charging_stations: List[Tuple[int, int]],
    radius: int,
    all_points: List[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """Grid points within L-inf *radius* of any charging station."""
    covered: Set[Tuple[int, int]] = set()
    for p in all_points:
        for cs in charging_stations:
            if linf_distance(p, cs) <= radius:
                covered.add(p)
                break
    return covered


def closest_distances(
    stations: List[Tuple[int, int]],
    points: List[Tuple[int, int]],
) -> List[int]:
    """For each point return L-inf distance to the nearest station."""
    return [min(linf_distance(p, s) for s in stations) for p in points]

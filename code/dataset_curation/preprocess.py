"""Canonical mask conversion and empirical burn-map computation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def jpg_sequence_to_array(
    folder: Path,
    threshold: float = 0.5,
    *,
    first_frame_only: bool = False,
) -> np.ndarray:
    """Load JPG masks as canonical binary float32 data."""
    files = sorted(
        (path for path in Path(folder).iterdir() if path.suffix.lower() in {".jpg", ".jpeg"}),
        key=natural_key,
    )
    if not files:
        raise FileNotFoundError(f"No JPG masks found in {folder}")
    frames: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for path in files:
        with Image.open(path) as image:
            normalized = np.asarray(image.convert("L"), dtype=np.float32) / np.float32(255.0)
        if shape is None:
            shape = normalized.shape
        elif normalized.shape != shape:
            raise ValueError(f"{path} has shape {normalized.shape}; expected {shape}")
        frames.append((normalized >= threshold).astype(np.float32))
        if first_frame_only:
            return frames[0]
    return np.stack(frames)


def empirical_burn_map(
    scenarios: Iterable[np.ndarray],
    *,
    noncumulative: bool = False,
    offsets: Iterable[int] | None = None,
) -> np.ndarray:
    """Average scenarios at each represented time, allowing unequal lengths."""
    arrays = [np.asarray(array, dtype=np.float32) for array in scenarios]
    if not arrays:
        raise ValueError("At least one scenario is required")
    spatial_shape = arrays[0].shape[1:]
    if any(array.ndim != 3 or array.shape[1:] != spatial_shape for array in arrays):
        raise ValueError("Scenarios must be T×N×M arrays on one common grid")
    offset_values = list(offsets) if offsets is not None else [0] * len(arrays)
    if len(offset_values) != len(arrays) or any(value < 0 for value in offset_values):
        raise ValueError("Offsets must be one non-negative integer per scenario")

    length = max(offset + array.shape[0] for offset, array in zip(offset_values, arrays))
    total = np.zeros((length, *spatial_shape), dtype=np.float64)
    counts = np.zeros(length, dtype=np.int64)
    for offset, array in zip(offset_values, arrays):
        values = array
        if noncumulative:
            values = np.diff(
                np.concatenate([np.zeros((1, *spatial_shape), dtype=np.float32), array]),
                axis=0,
            )
        stop = offset + len(values)
        total[offset:stop] += values
        # Offset timesteps are represented as pre-ignition zero frames, as in
        # the recovered config-aware aggregation.
        counts[:stop] += 1
    valid = counts > 0
    total[valid] /= counts[valid, None, None]
    return total.astype(np.float32)


def preprocess_layout(
    layout_dir: Path,
    scenario_ids: Iterable[str],
    output_dir: Path,
    *,
    offsets: dict[str, int] | None = None,
    threshold: float = 0.5,
) -> int:
    """Convert selected scenarios and write cumulative/noncumulative means."""
    masks = layout_dir / "Satellite_Images_Mask"
    if not masks.is_dir():
        masks = layout_dir / "Satellite_Image_Mask"
    if not masks.is_dir():
        raise FileNotFoundError(f"No satellite mask directory in {layout_dir}")

    scenario_output = output_dir / layout_dir.name / "scenarii"
    scenario_output.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(scenario_ids))
    arrays: list[np.ndarray] = []
    applied_offsets: list[int] = []
    for scenario_id in ids:
        array = jpg_sequence_to_array(masks / scenario_id, threshold=threshold)
        np.save(scenario_output / f"{scenario_id}.npy", array)
        arrays.append(array)
        applied_offsets.append((offsets or {}).get(f"offset_{scenario_id}", 0))
    if arrays:
        target = output_dir / layout_dir.name
        np.save(target / "burn_map.npy", empirical_burn_map(arrays, offsets=applied_offsets))
        np.save(
            target / "burn_map_noncumulative.npy",
            empirical_burn_map(arrays, noncumulative=True, offsets=applied_offsets),
        )
    return len(arrays)

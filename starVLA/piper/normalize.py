"""Quantile normalization with an invertible constant-dimension convention."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


def save(directory, stats):
    path = Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {field: np.asarray(value).tolist() for field, value in vars(item).items()}
                    for key, item in stats.items()
                }
            },
            indent=2,
        )
        + "\n"
    )


def load(directory):
    raw = json.loads((Path(directory) / "norm_stats.json").read_text())["norm_stats"]
    return {key: NormStats(**{k: np.asarray(v, dtype=np.float64) for k, v in item.items()}) for key, item in raw.items()}


def transform(values, stats, *, inverse=False, clip=False):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != len(stats.q01) or not np.isfinite(values).all():
        raise ValueError("Normalization expects finite values with the configured width")
    center = (stats.q01 + stats.q99) / 2
    # Constant dimensions (especially rotation diagonals) map to zero, not -1.
    scale = np.where(stats.q99 - stats.q01 > 1e-6, (stats.q99 - stats.q01) / 2, 1.0)
    result = values * scale + center if inverse else (values - center) / scale
    if clip:
        result = np.clip(result, -1, 1)
    return result.astype(np.float32)

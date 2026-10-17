"""
Writing results/metrics.json.

The file is the interface between a run and everything that reads it, including
results/report.html, so two properties matter more than convenience:

It is valid JSON. Python's `json` module will happily emit `NaN` and `Infinity`,
which no other JSON parser accepts, so every non-finite value becomes null here.

A metric that could not be computed is distinguishable from one that was never
attempted. An omitted key is ambiguous between the two, so an undefined metric
records null in `metrics` and its reason in `unavailable`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _clean(value):
    """Recursively replace non-finite floats with None, so the JSON is valid."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    # numpy scalars and torch tensors reach here from metric code; unwrap the
    # 0-d ones rather than letting json fail on them later.
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        return _clean(value.item())
    return value


def collect(
    *,
    job_name: str,
    job_type: str,
    dataset: str,
    seed,
    device: str,
    metrics: dict,
    unavailable: dict,
) -> dict:
    """
    Assemble the metrics.json payload.

    `metrics` is flat and uses the key names from CLAUDE.md's tables exactly.
    Anything else goes in `run`, so a reader can tell a metric from a label.
    """
    import datetime

    cleaned = _clean(metrics)

    # Every key whose value came out null needs a reason, or a reader cannot
    # tell "undefined for this run" from "the computation returned nothing".
    for key, value in cleaned.items():
        if value is None and key not in unavailable:
            unavailable[key] = "computed as undefined for this run"

    return {
        "run": {
            "job_name": job_name,
            "job_type": job_type,
            "dataset": dataset,
            "seed": seed,
            "device": device,
            "timestamp": datetime.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
        },
        "metrics": cleaned,
        "unavailable": dict(sorted(unavailable.items())),
    }


def write(payload: dict, results_dir: str) -> Path:
    """Write `payload` to results/metrics.json and return the path."""
    path = Path(results_dir) / "metrics.json"
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n")
    return path


def read(results_dir: str) -> dict:
    """Read back results/metrics.json."""
    return json.loads((Path(results_dir) / "metrics.json").read_text())

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


# The schema the metrics.json envelope is written against. A reader that finds
# a version it does not know should refuse rather than guess.
SCHEMA_VERSION = 1

# Why a metric is null. Three of these were specified; the last three were
# added, and each is flagged for approval because CLAUDE.md fixes the set.
#
# `not_implemented`: a metric that is defined and has no caller. The specified
# three do not cover it, and recording it as "not applicable to this job type"
# would be false.
#
# `not_configured`: the metric is implemented and this experiment did not
# declare the input it needs -- `evaluation_oracle`, `surrogate_training_set`,
# `baseline_results_dir`, `screening_baseline_results`. These were recorded as
# `not_applicable_to_job_type`, which says the metric has no meaning for this
# run when in fact it has one and nobody supplied the input; the two call for
# opposite actions from a reader.
#
# `undefined_for_this_run`: the computation ran and returned nothing, most
# often on an empty surviving set. This was the fallback's old label, so a
# committed smoke metrics.json recorded the implemented, well-tested
# `near_duplicate_rate` as "not_implemented" when the real reason was that no
# molecule reached it.
CATEGORIES = (
    "optional_tool_missing",
    "insufficient_sample",
    "not_applicable_to_job_type",
    "not_implemented",
    "not_configured",
    "undefined_for_this_run",
)


def _categorize(reason: str) -> dict:
    """
    Split a reason string into its category and its text.

    The category is carried as a `"<category>: "` prefix at the point the
    reason is written, so the metric code states it where the knowledge is.
    A reason with no prefix is `undefined_for_this_run`: the computation was
    attempted and came out empty. It used to be `not_implemented`, which
    mislabeled every implemented metric that returned nothing.
    """
    text = " ".join(str(reason).split())
    for category in CATEGORIES:
        if text.startswith(category + ":"):
            return {"category": category, "reason": text[len(category) + 1 :].strip()}
    return {"category": "undefined_for_this_run", "reason": text}


def _versions() -> dict:
    """
    The commit and the library versions every reported number depends on.

    Recorded here because the numbers are version-dependent and the versions
    were nowhere in `metrics.json`: every similarity, SA score and descriptor
    comes from RDKit, the likelihoods from torch, and the commit appeared only
    in `report.html` and only from a clean tree, so `aggregate.json` inherited
    none of it. A number whose toolchain is unrecorded cannot be reproduced,
    and `near_duplicate_rate` and `sa_score_*` in particular move between
    RDKit releases.

    `git` is read-only here, and a value that cannot be obtained is null rather
    than absent, so a reader can tell "not recorded" from "not asked".
    """
    import subprocess

    versions: dict = {}
    for name in ("rdkit", "torch", "numpy", "h5py"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", None))
        except ImportError:
            versions[name] = None

    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        versions["git_commit"] = commit.stdout.strip() or None
        # A dirty tree means the commit does not describe the code that ran,
        # which matters more than the commit itself.
        versions["git_dirty"] = bool(status.stdout.strip()) if status.stdout else None
    except (OSError, subprocess.SubprocessError):
        versions["git_commit"] = None
        versions["git_dirty"] = None

    return versions


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
        "schema_version": SCHEMA_VERSION,
        "run": {
            "job_name": job_name,
            "job_type": job_type,
            "dataset": dataset,
            "seed": seed,
            "device": device,
            "timestamp": datetime.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "versions": _versions(),
        },
        "metrics": cleaned,
        "unavailable": {
            key: _categorize(reason) for key, reason in sorted(unavailable.items())
        },
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

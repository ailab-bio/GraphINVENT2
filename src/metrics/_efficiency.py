"""
Model efficiency, reported for every trained model.

These are measurements of the run rather than of the molecules, so they are
collected by the job and recorded with the device they were measured on. A
wall-clock number without its device is not comparable to anything.
"""

from __future__ import annotations

import platform
import resource


def peak_memory_mb() -> float:
    """
    Peak resident set size of this process, in MB.

    `ru_maxrss` is in bytes on macOS and in kilobytes on Linux, which is the
    kind of difference that silently turns a 2 GB measurement into a 2 MB one.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return float(peak) / divisor


def n_parameters(model) -> int:
    """Total number of parameters, trainable or not."""
    return int(sum(p.numel() for p in model.parameters()))


def inference_s_per_10k(seconds: float, n_molecules: int) -> float | None:
    """
    Generation cost rescaled to 10,000 molecules.

    Linear in the sample size, which holds because generation runs in
    fixed-size batches; the fixed start-up cost is included, so a small
    measurement extrapolates slightly high.
    """
    if not n_molecules:
        return None
    return float(seconds) * 10000.0 / float(n_molecules)

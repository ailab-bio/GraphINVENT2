"""
PMO-style AUC Top-k computation.

Reference: Gao et al., "Sample Efficiency Matters: A Benchmark for Practical
Molecular Optimization", NeurIPS 2022.
"""

from __future__ import annotations

import heapq


def compute_auc_top_k(
    optimization_log: list,
    k: int = 10,
    budget: int = 10000,
    constraints: list = None,
) -> float:
    """
    Compute AUC Top-k over an oracle optimization curve.

    At each oracle call *t*, the running average of the top-*k* scores seen
    so far is computed (considering only molecules that satisfy all constraints,
    if provided).  The AUC is the area under this curve from t=0 to t=budget,
    normalised to [0, 1].

    This matches the primary evaluation metric of the PMO benchmark (Gao et
    al., NeurIPS 2022).

    Parameters
    ----------
    optimization_log : list of (oracle_call_count, score)
        Chronological record of oracle evaluations.  Each entry is a tuple of
        the cumulative oracle call count at the time of evaluation and the
        corresponding score.  Typically obtained from
        :attr:`CachedOracle.optimization_log`.
    k : int
        Number of top molecules to track (default 10).
    budget : int
        Maximum oracle call budget used for normalisation (default 10 000).
    constraints : list of bool, optional
        Per-entry flag indicating whether the molecule satisfies all
        constraints.  If ``None``, all entries are considered valid.
        Entries where ``constraints[i]`` is ``False`` do not count toward the
        top-*k* and are excluded from the curve.

    Returns
    -------
    float
        AUC Top-k in [0, 1].  Returns 0.0 for an empty log.

    Notes
    -----
    *Constrained case*: if fewer than *k* molecules have satisfied all
    constraints at oracle call *t*, the average is taken over however many
    exist (returning 0.0 if none).  This naturally penalises methods that
    waste budget on molecules that fail the constraints.

    *Integration*: trapezoidal rule over the raw call-count axis, anchored at
    t=0 (f=0) and extended to t=budget with the last observed f-value.

    Examples
    --------
    >>> log = [(1, 0.3), (2, 0.7), (3, 0.5), (4, 0.9), (5, 0.8)]
    >>> compute_auc_top_k(log, k=3, budget=5)
    0.6466...

    Constrained example (only even-indexed entries are valid):

    >>> constr = [False, True, False, True, True]
    >>> compute_auc_top_k(log, k=3, budget=5, constraints=constr)
    ...
    """
    if not optimization_log:
        return 0.0

    n = len(optimization_log)
    if constraints is None:
        constraints = [True] * n

    if len(constraints) != n:
        raise ValueError(
            f"constraints length ({len(constraints)}) must match "
            f"optimization_log length ({n})."
        )

    # Sort by oracle call count (should already be sorted, but be safe)
    events = sorted(
        zip(
            [e[0] for e in optimization_log],
            [e[1] for e in optimization_log],
            constraints,
        )
    )

    # Walk through events, maintaining a min-heap of the top-k scores
    top_k: list = []  # min-heap of top-k valid scores
    curve_t: list = [0]
    curve_f: list = [0.0]

    for t, score, valid in events:
        if t > budget:
            break
        if valid:
            if len(top_k) < k:
                heapq.heappush(top_k, score)
            elif score > top_k[0]:
                heapq.heapreplace(top_k, score)

        # Current f(t): average of top-k found so far
        if top_k:
            avg = sum(top_k) / k  # divide by k even if fewer than k found
        else:
            avg = 0.0

        curve_t.append(t)
        curve_f.append(avg)

    # Extend to budget with the last f value (flat extrapolation)
    if curve_t[-1] < budget:
        curve_t.append(budget)
        curve_f.append(curve_f[-1])

    # Trapezoidal integration, normalised by budget
    auc = _trapz(curve_f, curve_t) / budget
    return float(max(0.0, min(1.0, auc)))


def _trapz(y: list, x: list) -> float:
    """Simple trapezoidal integration (avoids NumPy dependency for this module)."""
    total = 0.0
    for i in range(1, len(x)):
        total += (y[i] + y[i - 1]) * (x[i] - x[i - 1]) / 2.0
    return total

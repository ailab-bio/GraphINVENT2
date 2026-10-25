# protac_selectivity

*Last updated: October 25th, 2026.*

## What this demonstrates

**The claim is methodological, not pharmacological.** It is that a two-objective margin — maximize predicted
SMARCA2 degradation, minimize predicted SMARCA4 degradation, on both pDC50 and Dmax — can be optimized in this
framework, and it reports what the margin did: its trajectory against oracle calls, and the margins of the top
100 molecules. **This is not a selectivity result and must never be written as one.**

The reason is in the data, and it is not a hedge. SMARCA2 and SMARCA4 are the right pair on the chemistry —
paralogous SWI/SNF ATPases whose bromodomains are close enough that warheads cross-react, with SMARCA2 the
synthetic-lethal dependency in SMARCA4-deficient tumors and SMARCA4 degradation the liability to avoid — and they
are the only pair in TPDdb with enough labels, at 186 and 123 compounds with a numeric DC50. But measured
2026-10-25, **all 123 of SMARCA4's DC50 compounds also carry a SMARCA2 DC50**: its set is a strict subset,
dual-annotated from the same papers, with no independent SMARCA4-only series anywhere in the database.

That cuts both ways and both directions matter. It is *good* for fitting the margin, because both values come
from the same compounds in the same assays, which is exactly what a paired delta needs. It is *fatal* to a
generalization claim, because there is no held-out counter-screen against which the margin could be shown to
hold. Annotation presence also partly records who counter-screened rather than which molecules are selective, and
the 64 right-censored SMARCA4 rows are the most selective compounds, so a fit that drops them truncates the label
exactly where the signal is. The fit is censoring-aware for that reason.

**So what the manuscript may say:** that the framework optimizes a two-objective margin, with the margin's
trajectory and top-100 distribution as the result. **What it may not say:** that the molecules are selective, or
that the margin transfers to compounds the paired corpus does not cover. Validating selectivity would need
counter-screening data that does not exist here.

## Prerequisites

`protac_transfer/jobs/scratch_control`, and a paired surrogate over both endpoints that does not exist yet. Per
the settled design, the pooled ensemble oracles for pDC50 and Dmax cover every target including SMARCA2 and
SMARCA4, so the margin is computed from those rather than from a separate delta model.

## Submit

    python submit.py experiments/protac_selectivity/jobs/rl_selectivity --slurm

## What to check

`results/metrics.json`: the trade-off front, and the fraction of the top 100 whose predicted delta exceeds the
censoring boundary, which are out-of-domain by construction.

## Core metrics come from

`generate_best_agent`. Compared against virtual screening only, never against REINVENT4.

## Failure mode

A front that collapses onto maximum SMARCA2 with no SMARCA4 discrimination means the paired labels carry no
usable signal, which the 0% SMARCA4-selective observation already predicts.

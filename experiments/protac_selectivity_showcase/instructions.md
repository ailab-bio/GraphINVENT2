# protac_selectivity_showcase

*Last updated: October 18th, 2026.*

## What this demonstrates

A two-objective trade-off between predicted SMARCA2 and SMARCA4 degradation, over the 123 matched (compound, cell
line) pairs that have a numeric DC50 for both. **This is not a selectivity result and must never be written as
one.** Annotation presence partly records who counter-screened rather than which molecules are selective, and the
64 right-censored SMARCA4 rows are the most selective compounds, so a fit that drops them truncates the label
exactly where the signal is. The fit is censoring-aware for that reason.

## Prerequisites

`protac_unconditional/jobs/prior_transfer`, and a paired delta-pDC50 surrogate that does not exist yet. Whether
the endpoint is paired delta-pDC50 over the 123 matched pairs is **question 6 in the report**; this directory
drafts it that way.

## Submit

    python submit.py experiments/protac_selectivity_showcase/jobs/rl_selectivity --slurm

## What to check

`results/metrics.json`: the trade-off front, and the fraction of the top 100 whose predicted delta exceeds the
censoring boundary, which are out-of-domain by construction.

## Core metrics come from

`generate_best_agent`. Compared against virtual screening only, never against REINVENT4.

## Failure mode

A front that collapses onto maximum SMARCA2 with no SMARCA4 discrimination means the paired labels carry no
usable signal, which the 0% SMARCA4-selective observation already predicts.

# SMARCA2 pDC50 surrogate

*Last updated: October 26th, 2026.*

The reward for `experiments/protac_goal_directed`. Refit 2026-10-19 after three
defects in the trainer were corrected; the numbers below supersede anything
reported earlier.

## Data

From `scripts/tpddb-activity-labels.py --target SMARCA2`, which is the single
source of pDC50 in this project and writes
`data/surrogates/smarca2_pdc50_PROVENANCE.json` beside the labels.

| | |
|---|---|
| Activity records for SMARCA2 | 1,567 for AR by comparison; SMARCA2 has 523 |
| Rows reporting a degradation concentration | 243 |
| Numeric with a unit, kept | 241 |
| Censored, dropped | 2 |
| Letter grades | 0 |
| Distinct compounds | 186 |
| Distinct cell lines | 3 (SW1573, HT-1080, A549) |

Endpoint is pDC50 = -log10(DC50 in molar). Replicates of one (compound, cell
line) collapse to the geometric mean of DC50. Censored rows are dropped, which
truncates the label distribution at the inactive end, so the retained range is
the model's applicability window rather than the whole potency range.

## Held-out performance, against the mean-prediction baseline

Random forest over a 2048-bit **count** Morgan fingerprint, nine
physicochemical descriptors and one-hot cell line and ligase: 2,061 features.
Split by **warhead scaffold**, which is the only split that means anything on a
bifunctional corpus, 194 train against 47 test with no warhead scaffold shared.

| | surrogate | mean-prediction baseline |
|---|---|---|
| RMSE | **0.488** | 0.639 |
| margin over baseline | **0.151** [95% bootstrap 0.017, 0.287] | |
| R² against the mean-prediction baseline | **0.416** | 0.0 by construction |
| R² against the test mean (`r2_score`) | 0.342 | |
| Spearman | 0.567 | undefined |
| MAE | 0.386 | |

**It beats the baseline, so the goal-directed arm is not killed** — but by a
margin whose lower bound nearly touches zero, and the bar it clears is not the
one that matters. Three qualifications, added 2026-10-26 after the trainer
learned to report an interval and two R² values:

**The two R² values are against different baselines and were conflated.** The
kill criterion is stated against the mean-prediction baseline, which is the
*training* mean, because that is the only mean a model has at fit time; sklearn's
`r2_score` measures variance explained against the *test* mean, which nothing
could have known. They differ by 0.074 here, and the 0.342 quoted in earlier
versions of this file and in the manuscript is the second one under the first
one's name.

**The margin's interval nearly includes zero.** Resampling the 47 held-out
residuals 2,000 times puts the RMSE margin at 0.151 with a 95% interval of
[0.017, 0.287]. The interval is not clustered by compound — 47 rows cover fewer
than 47 compounds — so the true interval is wider than this one. A whole
optimization allocation rests on a margin whose lower bound is 0.017 log units.

**That lower bound is far inside the label's own noise.** The pooled
within-compound, across-cell-line SD of this endpoint is **0.405 log units** (55
compounds with more than one row), and across the raw TPDdb table 49% of
repeated (compound, target) measurements differ by more than 0.5 log. So the
entire margin of 0.151 log is smaller than the assay-to-assay variability of the
quantity being predicted, and R² should be read against the achievable ceiling
that noise implies: against a total label SD of 0.851 on this table, an
irreducible 0.405 caps R² at **0.774**, not at 1.0. Measured on the committed
label file, reproducible in one pass over it. A Spearman of 0.567 on
warhead-disjoint compounds is still
real ranking signal; it is ranking signal measured against a noisy ruler.

**Still outstanding, and each needs a run rather than an edit.** There is no
cross-validation: one 80/20 split at one seed, so the ensemble-disagreement
reward-hacking detector `CLAUDE.md` specifies has no folds to disagree, and any
"standard deviation over seeds" for this panel has no referent. The three
configurations tabulated below were compared across *different* test sets, so
their R² values are not comparable to each other. Refit at 5–10 split seeds and
add grouped k-fold over warhead scaffolds.

What it does not license. The absolute RMSE is half a log unit, so this ranks
candidates and does not predict potency. Degradation depends on ternary-complex
cooperativity that no structure-only model sees, the labels pool three cell
lines, and the fingerprint cannot distinguish linker homologues: two real
degraders differing by one methylene score a binary Tanimoto of 1.000 even at
65,536 bits, which is why the count fingerprint is used here and is still only
a partial answer.

## For comparison, the splits and featurizations that were wrong

| configuration | split | RMSE | baseline | R² | Spearman |
|---|---|---|---|---|---|
| fingerprint bits plus descriptors | whole-molecule scaffold | 0.488 | 0.579 | 0.281 | 0.519 |
| plus count fingerprint, cell line, ligase | whole-molecule scaffold | 0.470 | 0.579 | 0.334 | 0.527 |
| **the same, warhead scaffold** | **warhead scaffold** | **0.488** | **0.639** | **0.342** | **0.567** |

The whole-molecule split flatters the apparent RMSE and understates the
baseline, because near analogues sit on both sides of it.

## The pair

`optimization.pickle` is fitted on the split above and is the reinforcement
learning reward. `evaluation.pickle` is the same featurization and model class
refitted on all 241 rows and only ever scores final molecules. A large gap
between them on optimized molecules, when they agree on held-out real
molecules, is evidence of exploitation rather than success.

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
| margin over baseline | **0.151** [95% bootstrap 0.003, 0.295], clustered on compound | |
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

**The margin's interval all but includes zero.** Resampling 2,000 times puts
the RMSE margin at 0.151 with a 95% interval of **[0.003, 0.295]**. The
resample is over held-out **compounds**, 45 of them covering the 47 rows, not
over rows: a compound measured in three cell lines is three rows and one piece
of evidence, and treating correlated rows as independent narrows the interval
in exactly the direction the decision depends on. An earlier version of this
file reported [0.017, 0.287] from an unclustered resample; clustering moves the
lower bound from 0.017 to **0.003 log units**, which is to say the margin is
not distinguishable from zero at this sample size. A whole optimization
allocation rests on that bound.

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

## Grouped cross-validation

Run 2026-10-26. `scripts/train-surrogate.py --cv grouped` holds out whole
groups, reports what the scheme estimates rather than a bare "held-out R²", and
keeps the fold models in the pickle. The full log is `pdc50.log` beside this
file. An earlier version of this section said no cross-validation existed and
asked for this run; this is it.

**Scheme one: five folds grouped by warhead scaffold.** This estimates the
error on novel molecules for a target the model has seen, which is what the
reward actually does. 72 groups over 241 rows.

| fold | n | RMSE | baseline | R² vs mean-prediction | Spearman | groups held out |
|---|---|---|---|---|---|---|
| 1 | 104 | 0.725 | 1.023 | +0.498 | +0.746 | 1 |
| 2 | 35 | 0.604 | 0.651 | +0.137 | +0.439 | 15 |
| 3 | 34 | 0.414 | 0.889 | +0.784 | +0.821 | 18 |
| 4 | 34 | 0.432 | 0.746 | +0.664 | +0.772 | 18 |
| 5 | 34 | 0.501 | 0.812 | +0.619 | +0.827 | 20 |
| **mean (SD)** | | **0.535 (0.130)** | | **+0.540 (0.248)** | | |

All five folds beat their own mean-prediction baseline, so by the stated kill
criterion the reward survives on this scheme. Two things to read off it:

- **The single split's 0.416 was never a number to quote alone.** The folds
  range from R² 0.137 to 0.784, an SD of 0.248 around a mean of 0.540. Any
  statement of this surrogate's quality to two decimal places is overprecision;
  the honest form is 0.54 ± 0.25.
- **One warhead scaffold is 104 of the 241 rows**, 43% of the table, which is
  why fold 1 holds out a single group. The corpus is one dominant congeneric
  series plus a long singleton tail (median group size 1), and that shape, not
  the model, is what the fold spread mostly reflects.

Against the 0.774 label-noise ceiling, the mean R² of 0.540 is **70% of the
achievable variance** — the model is closer to the noise floor than the raw
number suggests.

**Scheme two: five folds grouped by protein, on the pooled table.** This
estimates the error for a target the model has **never** seen, which is the
quantity any held-out-target claim needs. It cannot be computed from a
per-target fit at all, so it is run on the pooled multi-target table
(`data/surrogates/pooled_pdc50.tsv`, 1,584 rows over 18 canonical POIs, built
by `--target all`). Full numbers in `../pooled/pdc50.log`.

| | |
|---|---|
| mean R² across the five POI-grouped folds | **−0.150 (SD 0.159)** |
| folds beating their mean-prediction baseline | **1 of 5** |
| mean RMSE | 0.839 (SD 0.173) |
| the same table under a random split | R² 0.613 |

**This is a failed fit and is reported as one.** Four of the five folds are
worse than predicting the training mean, so the pooled surrogate has no
demonstrated ability to score a target it has not seen. Consequences, because
this one bears on an experiment that is designed and not yet run:

- `CLAUDE.md` gives POI-grouped CV as the estimate that lets
  `<program>_conditional_esm2` interpret its held-out-target results. The
  estimate now exists and it says the oracle cannot score an unseen target, so
  a held-out-target result cannot be scored through this surrogate. That has to
  be settled before that experiment means anything.
- The gap between R² 0.613 under a random split and −0.150 under POI grouping
  is what target leakage looks like from the inside. The random split puts
  analogues and replicate rows of the same compound on both sides; it is quoted
  here only as the contrast.
- Per-target RMSE varies more than threefold across the 18 proteins, from 0.307
  on PARP1 (n=12) to 1.717 on KAT2A (n=13). A pooled number hides that, which
  is why the per-target table is printed.

The pooled fit is **not** wired into any reward: no `goal_directed` config names
its pickle, and it is stored without its fold models since it is reported rather
than used.

**The fold models are the reward-hacking detector.** `pdc50.pickle` now carries
the five warhead-scaffold fold models beside the full-data fit, and
`SklearnOracle` reports their per-molecule spread as the uncertainty, recording
`uncertainty_source: cv_fold_models`. That is disagreement between models fitted
on *different rows*, which is evidence a molecule sits outside the data. The
previous signal was the spread across one forest's 200 trees, which all saw the
same rows: on the same four probe molecules the fold spread is 0.26–0.37 log and
the tree spread 0.92–1.08, and the two are not comparable, so nothing should
average or substitute them. Keeping the folds costs artifact size: 3.5 MB to
**19.9 MB**.

**Still outstanding.** The three configurations tabulated below were compared
across *different* test sets, so their R² values are not comparable to each
other; they are kept as a record of what the wrong split looked like. A
multi-seed refit (5–10 split seeds) would separate split variance from fold
variance, which the grouped CV above conflates into one spread.

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

`pdc50.pickle` is fitted on the split above and is the reinforcement learning
reward; it supersedes `optimization.pickle`, the superseded name from the
two-surrogate design, and `experiments/protac_goal_directed/experiment.yaml`
names it. `evaluation.pickle` is the same featurization and model class refitted
on all 241 rows and only ever scores final molecules. A large gap between them
on optimized molecules, when they agree on held-out real molecules, is evidence
of exploitation rather than success.

`CLAUDE.md` replaces this pair with **one pooled ensemble** whose fold
disagreement is the detector, which removes the gap there is to read. Half of
that is now in place: the fold models exist and the oracle reports their
disagreement. The other half — making the pooled fit the reward — is not done,
and the POI-grouped result above is the reason to think twice about it rather
than a reason to hurry: switching the reward would change every goal-directed
number, and the pooled model's only demonstrated advantage is on a split that
leaks. It is a one-line change to `oracles.path` when that is settled.

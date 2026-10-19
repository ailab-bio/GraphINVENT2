# SMARCA2 pDC50 surrogate

*Last updated: October 19th, 2026.*

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
| R² | 0.342 | 0.0 by construction |
| Spearman | 0.567 | undefined |
| MAE | 0.386 | |

**It beats the baseline, so the goal-directed arm is not killed.** A Spearman of
0.567 on warhead-disjoint held-out compounds is real ranking signal rather than
memorization of a congeneric series.

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

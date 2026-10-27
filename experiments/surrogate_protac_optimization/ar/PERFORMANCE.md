# ar

Fitted 2026-10-26 by `experiments/surrogate_protac_optimization/fit.sh`. Every number below comes from the `.log` beside this file for that endpoint; nothing is copied from another run.

**pdc50 is a failed fit and must not be used as a reward.** `CLAUDE.md`: a fit that fails to beat its baseline is reported as a failed fit rather than used as a reward. A fold that does not beat its baseline is not averaged away into a healthy-looking mean; it is a region of this target's chemistry where the model knows nothing, and a reward built on it optimizes noise there.

The shipped `model` in each pickle is refitted on **every** row, so the error estimate comes from the grouped folds and not from the single split. The fold models travel in the pickle as the ensemble-disagreement reward-hacking detector `CLAUDE.md` specifies.

The bootstrap below resamples whole **compounds**, not residual pairs: one compound supplies several rows across cell lines, and resampling rows treats correlated measurements as independent and gives an interval that is too narrow.

## `dmax`

**Clears the kill criterion.**

5 of 5 grouped folds beat their own mean-prediction baseline. Across folds, RMSE 0.136 (0.036)   margin 0.096 (0.051)   R^2 +0.622 (0.152); the parenthesized figure is the sample standard deviation over folds.

502 rows after parsing, censoring and deduplication.

Cross-validation grouped by `warhead_scaffold`, which estimates the error on novel molecules for a target the model has already seen, which is what the reward does. 66 groups over 502 rows, the largest holding 115.

```
fold 1: n=  115  RMSE 0.120 (baseline 0.154)  R^2 +0.397  rho +0.290  (1 groups held out)
fold 2: n=   97  RMSE 0.098 (baseline 0.200)  R^2 +0.759  rho +0.291  (13 groups held out)
fold 3: n=   97  RMSE 0.196 (baseline 0.366)  R^2 +0.713  rho +0.711  (15 groups held out)
fold 4: n=   97  RMSE 0.136 (baseline 0.200)  R^2 +0.535  rho +0.541  (18 groups held out)
fold 5: n=   96  RMSE 0.130 (baseline 0.239)  R^2 +0.704  rho +0.803  (19 groups held out)
across folds: RMSE 0.136 (0.036)   margin 0.096 (0.051)   R^2 +0.622 (0.152)
folds beating the mean-prediction baseline: 5 of 5
```

Replicate measurements of one (compound, target) pair disagree enough to cap R^2 for this label at **0.432**, not 1.0, so the fold R^2 is read against that rather than against a perfect fit.
The fold R^2 **exceeds** that ceiling, which means the ceiling is underestimated rather than that the fit beat it: on a bounded saturating label many replicates sit at the same clipped value and agree for that reason rather than because the assay is precise, so the within-group SD understates the real assay error. Read this fit against the fold spread.

Single split, for continuity with the pre-CV numbers: margin over the mean-prediction baseline 0.036 [95% bootstrap 0.009, 0.063], clustered on compound. Verdict on that split: YES

## `pdc50`

**FAILED FIT.**

3 of 5 grouped folds beat their own mean-prediction baseline. Across folds, RMSE 0.571 (0.109)   margin 0.045 (0.128)   R^2 +0.061 (0.356); the parenthesized figure is the sample standard deviation over folds.

344 rows after parsing, censoring and deduplication.

Cross-validation grouped by `warhead_scaffold`, which estimates the error on novel molecules for a target the model has already seen, which is what the reward does. 35 groups over 344 rows, the largest holding 97.

```
fold 1: n=   97  RMSE 0.494 (baseline 0.552)  R^2 +0.199  rho +0.451  (1 groups held out)
fold 2: n=   74  RMSE 0.490 (baseline 0.738)  R^2 +0.559  rho +0.756  (1 groups held out)
fold 3: n=   58  RMSE 0.755 (baseline 0.803)  R^2 +0.115  rho +0.599  (9 groups held out)
fold 4: n=   58  RMSE 0.572 (baseline 0.505)  R^2 -0.284  rho +0.423  (12 groups held out)
fold 5: n=   57  RMSE 0.543 (baseline 0.479)  R^2 -0.282  rho +0.171  (12 groups held out)
across folds: RMSE 0.571 (0.109)   margin 0.045 (0.128)   R^2 +0.061 (0.356)
folds beating the mean-prediction baseline: 3 of 5
```

Replicate measurements of one (compound, target) pair disagree enough to cap R^2 for this label at **0.679**, not 1.0, so the fold R^2 is read against that rather than against a perfect fit.
This fit reaches 9% of the achievable variance.

Single split, for continuity with the pre-CV numbers: margin over the mean-prediction baseline 0.030 [95% bootstrap -0.015, 0.099], clustered on compound. Verdict on that split: NOT ESTABLISHED -- the point margin is positive but the 95% interval includes zero, so this fit does not clear the kill criterion on this split

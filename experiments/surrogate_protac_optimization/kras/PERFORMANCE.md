# kras

Fitted 2026-10-26 by `experiments/surrogate_protac_optimization/fit.sh`. Every number below comes from the `.log` beside this file for that endpoint; nothing is copied from another run.

**dmax, pdc50 are failed fits and must not be used as a reward.** `CLAUDE.md`: a fit that fails to beat its baseline is reported as a failed fit rather than used as a reward. A fold that does not beat its baseline is not averaged away into a healthy-looking mean; it is a region of this target's chemistry where the model knows nothing, and a reward built on it optimizes noise there.

The shipped `model` in each pickle is refitted on **every** row, so the error estimate comes from the grouped folds and not from the single split. The fold models travel in the pickle as the ensemble-disagreement reward-hacking detector `CLAUDE.md` specifies.

The bootstrap below resamples whole **compounds**, not residual pairs: one compound supplies several rows across cell lines, and resampling rows treats correlated measurements as independent and gives an interval that is too narrow.

## `dmax`

**FAILED FIT.**

4 of 5 grouped folds beat their own mean-prediction baseline. Across folds, RMSE 0.061 (0.040)   margin 0.009 (0.034)   R^2 +0.084 (1.113); the parenthesized figure is the sample standard deviation over folds.

71 rows after parsing, censoring and deduplication.

Cross-validation grouped by `warhead_scaffold`, which estimates the error on novel molecules for a target the model has already seen, which is what the reward does. 13 groups over 71 rows, the largest holding 30.

```
fold 1: n=   30  RMSE 0.031 (baseline 0.047)  R^2 +0.556  rho -0.032  (1 groups held out)
fold 2: n=   14  RMSE 0.019 (baseline 0.032)  R^2 +0.671  rho +nan  (1 groups held out)
fold 3: n=   11  RMSE 0.120 (baseline 0.071)  R^2 -1.905  rho -0.023  (1 groups held out)
fold 4: n=    8  RMSE 0.071 (baseline 0.105)  R^2 +0.544  rho +0.761  (5 groups held out)
fold 5: n=    8  RMSE 0.066 (baseline 0.099)  R^2 +0.555  rho +0.878  (5 groups held out)
across folds: RMSE 0.061 (0.040)   margin 0.009 (0.034)   R^2 +0.084 (1.113)
folds beating the mean-prediction baseline: 4 of 5
```

Single split, for continuity with the pre-CV numbers: margin over the mean-prediction baseline 0.006 [95% bootstrap -0.001, 0.015], clustered on compound. Verdict on that split: NOT ESTABLISHED -- the point margin is positive but the 95% interval includes zero, so this fit does not clear the kill criterion on this split

## `pdc50`

**FAILED FIT.**

4 of 5 grouped folds beat their own mean-prediction baseline. Across folds, RMSE 0.389 (0.105)   margin 0.070 (0.077)   R^2 +0.256 (0.265); the parenthesized figure is the sample standard deviation over folds.

89 rows after parsing, censoring and deduplication.

Cross-validation grouped by `warhead_scaffold`, which estimates the error on novel molecules for a target the model has already seen, which is what the reward does. 19 groups over 89 rows, the largest holding 31.

```
fold 1: n=   31  RMSE 0.295 (baseline 0.319)  R^2 +0.143  rho +0.432  (1 groups held out)
fold 2: n=   15  RMSE 0.464 (baseline 0.495)  R^2 +0.121  rho +0.188  (2 groups held out)
fold 3: n=   15  RMSE 0.502 (baseline 0.495)  R^2 -0.029  rho +0.272  (4 groups held out)
fold 4: n=   14  RMSE 0.263 (baseline 0.437)  R^2 +0.638  rho +0.427  (5 groups held out)
fold 5: n=   14  RMSE 0.419 (baseline 0.545)  R^2 +0.409  rho +0.746  (7 groups held out)
across folds: RMSE 0.389 (0.105)   margin 0.070 (0.077)   R^2 +0.256 (0.265)
folds beating the mean-prediction baseline: 4 of 5
```

Single split, for continuity with the pre-CV numbers: margin over the mean-prediction baseline 0.080 [95% bootstrap -0.028, 0.190], clustered on compound. Verdict on that split: NOT ESTABLISHED -- the point margin is positive but the 95% interval includes zero, so this fit does not clear the kill criterion on this split

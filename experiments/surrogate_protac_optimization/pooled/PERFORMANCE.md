# pooled

Fitted 2026-10-26 by `experiments/surrogate_protac_optimization/fit.sh`. Every number below comes from the `.log` beside this file for that endpoint; nothing is copied from another run.

**pdc50 is a failed fit and must not be used as a reward.** `CLAUDE.md`: a fit that fails to beat its baseline is reported as a failed fit rather than used as a reward. A fold that does not beat its baseline is not averaged away into a healthy-looking mean; it is a region of this target's chemistry where the model knows nothing, and a reward built on it optimizes noise there.

The shipped `model` in each pickle is refitted on **every** row, so the error estimate comes from the grouped folds and not from the single split. The fold models travel in the pickle as the ensemble-disagreement reward-hacking detector `CLAUDE.md` specifies.

The bootstrap below resamples whole **compounds**, not residual pairs: one compound supplies several rows across cell lines, and resampling rows treats correlated measurements as independent and gives an interval that is too narrow.

## `pdc50`

**FAILED FIT.**

1 of 5 grouped folds beat their own mean-prediction baseline. Across folds, RMSE 0.839 (0.173)   margin -0.052 (0.058)   R^2 -0.150 (0.159); the parenthesized figure is the sample standard deviation over folds.

1584 rows after parsing, censoring and deduplication.

Cross-validation grouped by `poi`, which estimates the error on a target the model has never seen, which is what `<program>_conditional_esm2` needs in order to interpret its held-out target. 18 groups over 1584 rows, the largest holding 344.

```
fold 1: n=  344  RMSE 0.673 (baseline 0.622)  R^2 -0.172  rho +0.251  (1 groups held out)
fold 2: n=  310  RMSE 0.700 (baseline 0.611)  R^2 -0.313  rho -0.112  (2 groups held out)
fold 3: n=  308  RMSE 1.011 (baseline 0.905)  R^2 -0.247  rho +0.564  (4 groups held out)
fold 4: n=  309  RMSE 0.775 (baseline 0.817)  R^2 +0.102  rho +0.028  (5 groups held out)
fold 5: n=  313  RMSE 1.036 (baseline 0.980)  R^2 -0.118  rho +0.025  (6 groups held out)
across folds: RMSE 0.839 (0.173)   margin -0.052 (0.058)   R^2 -0.150 (0.159)
folds beating the mean-prediction baseline: 1 of 5
```

Replicate measurements of one (compound, target) pair disagree enough to cap R^2 for this label at **0.755**, not 1.0, so the fold R^2 is read against that rather than against a perfect fit.

Single split, for continuity with the pre-CV numbers: margin over the mean-prediction baseline 0.296 [95% bootstrap 0.221, 0.378], clustered on compound. Verdict on that split: YES

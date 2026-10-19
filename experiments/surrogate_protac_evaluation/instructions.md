# PROTAC evaluation surrogate

*Last updated: October 19th, 2026.*

## What this experiment claims

Nothing on its own. It exists to rescore the molecules an optimization campaign
produced, with a model that has seen strictly more data than the one that
rewarded them.

**It is never used for optimization, and its pickle is never named in a
`goal_directed` config's `oracles:` block.** Feeding it back into a run would
destroy the only evidence it provides.

## Run it

```bash
bash experiments/surrogate_protac_evaluation/fit.sh
```

Same featurization, same model class, same targets and endpoints as
`experiments/surrogate_protac_optimization`. The one difference is what it is
allowed to see: all the data, including the test split.

## How it is read

`eval_oracle_top10_mean` in a goal-directed `metrics.json` is the top-10
molecules of that run rescored here. Compare it with `top10_mean_at_10k` from
the optimization surrogate:

- **The two agree on optimized molecules.** The optimization surrogate was not
  exploited; the campaign found molecules both models like.
- **A large gap, and the two agree on held-out real molecules.** The campaign
  found a hole in the optimization surrogate rather than good chemistry. This
  is the failure the pair exists to detect, and it is the common one.
- **A large gap, and the two disagree on held-out real molecules too.** The
  refit changed the model, so the comparison says nothing about the campaign.
  Fix the surrogate before reading the campaign.

`top100_outside_domain_rate` catches the same failure from a direction neither
model shares, since both have the same blind spots.

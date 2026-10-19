# Surrogates

*Last updated: October 19th, 2026.*

One directory per surrogate. These are not GraphINVENT jobs: nothing here is
submitted with `submit.py`. Each directory holds a `fit.sh` that is the exact
command, a `PERFORMANCE.md` recording held-out performance against the
mean-prediction baseline, and the pickled model the `oracles:` block of a
goal-directed config points at.

**Two surrogates per endpoint, and they are not interchangeable.** The
*optimization* surrogate is fitted on the training and validation splits only
and provides the reinforcement-learning reward. The *evaluation* surrogate is
the same featurization and model class refitted on all the data including the
test split, and it only ever scores final molecules. Both are reported. A large
gap on optimized molecules, when the two agree on held-out real molecules, is
evidence of exploitation rather than success.

**The held-out number is a kill criterion, not a diagnostic.** A surrogate that
does not beat predicting the training mean cannot drive an optimization run: it
will still produce molecules that score well and mean nothing. The number goes
in `PERFORMANCE.md` before any goal-directed job that uses it is submitted.

**The split is the whole argument.** A whole-molecule Bemis-Murcko split on a
bifunctional corpus is a random split with extra steps, because the scaffold is
nearly unique per compound. Use `--split warhead_scaffold` for degraders, which
groups by the scaffold of the target-binding component.

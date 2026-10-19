# PROTAC optimization surrogate

*Last updated: October 19th, 2026.*

## What this experiment claims

That a model fitted on TPD-DB degrader activity predicts a held-out degrader's
pDC50 and Dmax better than predicting the training mean. It is not a GraphINVENT
job and nothing here is submitted with `submit.py`.

**Its held-out number is read first in `experiments/protac_goal_directed` and
in `experiments/protac_selectivity`, because it is the kill criterion for
both.** A surrogate that does not beat the mean cannot drive an optimization
run: the molecules it rewards would be the ones it happens to score high, and
the campaign would optimize an arbitrary function convincingly. Publish the
scatter and the baseline either way, including when the answer is that the
surrogate fails.

## Run it

```bash
bash experiments/surrogate_protac_optimization/fit.sh
```

## What it fits, and what it is allowed to see

**The training and validation splits only. No test molecule is anywhere in
it.** That is the entire reason this is a separate experiment from
`surrogate_protac_evaluation`, which is the same featurization and model class
refitted on all the data. The two exist to be compared: a large gap on
optimized molecules, when they agree on held-out real molecules, is evidence of
exploitation rather than success.

Three targets and two endpoints each:

| Target | pDC50 compounds | Dmax compounds |
|---|---|---|
| SMARCA2 | 186 | 152 |
| AR | 179 | 324 |
| BTK | 189 | 57 |

Measured 2026-10-19 from the TPD-DB tables. **BTK Dmax at 57 compounds is
marginal** under a warhead-scaffold split and is expected to produce a held-out
R² near zero; decide in advance that you will report it rather than drop it.

Potency and extent are different questions and are fitted separately. A
degrader can be potent and still only ever remove half the protein, so an
optimization that reads one as the other chases the wrong molecule.

**Held out by warhead scaffold.** A whole-molecule Bemis-Murcko scaffold of a
PROTAC is nearly unique per compound, so a scaffold split degenerates into a
random one and the held-out number becomes an analogue lookup.

## What it looks like when it has gone wrong

- Held-out R² at or below zero: the surrogate does not beat the mean. That is
  the kill criterion, and it is a finding rather than a setback.
- A test set in the single digits after the split: the target has too few
  distinct warhead scaffolds. `scripts/train-surrogate.py` exits loudly on an
  empty test set; a test set of three is worse, because it produces a number.
- An R² far above 0.6 on this data size: check for leakage between the splits
  before believing it.

# Molecular glue evaluation surrogate

*Last updated: October 19th, 2026.*

## What this experiment claims

It would be the oracle that rescores final glue molecules. **It cannot be fitted from TPD-DB as it stands, and that is
the finding.**

Measured 2026-10-19 with `scripts/tpddb-activity-labels.py --modality glue`
over every neosubstrate the conditional panel names:

| Target | pDC50 compounds | Dmax compounds |
|---|---|---|
| GSPT1 | 30 | 30 |
| IKZF1 | 5 | 5 |
| IKZF3 | 1 | 0 |
| RBM39 | 0 | 0 |

The floor in `manuscript/experiment-plan.md` is about a hundred compounds. The
largest glue target has thirty, and held out by scaffold that leaves a test set
in the single digits. A held-out number computed on six molecules is not an
estimate of anything, and reporting one would be worse than reporting none.

**By the kill criterion this directory's PROTAC counterpart states, a surrogate
that cannot beat the training mean cannot drive an optimization run.** So
`experiments/mg_goal_directed` has no learned reward available from TPD-DB,
and what it optimizes against has to be decided rather than assumed.

## Run it anyway

```bash
bash experiments/surrogate_mg_evaluation/fit.sh
```

It reproduces the counts above in one command and works unchanged the day a
larger glue activity set exists. It is deliberately not named in any
goal-directed config.

## Why these are separate from the PROTAC surrogates

Not economy, and not symmetry. Four reasons, each sufficient on its own:

- **The mechanism differs.** A PROTAC presents a defined E3 ligand and the
  relevant categorical variable is which ligase it recruits. A glue has no
  ligand-shaped recruiter, and the relevant variable is the neosubstrate it
  induces.
- **The size regimes differ by more than a factor of two**, so Morgan bit
  density differs enormously and a pooled model would be dominated by molecular
  size rather than by chemistry.
- **The data volumes differ by an order of magnitude**, so pooling lets the
  larger class set the fit.
- **The hold-out designs differ.** The PROTAC surrogate is held out by warhead
  scaffold. A glue has no warhead to hold out by.

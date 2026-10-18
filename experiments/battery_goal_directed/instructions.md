# battery_goal_directed

*Last updated: October 18th, 2026.*

## What this demonstrates

The published non-fluorinated, low-permittivity campaign, reproduced in GraphINVENT2 and extended with a virtual
screening baseline and the full metric suite. Reproducing a published campaign in the rewritten codebase is
itself the result.

## Blockers, in the order they bite

1. **The oracle does not exist.** The six S7 terms, the structural filter function, the permittivity chain with
   its conformer step, and the CPI implementation are all unwritten (Prompt E 5.3). This is the largest single
   piece of unbuilt work in the program.
2. **PiNet2 is not installed and its aarch64 viability is unknown.** If it will not install, the fallback is to
   retrain only the five electronic properties as XGBoost on Batt-P30K with 2048-bit Morgan2 fingerprints, and to
   report per-property MAE beside the published PiNet2 MAEs of 0.04, 0.02, 0.04, 0.06 eV and 0.59 D (unverified).
   A fingerprint model cannot distinguish conformers, which matters most for the dipole, and the text must say so.
3. **L2 must be closed.** S7 is a six-component geometric mean, and the budget curve currently integrates
   whichever component was called most. Until that is fixed this experiment is not measurable at all.
4. **The oracle has not been timed.** The conformer step is inside it. Time it on 100 molecules before anything
   else in this program; if it dominates by an order of magnitude the design changes.

## KILL CRITERION: validate the chain before using it

Run the chain over KBS-409 and show the known weakly solvating electrolytes come out with permittivity below 10
and CPI below 0.5 while the conventional dilute electrolytes do not. The labelled external set: MEK, 2Me-DOL,
DIGDBE, HME, MTBE and OTA as weakly solvating; DMA, DMF, DEF, MPC, EPC, EB, GBL, MF and EF as conventional. **If
the chain cannot reproduce that separation, nothing in the battery program runs.** KBS-409 is not in the source
repository either.

## The objective

    S7(G) = [ S_f3 * S_s1 * S_v * S_mp * S_dc * S_r ] ^ (1/6)

`S_f3` is read as "passes the base structural filters AND contains no F and no P". Table 1 of the source work says
"the opposite of S_f2", which read literally scores a molecule that fails the filters; that cannot be the intent,
since S_f3 is the only structural term in S7. The 0.20 minimum Tanimoto to the F/P-holdout is a dataset
construction criterion and is **not** one of the structural filters; the two are implemented separately.

## The funnel is not part of the reward

The reward is S7. The posterior verification funnel runs afterwards as a separate script, stage by stage, with a
count at each stage so our table lines up with the published one. The published counts (unverified): 180,000
generated; 7,832 with permittivity below 10 but failing another constraint; 4,647 meeting all requirements; 4,471
with SA score at most 4.0; minus 75 for maximum Tanimoto above 0.7 to the non-fluorinated KBS-409 solvents; minus
548 for CPI above 0.5; then PubChem CID lookup, a literature check, and a ring strain filter at predicted RSE
below 23 kcal/mol; 30 final candidates.

The source describes 4,647 as already including CPI below 0.5 and then removes 548 for CPI above 0.5 two steps
later. The consistent reading, which we implement, is that CPI was applied **after** the SA and Tanimoto stages.
Each filter is applied once, in a stated order, with a count at every stage.

## Submit

    python submit.py experiments/battery_goal_directed/jobs/rl_nonf_lowdc --slurm

## What to check

`results/metrics.json`: `quality_filtered_top10_mean`, `top100_scaffold_count`, and
`top100_outside_domain_rate` pointed at DC-200 and DN-218. The permittivity model is fitted on 200 molecules and
the donor number model on 218, so generated molecules will routinely fall outside both. **If most of the top 100
sit outside the domain of the model that selected them, that is the finding and it goes in the paper.**

Expect uniqueness and novelty to collapse over the run: published values fall from about 0.9 to 0.2 and from
about 0.57 to 0.10 over 100 steps (unverified), with hit rate rising from about 0.015 to about 0.057. Reproducing
that collapse demonstrates the metric suite rather than failing.

## Core metrics come from

`generate_best_agent`; budget curves from the `rl_nonf_lowdc` trajectory; the funnel from `funnel`.

## Failure mode

S7 identically zero for every molecule means a binary term is always failing: check `S_f3` first, since a single
failed filter zeroes the geometric mean. A rising score with a collapsing `top100_scaffold_count` is reward
hacking.

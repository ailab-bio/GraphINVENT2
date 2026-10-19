# Generating and Evaluating

*Last updated: October 18th, 2026.*

A `generate` job samples from a fixed checkpoint. It has two modes, and they answer different questions.
`sample_mode: generate` writes molecules and computes the metric table. `sample_mode: evaluate` writes no
molecules and computes the model's negative log-likelihood on each data split plus the
uniformity-completeness JSD. Both are in this tutorial because both are the same job type.

## Run them

```bash
python submit.py experiments/tutorial_generate/jobs/debug      # 7 s, sample_mode: generate
python submit.py experiments/tutorial_generate/jobs/evaluate    # 9 s, sample_mode: evaluate
```

Both need a checkpoint, so run [Unconditional training](./unconditional.md) first.

```yaml
job_type: generate
sample_mode: generate
pretrained_model_path: experiments/tutorial_unconditional/jobs/debug/checkpoints/model_restart_4.pth
n_samples: 100
batch_size: 50
device: cpu
```

Here the key is `pretrained_model_path` rather than `resume_from`, because this job samples from a fixed
checkpoint instead of continuing to train it. The experiment points at `dataset_dir: zinc1k`, the same dataset the
prior was trained on, because the feature vocabulary baked into the checkpoint has to match the dataset's.
Pointing it elsewhere makes the checkpoint fail to load.

## The molecules

```
results/molecules.smi  molecules.likelihood  molecules.valid      12 lines each
```

The three files line up by line number, with no header, so line *n* is the same molecule in all three. **Only
valid molecules are written.** An invalid graph contributes no line and its index is recorded instead. Earlier
versions wrote `[Xe]` as a placeholder, which RDKit parses as valid xenon, so anything re-reading the file
counted every line as valid and reported inflated validity. Nothing writes a placeholder now, and a run of
`grep -c Xe results/molecules.smi` returning zero is the check.

Four epochs on a thousand molecules produces output like this:

```
[N]
C1CN2CCC1CC2
[N+][NH2+]
```

One real bicyclic amine among fragments that are technically parsable and chemically nothing. That is what a
12% validity model looks like, and reading the file is a faster way to understand a number than reading the
number.

## The metric table

`results/metrics.json`, from the run above:

```
validity                   0.12        n_sampled              100
uniqueness                 1.0         n_valid                 12
novelty                    1.0         n_unique_valid          12
vun                        0.12        n_unique_valid_novel    12
near_duplicate_rate        0.0         validity_n_generated_raw 100
internal_diversity         0.917       n_scaffolds             12
scaffold_diversity         1.0
sa_score_mean              3.471       filter_pass_rate        0.833
sa_score_median            3.453       Ro5_pass_rate           1.0
property_window_rate       1.0
```

Validity, uniqueness and novelty compose as a funnel, each computed on what survived the previous stage:
`validity` is valid over N, `uniqueness` is unique canonical SMILES over valid, `novelty` is unique-valid
molecules absent from the **training** split over unique valid. `vun` is the composite, unique-valid-novel over
N, and it equals the product of the three. The four counts are reported so every denominator can be
reconstructed rather than inferred.

`validity_n_generated_raw` is how many graphs the generator produced before the list was truncated to
`n_samples`. A batch can come back larger than `batch_size`, so without that truncation the funnel would divide
by `n_samples` while counting more molecules than that, and validity could exceed 1.0. It is latent on a weak
model and fires on a good one.

`uniqueness` and `novelty` at exactly 1.0 are not a result here. With 12 unique valid molecules, none of which
resembles the training set, the funnel has nothing to bite on. `near_duplicate_rate` is the metric that
distinguishes a genuinely novel model from one that memorized and added a methyl: it is the fraction of
unique-valid molecules whose maximum ECFP4 Tanimoto to any training molecule exceeds 0.9. High novelty beside a
high near-duplicate rate means memorization.

### Why fcd is null here, and why that is correct

```
fcd                 null
fcd_n_generated     29
fcd_n_reference     99
unavailable.fcd     "insufficient_sample: the Frechet ChemNet Distance needs at least
                     1000 unique valid molecules on each side …"
```

ChemNet activations are 512-dimensional, so a sample covariance estimated from fewer than 512 molecules is
singular and is badly conditioned well above that. An FCD from a few hundred molecules is not a noisy version of
the real number, it is a different and much larger quantity. Published values use 30,000 molecules (MOSES) or
10,000 (GuacaMol), so there is no sample size at which a small-sample FCD becomes comparable to one of those.

The floor is therefore 1,000 unique valid molecules on each side, and below it the metric is not computed. On
this development set the reference split is about a hundred molecules, so fcd is unavailable **by design**, and
no tutorial here presents a development-set fcd as a result. Between 1,000 and 10,000 it is computed with a
`fcd_comparability` note saying it is comparable between our own runs and not to a published number.

Both sides go through the same canonicalization with the same chirality setting. They used not to: the generated
side was flattened to match `use_chirality` while the reference split went in raw, and since ChemNet reads SMILES
characters, one side carried stereo tokens and the other did not.

### The nulls that are not failures

`bRo5_pass_rate` is null because the metric has no verified definition. The literature describes "beyond Rule
of 5" as the region compounds fall into by exceeding Ro5, not a filter they pass, so the sense of a pass rate is
unsettled. Reporting a number there would be inventing one.

Every null in `metrics.json` carries a reason and a category: `optional_tool_missing`,
`insufficient_sample`, or `not_applicable_to_job_type`. A null with a reason is more useful than an absent key,
which is why the file is shaped as three blocks rather than one flat dictionary.

## Evaluate mode

```yaml
sample_mode: evaluate
```

Same checkpoint, different question:

```
nll_train         31.77
nll_validation    94.32
nll_test         287.46
uc_jsd             0.684
```

These are per-molecule negative log-likelihoods, summed over construction actions, so they are comparable across
splits of one dataset and **not** across datasets with different molecule sizes. Train far below validation far
below test is the signature of a model that has fit 689 training molecules and generalizes poorly, which is what
four epochs on a thousand molecules should produce.

This mode reads the test split, which is the one job that is allowed to. The split stays closed during
development: not for early stopping, not for a quick sanity check, not for choosing a hyperparameter. Running
evaluate mode is a final-evaluation action, and running it repeatedly while tuning is how a held-out set stops
being held out.

Evaluate mode writes no `molecules.*`, because it scores the model rather than sampling from it. It leaves the
per-batch files it used under `results/generation/` and a `logs/validation.log`.

## The report

Both modes write `results/report.html`, a single self-contained page that opens by double-clicking, with no
server and no network. It carries the metric table with the `unavailable` reasons shown as such, the run block,
and for a `generate` job a grid of structures chosen with a MaxMin diversity picker at a fixed seed so the same
run always shows the same molecules.

It also embeds the figures the job wrote. `results/generation.png` has up to three panels: the funnel with its
counts annotated, the distribution of each molecule's maximum Tanimoto to the training split, and the generated
against test distributions of MW, logP and SA as z-scores against the test split. The second is where
memorization shows — a mass piled against 1.0 is a model reproducing its training set while scoring well on
binary novelty. A conditional job writes `results/conditioning.png` beside it. Every figure is saved as `.png`
at 300 dpi and as `.pdf`; the PDF is the one that goes into the paper. A panel whose input is missing is not
drawn and the job prints a line saying what was missing, so a two-panel figure is a statement rather than a
truncation.

## When it goes wrong

A checkpoint that will not load is almost always a vocabulary mismatch between the dataset the job points at and
the dataset the checkpoint was trained on.

Validity above 1.0 means the generated list was not truncated to `n_samples`; `compute_core_metrics` raises
rather than reporting a rate above one, so you will see an error naming the cause rather than a wrong number.

A `metrics.json` with more nulls than you expect is worth reading rather than rerunning: the `unavailable` block
names the input or the tool that was missing.

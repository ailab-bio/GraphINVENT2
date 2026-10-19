# Unconditional Training

*Last updated: October 18th, 2026.*

An `unconditional` job fits the generative model to a preprocessed dataset. It is also how transfer learning
works: set `resume_from` and the same job type specializes an existing checkpoint onto new data. There is no
separate transfer job type, and adding one would mean two code paths that have to stay in agreement.

## Run it

```bash
python submit.py experiments/tutorial_unconditional/jobs/preprocess   # 18 s
python submit.py experiments/tutorial_unconditional/jobs/debug        # 132 s
```

The second needs the first. Both numbers are from this laptop on CPU, with other work running, so treat them as
an order of magnitude rather than a benchmark.

The training configuration:

```yaml
job_type: unconditional
epochs: 4
batch_size: 50
accumulation_steps: 1
sample_every: 2
n_samples: 100
device: cpu
```

`batch_size` counts subgraphs, not molecules, because the HDF5 holds one row per construction action.
`accumulation_steps` is the number of batches between optimizer updates, so the effective batch is
`batch_size × accumulation_steps`: raise it when the batch you want will not fit in memory. `sample_every` is the
evaluation cadence, in epochs.

The dataset is a fixed 1,000-molecule random subset of ZINC250k, committed at `data/datasets/zinc1k/zinc1k.smi`
so the tutorial reproduces without a download.

## Watching it train

`logs/convergence.log` is the curve:

```
epoch, lr, avg_train_loss, avg_valid_loss, model_score
Epoch 1, 0.00094103, 4.75634909, 3.32174039, NA
Epoch 2, 0.00081056, 2.59614658, 2.25029206, 0.435316
Epoch 3, 0.00028170, 1.65543735, 1.89079511, NA
Epoch 4, 0.00000001, 1.05871475, 1.82128751, 0.725597
```

Validation loss falling from 3.32 to 1.82 over four epochs is the thing to check first. `model_score` appears
only on evaluation epochs, which is why two rows carry `NA`. The learning rate follows a one-cycle schedule, so
the last epoch runs at almost zero by design.

`logs/validation.log` carries the per-molecule likelihoods and the uniformity-completeness JSD at each evaluation
epoch. Neither file is a results file: four epochs on a thousand molecules produces a model that cannot generate
much, which is the point of a tutorial rather than a problem with it.

## What it writes

```
logs/        convergence.log  validation.log
results/     metrics.json  report.html  config.resolved.yaml  generation/
checkpoints/ model_restart_2.pth  model_restart_4.pth
```

Checkpoints are saved on the `sample_every` cadence, which is why there are two rather than four.

A training job's `metrics.json` holds eight keys and no molecule-level table:

```json
"metrics": {
    "n_parameters": 7684609,
    "train_wall_clock_s": 126.98751497268677,
    "peak_memory_mb": 1634.90625,
    "inference_s_per_10k": null,
    "nll_train": null, "nll_validation": null, "nll_test": null, "uc_jsd": null
}
```

That is deliberate and it is the one place a reader is likely to be surprised. A training job has no sample of
its own to measure, and generating one to produce a core metric table would double the cost of training. The
funnel, the diversity metrics and the distribution match come from the `generate` job that follows. Every null
carries its reason in the `unavailable` block, so the file says why rather than leaving you to guess:

```
"nll_train": "per-split NLL is computed by a generate job with sample_mode: evaluate, not during training"
```

The four efficiency keys are reported for every trained model, which is what makes two training runs comparable
in cost as well as in quality.

### The per-batch files

`results/generation/` holds the molecules sampled at each evaluation epoch, as a trio per batch:

```
epoch_2_batch_0.smi  epoch_2_batch_0.likelihood  epoch_2_batch_0.valid    19 lines each
epoch_2_batch_1.smi  epoch_2_batch_1.likelihood  epoch_2_batch_1.valid    10 lines each
```

The three files line up by line number, with no per-batch header, so line *n* of the `.smi` is the molecule whose
likelihood is line *n* of the `.likelihood`. Only valid molecules are written: an invalid graph contributes no
line, and its index is recorded in the `.valid` file rather than being written as a placeholder molecule. A
`generate` job concatenates these into one trio named `molecules.*`; a training job leaves them per batch.

## Transfer learning

Transfer learning is this job type with one extra key:

```yaml
job_type: unconditional
resume_from: experiments/<other>/jobs/<prior>/checkpoints/model_restart_20.pth
```

`resume_from` loads the weights and continues training on whatever dataset the job points at. The key to use is
`resume_from` and not `pretrained_model_path`: the latter is for jobs that sample from a fixed checkpoint, and
setting it on a training job is rejected with an error naming the right key. It used to be accepted and ignored,
which trained from scratch while looking like transfer learning, and produced identical losses either way.

Architecture is read from the checkpoint and cannot be overridden. The GGNN sizes, `condition_dim`,
`condition_embedding_dim`, `condition_type` and `conditioning` all come from the file, and a config that sets one
of them to a conflicting value is an error rather than a silent override. Loading a model with the wrong
architecture either crashes confusingly or produces a silently wrong model, so it is checked before anything
loads.

### The control that makes it a result

A transfer run on its own says nothing. Pair it with a from-scratch run on the same data, with every other key
identical, and compare three things: the epoch-1 validation loss, the final validation loss, and the number of
epochs the transfer model needs to reach the from-scratch model's final loss. If the epoch-1 losses match, the
checkpoint is not being loaded, and no test currently catches that for you. Check the log line reading
`Loading model from:` and check the two losses differ.

The feature vocabularies also have to be compatible. A checkpoint cannot encode an atom type its training set did
not contain, and nothing yet validates that after the fact; `extra_dataset` lets you build a shared vocabulary up
front.

## Restarting

`restart: true` resumes a run that stopped. The epoch it resumes from is the highest `model_restart_<N>.pth` in
the checkpoints directory, so the file it is about to load is guaranteed to exist.

## When it goes wrong

Validation loss flat or rising from epoch 1 usually means the learning rate is wrong for the batch size. Loss
falling while the generated molecules stay invalid is normal early: the model learns the action distribution
before it learns to close a valence.

A `generate` job that fails to load this checkpoint almost always means the dataset vocabularies differ. The
tutorial experiments all point at `zinc1k` for that reason.

# Tutorial 3: Transfer learning

Transfer learning here means continuing supervised training from a checkpoint on a second,
usually smaller and narrower, dataset. It is not a separate job type: `unconditional` with
`resume_from` set to a checkpoint path does exactly this, and `unconditional` with
`resume_from: null` is pretraining. The training loop, loss, and schedule are identical; only
the initialisation differs.

The reason this works is that most of what the model has to learn is dataset-independent —
which atoms can bond to which, how rings close, when a fragment is finished. A focused dataset
of a few thousand molecules is far too small to teach that from scratch, but large enough to
shift an already-competent model's distribution toward its region of chemical space.

The corresponding risk is that the shift goes too far. Fine-tuning on a narrow set with a
learning rate suited to pretraining will overwrite the general structure the model relies on,
and validity collapses before the distribution has moved anywhere useful.

---

## Prerequisites

1. A checkpoint from [Tutorial 2: Pretraining](./02_pretraining.md), i.e. a
   `model_restart_<N>.pth` file with a `params_all.json` beside it in the same directory.
2. The fine-tuning dataset preprocessed with the **same feature vocabulary** as the
   pretraining dataset. See [Tutorial 1](./01_preprocessing.md).

---

## Vocabulary compatibility

The node and edge feature dimensions are determined by the preprocessing vocabulary, and
`load_state_dict` requires exact shape agreement, so the two datasets must share:

| Must match | Because it sets |
|-----------|-----------------|
| `atom_types`, `formal_charge`, `imp_H`, `chirality` | Node feature width |
| `use_chirality`, `use_explicit_H`, `ignore_H` | Node feature width |
| `use_aromatic_bonds` | Edge feature width |
| `max_n_nodes` | Every graph tensor shape and the readout width |
| `decoding_route` | Which subgraph sequence the targets describe |

The reliable way to guarantee this is to preprocess both datasets in a single multi-dataset
run, which computes one union vocabulary and encodes both against it:

```json
"dataset":     ["pretrain-set", "finetune-set"],
"smiles_file": [null,           "./data/raw/finetune.smi"]
```

If the pretraining dataset is already preprocessed, the alternative is to set
`auto_detect_features: false` on the fine-tuning run and copy the vocabulary out of the
pretraining dataset's `preprocessing_params.json`. Either way, note that `max_n_nodes` must be
at least as large as the largest molecule in *both* sets, and that enlarging it changes the
readout width, which means an existing checkpoint can no longer be loaded.

---

## Architecture inheritance, and its one trap

When `resume_from` (or `pretrained_model_path`) is set, `src/graphinvent/parameters/config.py`
reads the `params_all.json` sitting next to the checkpoint and fills in the GGNN architecture
parameters from it. That inheritance applies **only to keys absent from your job config**: any
architecture key you write in the `job` block wins over the checkpoint's value.

This matters because `jobs/unconditional/params.json` lists the full architecture block. If you
set `resume_from` in a copy of that template without also deleting the architecture keys, the
model is built at the template's sizes rather than the checkpoint's, and the run fails with a
shape mismatch from `load_state_dict` — or, if the sizes happen to coincide, succeeds while
silently ignoring what the checkpoint recorded.

Delete the architecture keys from the job block when you set `resume_from`. Keep them only if
you deliberately want to override an inherited value.

---

## What changes relative to pretraining

Only two things: where the weights come from, and how aggressively they are allowed to move.

| Parameter | Pretraining | Fine-tuning | Reason |
|-----------|-------------|-------------|--------|
| `resume_from` | `null` | path to `.pth` | Loads the checkpoint instead of initialising randomly |
| `init_lr` | `1e-4` | `1e-5` | Smaller steps preserve the pretrained representation |
| `max_rel_lr` | `10` | `5` | A lower peak in the one-cycle schedule |
| `epochs` | `100` | `20`–`50` | A small dataset reaches its useful minimum quickly, and further epochs mostly memorise |

These are starting points rather than tuned values. The right learning rate depends on how far
the fine-tuning distribution is from the pretraining one, and the useful diagnostic is watching
`fraction_valid_pt` in `generation.log` over the first few evaluation epochs: a sharp drop means
the updates are too large.

The epoch counter restarts at 1, so after 50 epochs of fine-tuning the final checkpoint is
`model_restart_50.pth` regardless of how long the pretraining run was.

---

## Directory layout assumed below

```
data/datasets/
  debug/                        pretraining dataset, already preprocessed
    train.h5  valid.h5  test.h5
  new-dataset/                  fine-tuning target
    train.smi valid.smi test.smi
    train.h5  valid.h5  test.h5

output/debug/unconditional/run/
  model_restart_100.pth         the checkpoint to fine-tune from
  params_all.json               architecture read from here
```

---

## Configuration file

```bash
cp jobs/unconditional/params.json jobs/unconditional/my_transfer.json
```

Edit the copy, deleting the architecture keys:

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "new-dataset",
    "job_name": "run",
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-06:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "unconditional",
    "resume_from": "./output/debug/unconditional/run/model_restart_100.pth",

    "device": "cuda",
    "restart": false,
    "use_tensorboard": true,
    "decoding_route": "bfs",
    "use_aromatic_bonds": true,

    "epochs": 50,
    "batch_size": 1000,
    "block_size": 100000,
    "accumulation_steps": 10,
    "init_lr": 1e-5,
    "max_rel_lr": 5,
    "min_rel_lr": 0.0001,
    "sample_every": 10,
    "n_samples": 2000,
    "n_workers": 0
  }
}
```

`submit.py` checks that the `resume_from` path exists before creating any directories, so a
typo fails immediately rather than after the data loader has been built.

---

## Running the job

Preprocess the fine-tuning dataset first if you have not already:

```bash
python submit.py --config jobs/preprocess/my_finetune_dataset.json
python submit.py --config jobs/unconditional/my_transfer.json
```

---

## Output

Written to `output/<dataset>/unconditional/<job_name>/`, with the same files as a pretraining
run: `params_all.json`, `convergence.log`, `generation.log`, `validation.log`,
`model_restart_<N>.pth` at each evaluation epoch, a `generation/` directory of sampled SMILES,
and `progress.png`. See [Tutorial 2](./02_pretraining.md#output) for what each contains.

---

## Judging whether it worked

The loss curve alone will not tell you. Validation NLL nearly always falls after a warm start,
because the model was already good and the new data is narrow; that is consistent both with
useful adaptation and with memorising a few hundred molecules.

Three things are more informative:

- `fraction_valid_pt` should stay near its pretraining level. A sharp fall in the first few
  evaluation epochs means `init_lr` or `max_rel_lr` is too high.
- `novelty` and the test-set similarity columns in `generation.log` distinguish adaptation from
  memorisation. A model that has memorised a small fine-tuning set produces low novelty and
  very high nearest-neighbour similarity.
- The property histograms should move toward the fine-tuning set's, not merely away from the
  pretraining set's. Both a well-adapted model and a broken one differ from the prior.

If the distribution has not moved at all after the full run, the learning rate is more likely
to be the cause than the epoch count, since the one-cycle schedule spends most of the run near
its floor.

---

## Next steps

- Generate from the fine-tuned model: [Tutorial 5: Sampling](./05_sampling.md), with
  `pretrained_model_path` set to the checkpoint you want, for example
  `"./output/new-dataset/unconditional/run/model_restart_50.pth"`.
- Optimise further toward a scoring function: [Tutorial 4: Reinforcement learning](./04_reinforcement_learning.md).

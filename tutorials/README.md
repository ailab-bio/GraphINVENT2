# GraphINVENT2 tutorials

Six documents covering the job types and the metrics. They are ordered by dependency rather
than by difficulty: preprocessing produces the HDF5 files everything else reads, and training
produces the checkpoints that generation, transfer learning, and RL all start from.

| # | Document | What it covers |
|---|----------|----------------|
| 1 | [Preprocessing](./01_preprocessing.md) | SMILES to HDF5, feature vocabularies, dataset splits |
| 2 | [Pretraining](./02_pretraining.md) | Supervised training from random initialisation (`unconditional`) |
| 3 | [Transfer learning](./03_transfer_learning.md) | Continuing from a checkpoint on a new dataset (`unconditional` + `resume_from`) |
| 4 | [Reinforcement learning](./04_reinforcement_learning.md) | Goal-directed optimisation (`goal_directed`) |
| 5 | [Sampling](./05_sampling.md) | Generating molecules from a checkpoint (`generate`) |
| 5 | [Conditional generation](./05_conditional_generation.md) | Property-conditioned training and sampling (`conditional`) |
| — | [Evaluation](./evaluation.md) | Every metric, what it means, and which job types actually compute it |

---

## Sequences

Train from scratch, then sample:

```
01_preprocessing → 02_pretraining → 05_sampling
```

Adapt an existing model to a narrower dataset:

```
01_preprocessing (new dataset) → 03_transfer_learning → 05_sampling
```

Optimise toward a scoring function:

```
02_pretraining → 04_reinforcement_learning → 05_sampling
```

Condition on measured properties:

```
01_preprocessing (TSV input) → 05_conditional_generation → 05_sampling
```

---

## How jobs are launched

Every job goes through `submit.py` with a JSON config:

```bash
python submit.py --config jobs/<job_type>/params.json
```

The config has two blocks. `"submission"` says how and where to run — interpreter, source
directory, dataset location, SLURM settings — and `"job"` says what to run, meaning `job_type`
plus every model and training hyperparameter.

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "debug",
    "job_name": "run",
    "use_slurm": false
  },
  "job": {
    "job_type": "unconditional"
  }
}
```

Anything omitted from the `job` block falls back to `src/graphinvent/parameters/defaults.py`,
which is the authoritative list of parameters and their defaults. `submit.py` validates the
config before creating any directories, and reports every problem it finds at once rather than
failing on the first.

Valid `job_type` values are `preprocess`, `unconditional`, `conditional`, `goal_directed`, and
`generate`. The older names `pretrain`, `transfer`, `rl`, `constrained_rl`, `sample`, and
`test` are rejected by the validator; `main.py` still remaps them if you invoke it directly,
but that path bypasses validation and is not the supported route.

Output goes to `output/<dataset>/<job_type>/<job_name>/`, where `job_name` comes from the
`submission` block. Each run is self-contained, including its TensorBoard logs; re-running a
job into a directory that already holds results moves the old ones into a timestamped
`_previous_run_<stamp>/` rather than overwriting them.

The files in `jobs/` are templates. Copy one before editing so that the original stays intact
and each experiment keeps its own config.

---

## Author

Rocío Mercado

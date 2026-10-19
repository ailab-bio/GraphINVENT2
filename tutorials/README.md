# GraphINVENT2 Tutorials

*Last updated: October 18th, 2026.*

Five walkthroughs, one per job type. Each points at a real experiment under `experiments/tutorial_<job_type>/`
that runs end to end on a laptop CPU in a few minutes, on a dataset committed to the repository. No cluster, no
GPU, no downloads.

| Job type | Tutorial | Experiment | Runs in |
|---|---|---|---|
| `preprocess` | [Preprocessing](./preprocess.md) | `experiments/tutorial_preprocess` | ~5 s |
| `unconditional` | [Unconditional training](./unconditional.md) | `experiments/tutorial_unconditional` | ~2 min |
| `generate` | [Generating and evaluating](./generate.md) | `experiments/tutorial_generate` | ~20 s |
| `conditional` | [Conditional generation](./conditional.md) | `experiments/tutorial_conditional` | ~3 min |
| `goal_directed` | [Goal-directed optimization](./goal_directed.md) | `experiments/tutorial_goal_directed` | ~1 min |

Transfer learning does not have its own tutorial because it is not its own job type: it is an `unconditional` job
with `resume_from` set, and [Unconditional training](./unconditional.md) covers it.

## How a job is launched

One positional argument, one optional flag:

```bash
python submit.py experiments/<experiment>/jobs/<job>            # here, now
python submit.py experiments/<experiment>/jobs/<job> --slurm    # submit to SLURM
```

Run it from the repository root. The editable install writes a `.pth` file that does not survive a repository path
containing spaces, so console entry points and bare `from metrics import ...` fail; `submit.py` puts the source
directories on the path itself.

The runner is a command-line choice and never a config key, which is what lets the same job directory run on a
laptop and on a cluster without editing anything.

## How a job is configured

Three layers of YAML, merged key by key, later wins:

1. `config/defaults.yaml`, every parameter with its default and a one-line comment
2. `experiments/<experiment>/experiment.yaml`, settings shared by the experiment's jobs
3. `experiments/<experiment>/jobs/<job>/config.yaml`, this job's settings

Omitting a key *is* inheriting it; there is no flag to turn inheritance on or off. Nested blocks merge key by key
as well, so setting one key under `slurm:` does not discard the rest. The merged result is written to the job's
`results/config.resolved.yaml`, so every run records exactly what it ran with.

## What a job leaves behind

```
experiments/<experiment>/jobs/<job>/
    results/        metrics.json, report.html, generated molecules, config.resolved.yaml
    checkpoints/    model state
    logs/           convergence.log, score.log for RL, tensorboard/
```

`results/metrics.json` is the record. It is written by every job type that produces metrics, which is `generate`
in both of its modes, `goal_directed`, `unconditional` and `conditional`. A `preprocess` job writes its outputs
into the dataset directory instead, because a dataset is shared by every job that uses it.

Re-running a job moves the previous run to `_previous_run_<timestamp>/` inside the job directory rather than
overwriting it. Those add up: `python cleanup.py --backups --dry-run` lists them and `--backups` removes them.

## Reading the numbers

`metrics.json` has three blocks. `metrics` holds the values, flat, keyed exactly as CLAUDE.md names them. `run`
records the job type, dataset, seed, device and timestamp. `unavailable` explains every null, with a category:
`optional_tool_missing`, `insufficient_sample`, or `not_applicable_to_job_type`. A null that carries its reason is
worth more than a missing key.

Metric definitions live in one place, CLAUDE.md's "Evaluation metrics" section, and are implemented once, in
`src/metrics/`. If a tutorial and CLAUDE.md disagree about what a metric means, CLAUDE.md is right and the
tutorial is a bug.

# GraphINVENT2

A graph-based molecular generative model for 2D molecular generation. It extends the original
GraphINVENT with conditional generation, and keeps transfer learning and RL-guided optimization while
making both easier to use and more efficient.

This is a research codebase with one maintainer. It is not a library, a product or a framework.

Four documents, four jobs. Read the one that matches what you want:

| | |
|---|---|
| **[CLAUDE.md](CLAUDE.md)** | the specification. What the system is, what it guarantees, what every job type and every metric means |
| **[RUN.md](RUN.md)** | the operations. How to set up, submit, check and recover, step by step, including on the cluster |
| **[manuscript/experiment-plan.md](manuscript/experiment-plan.md)** | the science. What we are running and why |
| **[TODO.md](TODO.md)** | the parking lot, currently empty. It points at [CLAUDE.md § Status](CLAUDE.md#status) |

The walkthroughs in **[tutorials/](tutorials/)** are one per job type, each runnable end to end on a
laptop CPU in a few minutes.

## Install

Python 3.13. `pyproject.toml` and `uv.lock` are the single source of truth; nothing else declares a
dependency.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.13 .venv
source .venv/bin/activate
uv sync --frozen --extra metrics --extra surrogates
uv pip install -e . -e ".[dev]"
bash tools/install.sh              # PROTAC-Splitter, RetroTAC, REINVENT4 and the MCF data
```

On a GPU cluster the CUDA wheel goes on top of the locked one, because the locked resolution is the
CPU-only build on Linux aarch64. `RUN.md` Step 3 has the exact line and the check that it worked.

External tools that are not a plain install are inventoried in
**[tools/README.md](tools/README.md)**: what each is, which metrics need it, its pinned version, and
what records null when it is absent.

## Run a job

```bash
python submit.py experiments/<experiment>/jobs/<job>            # local
python submit.py experiments/<experiment>/jobs/<job> --slurm    # submit to SLURM
```

One positional argument, one optional flag. The runner is a command-line choice and never a config
key, so the same job directory runs on a laptop and on a cluster unchanged. Run from the repository
root.

Every run lives in an experiment directory and writes nothing outside one:

```
experiments/<experiment>/
    experiment.yaml      settings shared by all jobs here
    instructions.md      what this experiment claims and in what order its jobs run
    results/             cross-job aggregates: aggregate.json and report.html over seeds
    jobs/<job>/
        config.yaml      this job's settings
        run.sbatch       rendered from the one SLURM template
        results/         metrics.json, molecules.smi, report.html
        checkpoints/     model state
        logs/            stdout, stderr, convergence.log, tensorboard
```

Configuration is three layers merged key by key, later winning: `config/defaults.yaml`, then the
experiment's `experiment.yaml`, then the job's `config.yaml`. Omitting a key *is* inheriting it. The
merged result is written to the job's `results/config.resolved.yaml`, so every run records what it
ran with.

## Five job types

| job_type | What it does |
|---|---|
| `preprocess` | SMILES file to training-ready HDF5, and the three data splits |
| `unconditional` | train a generative model; transfer learning is this with `resume_from` set |
| `conditional` | train a model that conditions on property values or a class label |
| `goal_directed` | optimize a trained model toward an oracle with policy-gradient RL |
| `generate` | sample molecules from a checkpoint and evaluate them |

## Build the data

```bash
python scripts/download_data.py                       # all sources, checksummed, idempotent
python scripts/zinc-create-dataset.py --output data/datasets/zinc250k
python scripts/tpddb-create-dataset.py --modality protac --output data/datasets/tpddb_protac
```

Raw downloads are not committed; their manifests are, so every number is traceable to a dated
snapshot. `scripts/` also holds the activity-label parser, the surrogate trainer, the screening
baseline and the cross-seed aggregator. Each prints what it did and writes its own provenance.

## Check what you changed

```bash
pytest                                      # the test suite
ruff check . && ruff format --check .
python scripts/check-experiments.py         # every job config resolves, every input exists
python scripts/estimate-walltime.py         # the derived walltime and memory per job
python manuscript/check-manuscript.py       # the manuscript's mechanical checks
```

## Results

Metrics go to `results/metrics.json` as `{schema_version, run, metrics, unavailable}`, with the
`metrics` block flat and keyed exactly as `CLAUDE.md` tabulates. A null always carries a reason and a
category, so a missing number can be told from an undefined one. `results/report.html` renders the
same file and opens offline.

Across seeds, `scripts/aggregate-seeds.py` writes `experiments/<experiment>/results/aggregate.json`
and a matching report. Nothing reported should come from a single seed.

## Citing

GraphINVENT2 builds on GraphINVENT: Mercado et al., *Mach. Learn.: Sci. Technol.* 2:025023 (2021),
and the RL formulation of Atance et al., *J. Chem. Inf. Model.* 62:4863 (2022).

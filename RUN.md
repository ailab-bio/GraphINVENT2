# RUN.md

**This is the file to follow. The checklist below is the whole sequence; everything after it is reference you
read only when a step needs explaining or something goes wrong.**

Three other documents exist and none of them is a runbook: `CLAUDE.md` is the specification, which says what a
metric means; `manuscript/experiment-plan.md` is the science, which says why an experiment exists; and each
`experiments/<name>/instructions.md` says what that one experiment claims and how to read its output. You do not
need any of them to run jobs.

---

# THE CHECKLIST

Work down it. Each line is one thing to do, in order; where two lines are marked **(parallel)** they do not
depend on each other. The **detail** column points at the section further down that explains the step.

## A. Get the cluster working (once, about an hour of your attention)

| | Do this | Done when | Detail |
|---|---|---|---|
| A1 | Log in, check quotas, pick a project filesystem | you know where `data/` and `experiments/` will live | [Step 1](#step-1-log-in-and-check-your-quotas) |
| A2 | Clone into a path with **no spaces** | `pwd` has no spaces in it | [Step 2](#step-2-clone-the-repository-somewhere-with-no-spaces-in-the-path) |
| A3 | Build the environment, then add the CUDA wheel **on top of the lock** | `python -c "import torch; print(torch.version.cuda)"` prints a version, not `None` | [Step 3](#step-3-build-the-environment) |
| A4 | `bash tools/install.sh` | PROTAC-Splitter, RetroTAC and REINVENT4 report installed | [tools/README.md](tools/README.md) |
| A5 | Submit the self-test | it reports CUDA true | [Step 4](#step-4-submit-the-self-test) |
| A6 | Set `slurm.account` | submission stops failing with "account unset" | [Step 5](#step-5-set-the-project-account) |
| A7 | Run one tutorial job through SLURM | a `metrics.json` appears | [Step 6](#step-6-run-one-development-job-through-slurm) |
| A8 | **Measure one epoch on the GPU** and update the anchors | `scripts/estimate-walltime.py` numbers come from your hardware | [Step 7](#step-7-measure-one-epoch-on-the-gpu-before-trusting-any-walltime) |

**Do not skip A8.** Every walltime in the repository is a laptop-CPU extrapolation until you do, and three jobs
are currently sized above a typical partition limit because of it.

## B. Get the data (about an hour, mostly waiting)

| | Do this | Done when | Detail |
|---|---|---|---|
| B1 | `python scripts/download_data.py --source tpddb --source zinc --source battslm` | a `MANIFEST.json` in each `data/raw/*` | [Step 2.2](#step-22-the-downloads) |
| B2 | Build the four corpora and the two conditional ones | the row counts in Step 2.3 match | [Step 2.3](#step-23-build-the-degrader-datasets) |
| B3 | Re-render the sbatch scripts with cluster paths | `run.sbatch` files carry your project path | [Step 2.1](#step-21-re-render-every-sbatch-with-cluster-paths) |
| B4 | `python scripts/check-experiments.py` | far fewer "RESOLVE FAILED" lines than before B2 | — |

## C. The kill point (45 minutes, and it can end the program)

| | Do this | Done when | Detail |
|---|---|---|---|
| C1 | `bash experiments/surrogate_protac_optimization/fit.sh` | held-out R² printed for pDC50 and Dmax, three targets | [Step 2.4](#step-24-fit-the-protac-surrogates-which-is-a-kill-point) |
| C2 | `bash experiments/surrogate_protac_evaluation/fit.sh` | same, on all data | same |
| C3 | Read the R² against the mean-prediction baseline | **you have decided whether to continue** | `experiments/surrogate_protac_optimization/instructions.md` |

**If C3 fails, `protac_goal_directed` and `protac_selectivity` do not run.** That is a result, and it gets
reported rather than retried.

## D. Train (days of wall clock, minutes of yours)

Submit in this order. Everything in one row can go at once; a row waits on the row above it.

| | Submit | Jobs | Waits on |
|---|---|---|---|
| D1 | the preprocess jobs for every corpus you built | `*/jobs/preprocess_*` | B2 |
| D2 | the base priors **(parallel)** | `mg_prior/jobs/prior_zinc`; `protac_prior/jobs/prior_enumerated` and `battery_prior/jobs/prior_pubchem120k` are blocked, see below | D1 |
| D3 | the transfer pairs **(parallel)** | each `*_transfer/jobs/prior_transfer` **and** `prior_scratch` | D2 for transfer, D1 for scratch |
| D4 | the conditional models **(parallel)** | the three `*_conditional/jobs/train` | D1 |
| D5 | the optimization campaigns | `protac_goal_directed/jobs/rl_smarca2` and `protac_selectivity/jobs/rl_selectivity`. The other two `rl_*` jobs are blocked, see below | C3 passed, D3 |

`prior_scratch` is the from-scratch control and does **not** wait on a prior. Submitting it in D1 alongside the
preprocess jobs is fine and saves a day.

## E. Sample and report (an hour of yours, hours of wall clock)

| | Do this | Jobs |
|---|---|---|
| E1 | Submit every sampling job: **31 samplers at three seeds each, 93 jobs** | `*/jobs/generate_*` |
| E2 | Submit the screening baselines | `*/jobs/vs_baseline_*` |
| E3 | Aggregate across seeds, per experiment | `python scripts/aggregate-seeds.py experiments/<name>` |
| E4 | Open `experiments/<name>/results/report.html` | — |

Nothing is reported from a training job. Each experiment's `instructions.md` names the sampling job that
produces its numbers.

## What is NOT in this checklist, and why

| | Why it is absent |
|---|---|
| the three `reinvent_*` jobs | they refuse to run. REINVENT4 needs its Zenodo prior specialized onto our corpus and a wrapper exposing our scorer as its `ExternalProcess` component. Submitting one would optimize **our** model and write a `metrics.json` that reads as the baseline |
| `mg_goal_directed` | its reward cannot be fitted: the largest glue target has 30 labelled compounds against a floor of about 100. See `experiments/surrogate_mg_optimization/instructions.md` and decide what it optimizes |
| `protac_prior` | needs `protac_enumerated`, which upstream's generator did not finish building in three hours |
| the battery program beyond `prior_scratch` | PubChem-120K is not distributed, `Batt-P30K.h5` upstream is a 134-byte LFS pointer, and the oracle chain does not exist |

---

# Reference

Everything below explains a line of the checklist. Read it when you need it.

---

# Part 1: from nothing to a working GPU job

Everything here was written on a MacBook with no cluster access and no NVIDIA
GPU. The local commands were run; **nothing involving SLURM, CUDA or aarch64
was**. Steps that could not be tested say so, and every default taken from the
Arrhenius documentation is marked unverified. Treat the expected output as a
prediction to check, not a promise.

The whole of Part 1 is about an hour of attention, with the slow parts
submitted and left.

---

## Step 1: log in and check your quotas

```bash
ssh <username>@arrhenius.nsc.liu.se
quota
storagequota
df -h /nobackup/proj/disk/naiss2024-5-630
```

- **MY TIME:** 5 minutes
- **WALL CLOCK:** immediate
- **DEPENDS ON:** nothing
- **CHECK BEFORE CONTINUING:** `storagequota` reports a project directory under
  `/nobackup` with room for your datasets. Budget from the numbers in
  [Storage](#storage-and-the-data-root): plan on tens of GB for a drug-like
  dataset and over a hundred for anything PROTAC-sized.
- **IF IT FAILED:** `/home` has a 30 GiB quota and is not for project data.
  If `storagequota` shows no project directory, you do not have an allocation
  on the project filesystem yet, and nothing below will work. Sort that first.

*Unverified: the login hostname, and that `quota` and `storagequota` are the
commands. Taken from the Arrhenius quickstart.*

---

## Step 2: clone the repository somewhere with no spaces in the path

```bash
cd /nobackup/proj/disk/naiss2024-5-630
git clone <repo-url> GraphINVENT2
cd GraphINVENT2
```

- **MY TIME:** 2 minutes
- **WALL CLOCK:** under a minute
- **DEPENDS ON:** 1
- **CHECK BEFORE CONTINUING:** `pwd` contains no spaces, and
  `ls config/defaults.yaml submit.py` succeeds.
- **IF IT FAILED:** a path containing spaces breaks the editable install; this
  is why the laptop copy runs with `python submit.py` from the repo root. On
  the cluster just avoid the problem.

---

## Step 3: build the environment

The one trap worth stating before the commands: **on Linux aarch64 the plain
PyPI torch wheel is CPU-only up to and including 2.10**. A build that resolves
to 2.10 imports fine, runs fine, and reports `torch.cuda.is_available() ==
False` with no error anywhere. `pyproject.toml` pins `torch>=2.11` for exactly
this reason. If you need an older torch, install it from the CUDA 13 index
instead.

```bash
module list                       # expect nothing loaded, or only what you chose
module load GPU/buildenv-nvhpc/25.9-cu13.0

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate

uv sync --frozen --extra metrics --extra surrogates   # everything from uv.lock
# Then the CUDA wheel, which is deliberately NOT in the lock: see the comment
# on torch in pyproject.toml. On aarch64 the locked PyPI build is CPU-only.
uv pip install --index-url https://download.pytorch.org/whl/cu130 "torch>=2.11"
uv pip install -e .                # the project itself, editable
uv pip install -e ".[dev]"         # ruff, pytest, pytest-timeout: needed to run the suite
bash tools/install.sh              # PROTAC-Splitter, RetroTAC and the MCF data
```

Without the `metrics` extra, `fcd` records null in every run and the headline
unconditional number does not exist. It is a pure-Python wheel over
torch/numpy/scipy and the ChemNet weights ship inside it, so there is nothing to
download at run time and nothing to point at a cache.

- **MY TIME:** 10 minutes
- **WALL CLOCK:** 10 to 20 minutes, mostly downloading torch
- **DEPENDS ON:** 2
- **CHECK BEFORE CONTINUING:**
  `python -c "import torch, platform; print(platform.machine(), torch.__version__, torch.version.cuda)"`
  prints `aarch64`, a version of 2.11 or higher, and a CUDA version that is
  **not** `None`. A `None` there means a CPU-only wheel and step 5 will fail.
- **IF IT FAILED:**
  - A torch wheel around 100 to 150 MB is the CPU-only aarch64 build. Reinstall
    with `--index-url https://download.pytorch.org/whl/cu130`.
  - `Building wheel for <package>` means pip fell through to a source build;
    stop and check that package against the audit rather than waiting for a
    compile that will probably fail.
  - **Do not load an `hdf5` module.** The h5py wheel bundles its own libhdf5,
    and two copies in one process abort with a version mismatch.
  - `vina` has no aarch64 wheel at any version. `uv sync --extra docking` will
    try to compile it. Leave that extra alone until you need docking, then
    install Vina into a separate conda-forge environment and call it as a
    subprocess.

*Unverified: the module name, that uv's managed CPython 3.13 is available for
aarch64, and every wheel resolution. All documentation-derived.*

---

## Step 4: submit the self-test

```bash
sbatch -A <your-project>-gpu scripts/cluster_selftest.sbatch
squeue -u $USER
```

- **MY TIME:** 2 minutes to submit, 5 to read the output
- **WALL CLOCK:** under 10 minutes once it starts
- **DEPENDS ON:** 3
- **CHECK BEFORE CONTINUING:** `selftest-<jobid>.out` ends with
  `PASS: all 7 checks passed.`
- **IF IT FAILED:** the script stops at the first failure and names it. The
  seven checks are, in order: the interpreter and its architecture; torch and
  CUDA; an RDKit SMILES round trip; an h5py write and read on the data root,
  which also proves the filesystem is writable from a compute node; an xgboost
  fit; a GGNN forward and backward on the GPU with finite gradients; and one
  development job end to end. Each failure message says what to do.
  - Check 2 failing with "built against CUDA" but no visible device usually
    means the driver is older than the toolkit the wheel needs. `nvidia-smi` on
    the compute node tells you the driver version.
  - Check 4 failing is a filesystem problem, not a Python one: the data root is
    somewhere the compute nodes cannot write.

The same checks run without SLURM, which is how the logic was tested:

```bash
python scripts/cluster_selftest.py          # add --require-cuda on a GPU node
```

On the laptop all seven pass in about 10 seconds.

*Unverified: everything about the sbatch route. The plain-Python route was run.*

---

## Step 5: set the project account

`slurm.account` is mandatory and submission fails without it, by design, rather
than letting the scheduler reject the job later.

Put it where it applies to everything:

```yaml
# config/defaults.yaml
slurm:
  account: naiss2024-5-630-gpu
```

or per experiment, which is better when one project pays for some jobs and
another pays for the rest:

```yaml
# experiments/<exp>/experiment.yaml
slurm:
  account: naiss2024-5-630-gpu
```

- **MY TIME:** 2 minutes
- **DEPENDS ON:** 1
- **CHECK BEFORE CONTINUING:**
  `python submit.py experiments/tutorial_unconditional/jobs/debug --slurm`
  no longer raises the "slurm.account is unset" error.
- **IF IT FAILED:** on Arrhenius a GPU job needs the `-gpu` suffix on the
  account name and a CPU job uses the bare name. Getting this wrong is
  rejected at submission, not at run time. *Unverified.*

Nested config blocks merge key by key, so setting `account` alone keeps the
rest of the `slurm:` block.

---

## Step 6: run one development job through SLURM

```bash
python submit.py experiments/tutorial_unconditional/jobs/debug --slurm
```

- **MY TIME:** 2 minutes
- **WALL CLOCK:** a few minutes
- **DEPENDS ON:** 4, 5
- **CHECK BEFORE CONTINUING:** four things, in this order:
  1. `experiments/tutorial_unconditional/jobs/debug/run.sbatch` exists and the
     `#SBATCH` lines say what you expect. It is byte-identical to the script
     that was submitted; there is no second source of truth.
  2. `squeue -u $USER` shows the job, then stops showing it.
  3. `.../logs/slurm-<jobid>.out` ends without a traceback.
  4. `.../checkpoints/` holds a `model_restart_*.pth`, and
     `.../results/config.resolved.yaml` records what actually ran.
- **IF IT FAILED:**
  - Rejected at submission: read `run.sbatch`. The account, the partition and
    the `-t` are the usual culprits.
  - The job starts and dies immediately: the module line or the venv path. The
    script does `cd` to the repository root and runs the same `submit.py` you
    just ran, so anything that works interactively should work here.
  - Nothing in `logs/`: the `#SBATCH -o` path is absolute and is resolved
    before the script runs. If the job directory is not writable from the
    compute node, SLURM has nowhere to put the output and fails silently.

To submit by hand instead, `sbatch experiments/<exp>/jobs/<job>/run.sbatch`
does the same thing.

To watch a run, point TensorBoard at the job's log directory:

```bash
tensorboard --logdir experiments/<exp>/jobs/<job>/logs/tensorboard/
```

This needs `use_tensorboard: true`, and without it an optimization run keeps no
per-step record at all. Any run you intend to report should set it, because the
collapse of diversity over steps is the thing the metric suite exists to make
visible and it is only observable as a trajectory.

*Unverified: all of it. The rendering was tested; the submission was not.*

---

## Step 7: measure one epoch on the GPU, before trusting any walltime

**Every walltime in this repository is a CPU extrapolation.** The anchors in
`scripts/estimate-walltime.py` were measured on a laptop, and three jobs come
out above a typical partition limit because of it. The ratio between CPU and
GPU is not uniform across job types, so there is no correction factor to apply:
training is dense linear algebra and should gain the most, generation is a
sequential loop with a Python-level decode and will gain least. Measure the one
number that decides the schedule.

`mg_prior/jobs/prior_zinc` is the long pole of the whole project, so measure
that one. Build a subset, preprocess it, and time a single epoch:

```bash
# A seeded 3,000-molecule subset of the corpus you already built.
M=$SCRATCH/zinc_epoch
mkdir -p $M/data/zinc_subset $M/jobs/pre $M/jobs/train
python - <<'PY'
import os, pathlib, random
M = pathlib.Path(os.environ["M"])
body = [l for l in pathlib.Path("data/datasets/zinc250k/zinc250k.smi").read_text().splitlines() if l.strip()]
if body[0].split()[0].upper() == "SMILES":
    body = body[1:]
random.Random(1).shuffle(body)
(M / "data/zinc_subset/zinc_subset.smi").write_text("\n".join(body[:3000]) + "\n")
PY

cat > $M/experiment.yaml <<EOF
dataset_dir: $M/data/zinc_subset
max_n_nodes: 54
seed: 1
EOF
cat > $M/jobs/pre/config.yaml <<EOF
job_type: preprocess
smiles_file: $M/data/zinc_subset/zinc_subset.smi
split_type: random
device: cpu
EOF
cat > $M/jobs/train/config.yaml <<EOF
job_type: unconditional
epochs: 1
batch_size: 1000
block_size: 100000
sample_every: 1000
n_samples: 10
device: cuda
EOF

python submit.py $M/jobs/pre            # ~1 minute
python submit.py $M/jobs/train          # the number you want
```

Read the `time elapsed` line from the training job and divide by the training
subgraph count, which the preprocess job prints and `train.h5` carries:

```bash
python -c "import h5py; print(h5py.File('$M/data/zinc_subset/train.h5')['nodes'].shape[0])"
```

**Measured here for comparison**, same subset, same settings, 57,033 training
subgraphs from 2,400 molecules (23.8 per molecule):

| device | one epoch | per subgraph | 40 epochs on the full corpus | request after x1.5 |
|---|---|---|---|---|
| laptop CPU | 91.9 s | 1.612 ms | 85 h | 128 h |
| laptop MPS | 54.9 s | 0.962 ms | 51 h | 76 h |
| GH200 | **measure it** | | | |

The MPS row is this laptop's GPU and is not a prediction of the GH200; it is
there to show that the CPU row is an upper bound rather than the answer. Put
your measured per-subgraph number into `ANCHORS` in
`scripts/estimate-walltime.py` keyed by `max_nodes`, re-run it with `--write`,
and every job in the project is re-sized from a number that came from the
hardware it will run on.

If the GH200 number still puts `prior_zinc` above the partition limit, split it
across submissions. Training resumes from its last checkpoint with the
optimizer and the learning-rate schedule intact, so three 40-hour submissions
and one 120-hour submission reach the same model.

## Step 8: find the usable batch size

See [the batch size tuning procedure](#the-batch-size-tuning-procedure).

- **MY TIME:** 20 minutes
- **WALL CLOCK:** 30 minutes of interactive GPU time
- **DEPENDS ON:** 6

---

## Storage and the data root

Cluster home directories are small and the HDF5 files are not. Measured on a
400-molecule subset of a drug-like set, so you can scale it:

| `max_nodes` | Uncompressed bytes per subgraph | Uncompressed for 400 molecules |
|---|---|---|
| 70 | 68,182 | 679 MB |
| 85 | 87,892 | 876 MB |
| 100 | 109,402 | 1,090 MB |
| 120 | 140,882 | 1,404 MB |

Roughly one subgraph per heavy atom plus two, so a dataset of *M* molecules
averaging *A* heavy atoms with `max_nodes = N` needs about
`M × (A + 2) × bytes_per_subgraph(N)` before compression. At
`max_nodes = 100` that is about 11 MB per molecule uncompressed.

**The HDF5 is written gzip-compressed, which changes these numbers by a lot.**
`action_probs` and `edges` are one-hot and over 99% zeros. Measured on the same
400-molecule subset at `max_nodes = 120`: 1,561 MB of array data became
2.57 MB on disk, a factor of 607, and the arrays read back byte-identical to
an uncompressed run of the same preprocessing. The cost is CPU: preprocessing
went from 20.9 s to 34.3 s and two training epochs from 62.8 s to 66.7 s on
this laptop.

So plan disk from the uncompressed column divided by a few hundred, and plan
**walltime and RAM** from the uncompressed column, because that is what is
decompressed into memory. The compression ratio depends on the vocabulary and
the sparsity, so measure it on your own subset rather than trusting 607.
**Measure on a small subset before committing to a full preprocessing run.**

### Choosing `max_nodes`

`max_nodes` is the 99th-percentile heavy-atom count of the dataset, not the
maximum, because every stored tensor is padded to it and the `edges` dataset
grows with its square. Preprocessing detects the percentile itself; a value
set in the config wins, so a run stays reproducible.

```bash
python scripts/scan_features.py --smi <your>.smi
```

That prints both the percentile and the observed maximum. On the drug-like set
here the percentile is 60 against a maximum of 72, so taking the maximum would
have made every row 44% larger for the sake of the last 1% of molecules.
Molecules above `max_nodes` are dropped before splitting and counted as
`split.n_dropped_oversized` in `preprocessing.yaml`. Check that number: if it
is far above 1% of the set, the percentile is not describing your data.

Keep the repository and the data on the project filesystem:

```
/nobackup/proj/disk/naiss2024-5-630/
    GraphINVENT2/            the clone, including experiments/
    data/datasets/           the datasets
```

`data_root` in `config/defaults.yaml` is the base for any relative
`dataset_dir`, so pointing the whole project at a different filesystem is one
line:

```yaml
data_root: /nobackup/proj/disk/naiss2024-5-630/data/datasets
```

A job directory's outputs are written inside the job directory, so
`experiments/` follows the clone. If you need it elsewhere, either pass the job
path directly, since `submit.py` takes any path, or symlink
`experiments -> /nobackup/.../experiments`. Nothing in the code assumes either
location.

---

## What survives a walltime kill

There is no way to extend a running job, so the question of what a kill costs
you decides how much of a guess the walltime can afford to be. Set
`restart: true` in the job's config and resubmit; the job finds its own state.

| job_type | Killed at the walltime, it... | What it resumes from |
|---|---|---|
| `preprocess` | **resumes** | The HDF5 files already written. It detects which of `valid.h5`, `test.h5`, `train.h5` are complete and continues from the first that is not, so a kill during `train.h5` does not redo the other two |
| `unconditional` | **resumes** | The last `model_restart_<epoch>.pth`, with the optimizer and the learning-rate schedule restored. A kill costs at most one epoch |
| `conditional` | **resumes** | The same path: it is the same training loop |
| `goal_directed` | **resumes** | The last RL checkpoint. A budget-capped run also keeps the oracle call log, so the budget already spent is not spent again |
| `generate` | **starts over** | Nothing. This is deliberate: a generate job is minutes, and `molecules.smi`, `molecules.likelihood` and `molecules.valid` have to line up by line number, so a half-written sample is worse than no sample |

Because `generate` is the one that starts over, it is also the one whose
walltime should be generous rather than tight: there is nothing to lose by
over-requesting and a whole sample to lose by under-requesting.

Previous outputs are not overwritten in place. A rerun moves the old
`results/` aside into a timestamped `_previous_run_*` directory first, so a
kill followed by a restart cannot leave one run's molecules beside another
run's metrics.

## The walltime estimation procedure

`-t` must be set explicitly on every job: the Arrhenius default is 2 hours,
which kills a real run part-way through. There is no way to extend a running
job, so guess high.

1. **Run a short job of the same type.** Set `epochs` to a number that
   finishes in a few minutes, keep every other setting identical to the real
   run, and submit it.
2. **Read the per-unit cost** from `logs/convergence.log`, which has one row
   per epoch, or from the `time elapsed` line at the end of the job. For
   `generate`, use the molecules-per-second figure the job prints.
3. **Extrapolate linearly** in epochs or in molecules. Training cost per epoch
   is flat; generation cost per molecule is flat.
4. **Add the fixed costs**: preprocessing is paid once, not per epoch; each
   evaluation round at `sample_every` costs roughly one generation of
   `n_samples` molecules; the end-of-run evaluation costs one more.
5. **Multiply by 1.5 and round up to the hour.** A job killed at 99% has
   produced nothing but its last checkpoint.

`scripts/estimate-walltime.py` applies all five steps to every job and writes
the number, the memory request and the derivation into each `config.yaml`,
which the SLURM template then renders into `run.sbatch`:

```bash
python scripts/estimate-walltime.py            # print the table
python scripts/estimate-walltime.py --write    # write it into each config
```

**Every number it prints is a CPU extrapolation**, from the measured per-unit
costs below and in `manuscript/experiment-plan.md`. Nothing has been measured
on a GPU. Training will finish well inside the request there; generation will
gain much less, because it is a sequential autoregressive loop with a
Python-level graph-to-molecule decode. That asymmetry is why a CPU
extrapolation is the right thing to request rather than a correction of it.

A job whose dataset is not built yet cannot be sized, and the script says so
per job instead of printing a number derived from a molecule count nobody
measured. Re-run it after the datasets are built, then again on Arrhenius with
measured numbers in place of the anchors.

**CPU numbers measured here, as a starting point.** 400 drug-like molecules,
`max_nodes` 70 (9,413 training subgraphs), batch size 250, on a 10-core Apple
M-series laptop:

| Job type | Measured | Per unit |
|---|---|---|
| `preprocess` | 17.8 s for 400 molecules | ~45 ms per molecule |
| `unconditional` | 50 s for 2 epochs | ~25 s per epoch over 9,413 subgraphs |
| `generate` | 11.3 s / 26.2 s / 47.2 s for 250 / 1,000 / 2,000 molecules | ~21 ms per molecule, plus ~6 s fixed |

These were measured after HDF5 compression was enabled, which is why
preprocessing is slower than a bare write would be; see
[Storage](#storage-and-the-data-root). Generation was measured at four sample
sizes because it has a fixed start-up cost that a single data point hides: fit
your own estimate as `fixed + n x per_molecule` rather than scaling one number.

**These must be re-measured on the GPU before they are trusted.** The ratio
between CPU and GPU will not be uniform across job types: training is dense
linear algebra and should gain the most, generation is a sequential
autoregressive loop with a Python-level graph-to-molecule decode and will gain
much less, and preprocessing is pure RDKit and numpy on the CPU and will not
gain at all. Re-run step 1 of this procedure on the GPU for each job type and
replace the table.

---

## The batch size tuning procedure

**Batch size is a hyperparameter, and changing it needs approval, not just a
job that fits.** It changes the gradient estimate and therefore the training
dynamics, and the values in use come from prior work. What follows finds the
largest value that *fits*, which is the input to that conversation, not a
licence to change it.

Memory grows with batch size and with `max_nodes` together: the model holds a
`batch × max_nodes × max_nodes × n_edge_features` edge tensor and an action
probability vector linear in `max_nodes`. Measured on CPU, same 400 molecules,
one epoch, batch size 250:

| `max_nodes` | peak RSS, preprocess | peak RSS, train |
|---|---|---|
| 70 | 825 MB | 2,154 MB |
| 85 | 1,047 MB | 2,207 MB |
| 100 | 1,119 MB | 2,238 MB |
| 120 | 1,237 MB | 2,447 MB |

Those are CPU resident-set numbers and include the interpreter, the dataset
block in memory and the model. They establish the shape of the curve, not the
GPU figure. With 96 GB of HBM the limit is likely to be far above the current
default, but that extrapolation is unverified.

To find it in one sitting:

1. Grab an interactive GPU node:
   `interactive -A naiss2024-5-630-gpu -p gpu --gpus=1 -c 72 -t 01:00:00`
   *(unverified: that `interactive` is the command on this cluster)*
2. Copy a development job directory and point it at the real dataset.
3. For `batch_size` in 250, 500, 1000, 2000, 4000: set it in the job's
   `config.yaml`, run `python submit.py <job dir>` with `epochs: 1`, and record
   `torch.cuda.max_memory_allocated()` and the wall clock for the epoch.
4. Stop at the first value that raises `CUDA out of memory`, and take the one
   below it.
5. Apply a safety margin: use **half** the largest value that fit. Peak memory
   depends on the largest molecule in the batch, so a value that survives one
   epoch on a shuffled dataset can still fail later on an unlucky batch.
6. Record the number, the `max_nodes` it was measured at, and the dataset.
   It does not transfer between datasets.

Then bring the number to a decision about whether to change the hyperparameter.

---

## What this repository assumes about the node

Recorded here because these are the settings a cluster run depends on and the
laptop never exercises. Everything in this section was reasoned from the code
and the Arrhenius documentation; **none of it has been run on the cluster.**

### Thread counts are set from `slurm.cpus_per_task`

`config/slurm.sh.template` exports `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` from `slurm.cpus_per_task`, which defaults to 72 (one
GH200 superchip). Left unset, the OpenMP runtimes behind torch, numpy and RDKit
size their pools from the node's **256** cores rather than from the cpuset the
job was allocated — oversubscription that is both antisocial on a shared node
and slower than the correct count. PyTorch takes its intra-op thread count from
`OMP_NUM_THREADS`, so no code change is involved.

If you want a different number, change `slurm.cpus_per_task`; do not export the
variables by hand, because then two places disagree. For a job whose bottleneck
is single-threaded graph construction rather than linear algebra, a smaller
number may be faster — measure before changing it.

### A requested accelerator that is absent is now fatal under SLURM

`device: cuda` with no CUDA available falls back to CPU when you run
interactively, so a laptop runs any job directory unchanged. **Inside a SLURM
allocation it raises instead.** This matters because the aarch64 failure mode is
silent: a CPU-only torch wheel reports `torch.cuda.is_available() == False` with
no error of its own, and the old behavior spent the full walltime on CPU and
wrote metrics that looked right. The check keys off `SLURM_JOB_ID`.

To run on CPU inside an allocation deliberately, set `device: cpu` in the job
config. That is the only way, and it is explicit on purpose.

### `n_workers` must be 0

The HDF5 dataset keeps an open file handle. It cannot be pickled for a spawned
worker, and it is not fork-safe for a forked one — where it corrupts the loaded
graphs rather than failing. macOS uses spawn, so the laptop has only ever seen
the loud half of that; Linux uses fork, which is the silent half. Any
`n_workers` above 0 is now refused with a message. Control loading cost with
`block_size`.

### Do not trust `run.sbatch` from a clone

`submit.py --slurm` renders the template with the absolute repo root and
interpreter of the machine that rendered it, then writes it both to
`logs/submit.sh` and to `run.sbatch` for hand submission. A `run.sbatch`
committed from somewhere else therefore points at that machine's paths. It is
gitignored now; if `git status` still shows one, it is tracked from before and
needs `git rm --cached`.

### Two things that are known to be wrong on the cluster and are not fixed

- **`cleanup.py` ignores `data_root`.** It looks in `data/datasets` and
  `experiments` relative to the working directory, so with a `data_root` under
  `/nobackup` it will find nothing, delete nothing, and report success. Clean
  the project filesystem by hand until this is fixed.
- **`src/oracles/_vina.py` sizes its process pool from `os.cpu_count()`**,
  which reports 256 on these nodes rather than the 72 the job was given, and
  forks a parent that may already hold a CUDA context. Do not use the docking
  oracle on the cluster until that is changed.

### There is no lock file

`CLAUDE.md` describes `pyproject.toml` plus `uv.lock` as the single source of
truth, but **no `uv.lock` is committed**, so `uv sync` on the cluster resolves
whatever is current rather than what the laptop ran. Generate one from the
machine that will run the experiments and commit it before any result you intend
to publish:

```bash
uv lock
```

---

# ACTIONS THAT ARE MINE, NOT THE AGENT'S

Two things only I can do, listed here because the agent does not run git and cannot reach some hosts.

**1. Untrack the six tutorial sbatch files committed with laptop-absolute paths.**

```bash
git rm --cached experiments/tutorial_*/jobs/*/run.sbatch
```

They are gitignored now, so this removes them from the index without deleting them from disk. Until it is run, a
fresh clone on the cluster gets six sbatch scripts pointing at a MacBook.

**2. Place the two ACS Nano PDFs.** `manuscript/refs/acsnano-6c06255-main.pdf` and `-si.pdf`. Every battery number
in this runbook and in the experiment directories is marked unverified because no agent has been able to read
them.

---

# Part 2: data, surrogates, and the session plan

Part 1 ends with one tutorial job proven on the cluster. Part 2 onward is the real program: ten experiments, 56
jobs. Everything here was written on a MacBook with no cluster access. **Every walltime and memory figure is a
CPU extrapolation, not a measurement**, derived by `scripts/estimate-walltime.py` from the per-unit costs above;
the derivation is in each job's `config.yaml` and is carried into its `run.sbatch`. Twenty-one jobs could not be
sized at all, because their dataset is not built yet and the molecule count the extrapolation needs is unknown;
re-run the script after the datasets exist.

## Session sizing

**The order is in [THE CHECKLIST](#the-checklist) at the top of this file.** This section used to carry a second
ordering as a table of fifteen sessions; it disagreed with the checklist after the experiments were
restructured, and two orderings that disagree are worse than one. It is gone.

What is worth keeping from it is the shape of the week. About eight hours of your attention, spread across
roughly sixteen sittings of forty-five minutes, each ending with jobs running and the laptop closeable. The wall
clock is dominated by the base priors, which are day-scale: `prior_zinc` is the long pole of the whole project
at 4.75 M training subgraphs.

**Every dependency above is prose, and none of it is expressed as a SLURM dependency.** There is no
`--dependency=afterok` in `config/slurm.sh.template`, in `submit.py` or in any rendered `run.sbatch`,
so nothing stops a job being submitted before the job that writes its input has finished. The
failure is not loud: a `generate` job submitted before its training job completes fails on a missing
checkpoint, which is recoverable, but a `preprocess` job submitted twice concurrently writes the
same HDF5 from two processes, which is not. Until the interface changes, the ordering in this table
is the only thing enforcing it, and the session boundaries exist so that each sitting ends with
everything it submitted able to run. `submit.py` takes one positional argument and one optional
flag by specification, so adding `--after` is an interface change rather than a fix, and it is on
the decision list in `CLAUDE.md` § Status.

## Step 2.1: re-render every sbatch with cluster paths

```bash
python scripts/render-sbatch.py --all
```

- **MY TIME:** 15 minutes, mostly reading two of the scripts
- **WALL CLOCK:** seconds
- **DEPENDS ON:** Part 1 step 5, because the account must be set
- **CHECK BEFORE CONTINUING:** `grep -c /nobackup experiments/protac_goal_directed/jobs/rl_smarca2/run.sbatch`
  returns a non-zero count, and no sbatch file contains `/Users/`. The versions committed to the repository were
  rendered on a MacBook and point at it, which is why this step exists and why it comes before any submission.
- **IF IT FAILED:** `slurm.account is unset` means Part 1 step 5 did not take. Every experiment sets a
  `naiss2024-5-630-gpu` account in its `experiment.yaml`; change it there if your project id differs.

## Step 2.2: the downloads

```bash
python scripts/download_data.py --source tpddb --source zinc
```

- **MY TIME:** 30 minutes
- **WALL CLOCK:** a few minutes
- **DEPENDS ON:** 2.1
- **CHECK BEFORE CONTINUING:** `data/raw/tpddb/MANIFEST.json` and `data/raw/zinc/MANIFEST.json` exist and record a
  sha256 per file. The TPD-DB tables should be `PROTAC_main_table.txt` 6.9 MB and `MG_main_table.txt` 1.7 MB.
- **IF IT FAILED:** the script stops rather than substituting a source. TPD-DB publishes no version in its files,
  so the manifest date is the only provenance you will ever have for these numbers. Keep it.

*Not yet wired into this script, and therefore blockers for the experiments that need them: the PROTAC-Splitter
enumerated library, Batt-SLM.smi, Batt-P30K, PubChem-120K, KBS-409 and the F/P-holdout. `download_data.py` covers
TPD-DB and ZINC250k only.*

*PROTAC-DB is unreachable from the development machine over both https and http: both time out at the connection
level with zero bytes received, so this is not the `http://` error a previous pass was suspected of making. Try it
from the cluster, which sits on a different network.*

## Step 2.3: build the degrader datasets

```bash
# The two task corpora.
python scripts/tpddb-create-dataset.py --modality glue   --output data/datasets/tpddb_glue/
python scripts/tpddb-create-dataset.py --modality protac --output data/datasets/tpddb_protac/

# The two CONDITIONAL corpora, which are different datasets and not a flag on
# the ones above: the class columns change the HDF5, so a conditional job
# cannot read a corpus built without them.
# The flag takes TARGET SYMBOLS, and the column is named is_<symbol> in
# lowercase. `LABEL=A|B` merges two symbols under one label, which is how
# IKZF1 and IKZF3 become one class. The names and their ORDER have to match
# conditioning.properties in the experiment config, because the condition
# vector is built positionally from these columns.
python scripts/tpddb-create-dataset.py --modality protac \
    --conditioning-classes "AR,BTK,EGFR" \
    --output data/datasets/tpddb_protac_cond/
python scripts/tpddb-create-dataset.py --modality glue \
    --conditioning-classes "IKZF=IKZF1|IKZF3,GSPT1,RBM39" \
    --output data/datasets/tpddb_glue_cond/

# The molecular glue base corpus.
python scripts/zinc-create-dataset.py --output data/datasets/zinc250k/
```

- **MY TIME:** 40 minutes
- **WALL CLOCK:** a few minutes each
- **DEPENDS ON:** 2.2
- **CHECK BEFORE CONTINUING:** the measured numbers, which the build should reproduce. PROTAC: 21,429
  RDKit-parsable rows, 21,429 unique by isomeric canonical SMILES, **19,851 unique once flattened**, median 63
  heavy atoms, p99 108. Glue: **5,216 rows** as of the 2026-10-19 build. An earlier pass recorded 5,998 here;
  that number predates a curation change and the build no longer reproduces it, so 5,216 is the one to check
  against. If you see 5,998 again, something reverted.
- **IF IT FAILED:** a count far from these means the table schema moved.

**Read the flattening numbers before trusting the deduplication.** The build deduplicates on isomeric canonical
SMILES, and on that basis there are zero duplicates in 21,429 rows, which looks implausible for patent-sourced
data and has been questioned as a silently failed deduplication. It has not failed: fed three spellings of
ethanol it returns one molecule. But `use_chirality` is false, so the model trains on flat graphs, and flattening
collapses 1,578 PROTACs and 295 glues. **The real training-set size is 19,851, not 21,429.** Report that number.
Tautomers are not normalized by either path, which is a separate and unaddressed gap.

## Step 2.4: fit the PROTAC surrogates, which is a kill point

**These are not `submit.py` jobs.** A surrogate experiment is a fit script, not
a job directory, so it has no `config.yaml` and `submit.py` will reject it. Run
it directly, on the login node or in an interactive allocation:

```bash
bash experiments/surrogate_protac_optimization/fit.sh
bash experiments/surrogate_protac_evaluation/fit.sh
```

The glue surrogates have their own script and it fits nothing: the largest glue
target carries thirty labelled compounds against a floor of about a hundred.
Run it to reproduce the counts, then read
`experiments/surrogate_mg_optimization/instructions.md` before deciding what
`mg_goal_directed` optimizes:

```bash
bash experiments/surrogate_mg_optimization/fit.sh
```

- **MY TIME:** 45 minutes, most of it reading the held-out scatter
- **WALL CLOCK:** under an hour
- **DEPENDS ON:** 2.3
- **CHECK BEFORE CONTINUING:** the held-out R², Spearman and RMSE against a mean-prediction baseline, printed by
  `scripts/train-surrogate.py` and saved beside the pickle.
- **IF IT FAILED, OR IF R² DOES NOT BEAT THE MEAN BASELINE:** `protac_goal_directed` does not run, and that is
  the finding rather than a setback. Publish the scatter and the baseline either way. SMARCA2 has 241 numeric
  DC50 rows over 186 compounds, which is marginal under a warhead-scaffold split, so expect an R² near zero and
  decide in advance that you will report it.

*`experiments/surrogate_protac_optimization/` exists and carries `protac_pdc50_smarca2`. The fitting script
`scripts/train-surrogate.py` works and has been run on the parsed labels; it has not been run on the cluster.*

---

# Part 3: the PROTAC program

Three experiments plus the selectivity showcase, 21 jobs. Each has its own `instructions.md` carrying the
prerequisites, the submit command, what to check, and the failure mode; this part gives the order and the gates.

Order: `protac_transfer` jobs `preprocess_tpddb` → `prior_scratch`, and in parallel `preprocess_enumerated`
→ `prior_enumerated` → `prior_transfer`, then the two generate jobs. Then `protac_conditional`. Then, gated,
`protac_goal_directed` and the showcase.

**The enumerated library is a blocker for the base prior.** It is not in `download_data.py`. Until it is, the
PROTAC program can run `preprocess_tpddb`, `prior_scratch` and `generate_scratch` only, which means no transfer
result for degraders.

**max_nodes is 108, not 120.** Measured as the 99th percentile on the curated set. The brief said 120; using it
would pad every tensor by 11% in the node dimension and 23% in the edge dimension for nothing.

**The conditioning panel is AR, BTK and EGFR, not AR, BRD4 and SMARCA2.** BRD4 has 140 molecules and SMARCA2 267,
both under the 300 minimum. The substitution order in the brief gives this panel. AR is the easy class and
BTK against EGFR is the hard kinase pair, so the panel can still fail, which was the design intent.

## GATE before any goal-directed submission

```bash
grep -n "composite" src/graphinvent/ScoringFunction.py     # L2: must find the composite score logged
grep -n "n_generated_raw" src/metrics/_core.py             # L3: must find the truncation field
ls src/graphinvent/_metrics.py                             # L1: must NOT exist
```

All three must pass. Until then the budget curve integrates one oracle component rather than the composite
reward, and validity can exceed 1.0 on a good model. Unconditional and conditional training are unaffected and
can proceed.

---

# Part 4: the molecular glue program

Three experiments, 17 jobs. The cheapest program: glues are drug-like, median 34 heavy atoms, `max_nodes` 54.
**Run this program first.** It exercises every code path the PROTAC program uses at a fraction of the allocation,
so a pipeline bug costs hours here rather than days there.

Order: `preprocess_zinc` and `preprocess_glue` in parallel, then `prior_zinc`, then `prior_transfer` and
`prior_scratch`, then the generate pair, then `mg_conditional`, then, gated, `mg_goal_directed`.

**The conditioning panel measures out as specified**: IKZF1/3 merged 1,493 molecules, GSPT1 2,358, RBM39 526, all
clearing 300. IKZF1/3 against GSPT1 is the hard pair, both cereblon neosubstrates reached by glutarimide
chemistry; RBM39 is the positive control through DCAF15 and aryl sulfonamides. If RBM39 does not separate, the
mechanism is broken rather than the task hard.

**`mg_goal_directed` optimizes a chemotype-selectivity classifier, not potency**, because the glue activity data
has 100 numeric DC50 rows in total and RBM39 has none. Whether to run it at all is an open question.

---

# Part 5: the battery program

Three experiments, 18 jobs, and the most blocked of the three. In dependency order the blockers are: the oracle
does not exist; PiNet2 is not installed and its aarch64 viability is unknown; PubChem-120K is not distributed and
cannot be reconstructed without the supporting information; `Batt-P30K.h5` in the source repository is a 134 byte
Git LFS pointer; KBS-409 and the F/P-holdout are not in the source repository either.

What is available from `github.com/Teoroo-CMC/Batt-SLM`: `Batt-SLM/Batt-SLM.smi` at 1.88 MB,
`Batt-SLM/filter_smiles.py`, the CPI implementation with `SolvFunc-87.csv`, and `Redox-Pot/redox_potential.py`.

So `battery_transfer` can run `preprocess_battslm`, `prior_scratch` and `generate_scratch` once the download
is wired, and nothing else in the program can run yet.

## KILL POINT: validate the oracle chain before it scores anything

Run the chain over KBS-409 and confirm the known weakly solvating electrolytes come out with permittivity below
10 and CPI below 0.5 while the conventional dilute electrolytes do not. Weakly solvating: MEK, 2Me-DOL, DIGDBE,
HME, MTBE, OTA. Conventional: DMA, DMF, DEF, MPC, EPC, EB, GBL, MF, EF. **If the separation does not reproduce,
nothing in the battery program runs.**

## KILL POINT: time the oracle before designing around it

The oracle has a conformer step inside it: three lowest-energy ETKDGv3 conformers, MMFF94-optimized, dipole
averaged. Time the full chain on 100 molecules and report seconds per molecule by stage before submitting
anything. 200 steps at batch 256 is 51,200 scorings, so at one second per molecule that is 14 hours of oracle
alone. If it dominates by an order of magnitude, the design changes rather than the walltime.

---

# Part 6: the baselines

**Virtual screening** over each task corpus train and validation split, same oracle, same budget. For the PROTAC
program there is a second and harder screen, surrogate-ranked enumeration over the recombinant library, which is
what a medicinal chemistry team would do and which a reviewer will ask for. Neither is implemented: there is no
screening job type, and the eight `vs_*` metric keys are emitted as nulls with reasons.

**REINVENT4** is not installed. Six fairness conditions must all be documented as met before the comparison is
evidence: pinned version and commit; its prior trained on the same task corpus; matched tuning effort, stated for
both sides; its diversity filter disabled or matched on our side, with both configurations reported; matched
molecular size constraints, which matters most for PROTAC mass; and a shared oracle cache so the call counts mean
the same thing. None is met today. A comparison failing any one of them does not go in the paper.

---

# Part 7: aggregation and the report

Three seeds per reported number, **seeds 1, 2 and 3 rather than 0, 1 and 2**. `main.py` guards seeding on
`seed > 0`, so seed 0 is never applied: two seed-0 generate runs produced different molecules. Until that is
fixed, seed 0 is not a seed.

Seeds 2 and 3 are run by copying the job directory and changing one key, because multi-seed submission does not
exist. The aggregator that writes `experiments/<exp>/results/aggregate.json` does not exist either, so nothing
can yet report a mean and a standard deviation over seeds. Both are open items and both gate every error bar in
the paper.

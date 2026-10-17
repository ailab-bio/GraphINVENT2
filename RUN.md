# RUN.md

Operational runbook for GraphINVENT2 on a SLURM cluster.

**Part 1, below, is the only part written so far. Parts 2 onward are appended
by whoever builds the experiments; do not fold their content into Part 1.**

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

uv pip install "torch>=2.11"      # or: --index-url https://download.pytorch.org/whl/cu130
uv pip install -e .
uv pip install -e ".[surrogates]"
```

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

*Unverified: all of it. The rendering was tested; the submission was not.*

---

## Step 7: find the usable batch size

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

**CPU numbers measured here, as a starting point.** 400 drug-like molecules,
`max_nodes` 70, batch size 250, on a 10-core Apple M-series laptop:

| Job type | Measured | Per unit |
|---|---|---|
| `preprocess` | 9.5 s for 400 molecules | ~24 ms per molecule |
| `unconditional` | 39 s for 2 epochs | ~15 s per epoch over 8,010 subgraphs |
| `generate` | 5.3 s for 1,000 molecules | ~5 ms per molecule |

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

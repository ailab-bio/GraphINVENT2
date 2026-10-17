"""
Check that the whole stack works on this machine, before anything long is run.

Seven checks, in dependency order, stopping at the first failure. Each prints
PASS or FAIL and, on failure, says what was expected and what to do.

    python scripts/cluster_selftest.py            # here, whatever device exists
    python scripts/cluster_selftest.py --require-cuda

``--require-cuda`` turns "no GPU" from a skip into a failure, which is what the
sbatch wrapper passes on a GPU node. Without it the same script is a useful
laptop check.
"""

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "src" / "graphinvent")]

DEV_JOB = REPO_ROOT / "experiments" / "tutorial_unconditional" / "jobs" / "debug"


class Failed(Exception):
    """A check failed; the message says what to do about it."""


def check_python() -> str:
    machine = platform.machine()
    version = platform.python_version()
    if machine not in ("aarch64", "arm64", "x86_64"):
        raise Failed(f"unexpected architecture {machine}")
    return f"{version} on {machine} ({sys.executable})"


def check_torch(require_cuda: bool) -> str:
    import torch

    built = torch.version.cuda
    if not require_cuda:
        return f"torch {torch.__version__}, built against CUDA {built}"
    if built is None:
        raise Failed(
            f"torch {torch.__version__} was built without CUDA. On aarch64 the "
            "plain PyPI wheel is CPU-only below 2.11; install from "
            "https://download.pytorch.org/whl/cu130 or pin torch>=2.11."
        )
    if not torch.cuda.is_available():
        raise Failed(
            f"torch is built against CUDA {built} but no device is visible. "
            "Either this is not a GPU node, or the driver is older than the "
            "toolkit requires. Check nvidia-smi."
        )
    return (
        f"torch {torch.__version__}, CUDA {built}, "
        f"{torch.cuda.get_device_name(0)}, "
        f"capability {'.'.join(map(str, torch.cuda.get_device_capability(0)))}"
    )


def check_rdkit() -> str:
    from rdkit import Chem

    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Failed(f"RDKit could not parse {smiles}")
    if Chem.CanonSmiles(Chem.MolToSmiles(mol)) != Chem.CanonSmiles(smiles):
        raise Failed("RDKit did not round-trip aspirin")
    import rdkit

    return f"rdkit {rdkit.__version__}, aspirin round-tripped"


def check_h5py(data_root: Path) -> str:
    import h5py
    import numpy as np

    data_root.mkdir(parents=True, exist_ok=True)
    path = data_root / f".selftest_{int(time.time())}.h5"
    payload = np.arange(1000, dtype=np.int8).reshape(100, 10)
    try:
        with h5py.File(path, "w") as f:
            f.create_dataset("x", data=payload)
        with h5py.File(path, "r") as f:
            if not np.array_equal(f["x"][:], payload):
                raise Failed(f"data written to {path} read back wrong")
    except OSError as error:
        raise Failed(
            f"could not write {path} from this node: {error}. The data root "
            "must be a filesystem the compute nodes can write to."
        ) from None
    finally:
        path.unlink(missing_ok=True)
    return (
        f"h5py {h5py.__version__} (HDF5 {h5py.version.hdf5_version}), wrote {data_root}"
    )


XGBOOST_PROBE = """
import numpy as np, xgboost as xgb
x = np.random.default_rng(0).normal(size=(64, 4))
m = xgb.XGBClassifier(n_estimators=2, max_depth=2, verbosity=0)
m.fit(x, (x[:, 0] > 0).astype(int))
assert m.predict(x).shape == (64,)
print(xgb.__version__)
"""


def check_xgboost() -> str:
    # In its own process: xgboost and torch each bring their own OpenMP, and
    # importing the second one can abort the interpreter outright rather than
    # raising, which would take this script down with it.
    probe = subprocess.run(
        [sys.executable, "-c", XGBOOST_PROBE], capture_output=True, text=True
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        if any("ModuleNotFoundError" in line for line in detail):
            raise Failed(
                "xgboost is not installed. It is the 'surrogates' extra: "
                "uv sync --extra surrogates. On aarch64 pin xgboost==3.2.0, "
                "which is the first build with GPU support."
            )
        raise Failed(
            "xgboost failed to import or fit: "
            + (detail[-1] if detail else f"exit {probe.returncode}")
        )
    return f"xgboost {probe.stdout.strip()}, trivial model fitted in a subprocess"


def check_model_step(require_cuda: bool) -> str:
    import torch
    from parameters import config

    device = "cuda" if require_cuda else "cpu"
    params = config.resolve(DEV_JOB)
    params["device"] = device
    constants = config.derive(params)
    from gnn.mpnn import GGNN

    torch.manual_seed(0)
    model = GGNN(constants=constants).to(device)
    nodes = torch.zeros(
        4, constants.max_n_nodes, constants.n_node_features, device=device
    )
    nodes[:, 0, 0] = 1.0
    edges = torch.zeros(
        4,
        constants.max_n_nodes,
        constants.max_n_nodes,
        constants.n_edge_features,
        device=device,
    )
    output = model(nodes, edges)
    output.sum().backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    nonfinite = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and not p.grad.isfinite().all()
    ]
    if nonfinite:
        raise Failed(f"non-finite gradients in {nonfinite[:3]}")
    return (
        f"GGNN forward+backward on {device}, output {tuple(output.shape)}, "
        f"{len(missing)} parameters without gradient"
    )


def check_development_job() -> str:
    result = subprocess.run(
        [sys.executable, "submit.py", str(DEV_JOB)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr).strip().splitlines()[-15:]
        raise Failed("the development job failed:\n    " + "\n    ".join(tail))

    # These three stand in for `results/metrics.json`, which is what a check of
    # "the job produced its outputs" should really assert on. No job writes one
    # yet; see the Status section of CLAUDE.md. Swap these for metrics.json once
    # it exists, rather than adding to them.
    produced = [
        DEV_JOB / "results" / "config.resolved.yaml",
        DEV_JOB / "logs" / "convergence.log",
    ]
    missing = [p for p in produced if not p.exists()]
    if missing:
        raise Failed(f"the job finished but did not write {missing}")
    checkpoints = sorted((DEV_JOB / "checkpoints").glob("model_restart_*.pth"))
    if not checkpoints:
        raise Failed(f"no checkpoint under {DEV_JOB / 'checkpoints'}")
    return (
        f"{DEV_JOB.relative_to(REPO_ROOT)} completed, {len(checkpoints)} checkpoint(s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail, rather than skip, when no CUDA device is visible",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="filesystem to test writing to; defaults to data_root in the config",
    )
    args = parser.parse_args()

    if args.data_root is None:
        import yaml

        defaults = yaml.safe_load((REPO_ROOT / "config" / "defaults.yaml").read_text())
        args.data_root = REPO_ROOT / defaults["data_root"]

    checks = [
        ("python interpreter and architecture", check_python),
        ("torch and CUDA", lambda: check_torch(args.require_cuda)),
        ("rdkit SMILES round trip", check_rdkit),
        ("h5py write and read on the data root", lambda: check_h5py(args.data_root)),
        ("xgboost fit", check_xgboost),
        ("GGNN forward and backward", lambda: check_model_step(args.require_cuda)),
        ("development job end to end", check_development_job),
    ]

    width = max(len(name) for name, _ in checks)
    failures = 0
    for number, (name, check) in enumerate(checks, 1):
        started = time.perf_counter()
        try:
            detail = check()
        except Failed as error:
            print(f"{number}. FAIL  {name:<{width}}  {error}", flush=True)
            failures += 1
            break
        except Exception as error:  # an unexpected error is still a failure
            print(
                f"{number}. FAIL  {name:<{width}}  {type(error).__name__}: {error}",
                flush=True,
            )
            failures += 1
            break
        print(
            f"{number}. PASS  {name:<{width}}  {detail}  "
            f"[{time.perf_counter() - started:.1f}s]",
            flush=True,
        )

    total = len(checks)
    if failures:
        print(f"\nFAIL: stopped at check {number} of {total}. Fix it and rerun.")
        return 1
    print(f"\nPASS: all {total} checks passed. The stack works on this node.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

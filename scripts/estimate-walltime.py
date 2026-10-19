#!/usr/bin/env python3
"""
Derive a walltime and a memory request for every job, by RUN.md's procedure.

    python scripts/estimate-walltime.py            # print the table
    python scripts/estimate-walltime.py --write    # write into each config.yaml

The procedure is RUN.md Part 1: take a measured per-unit cost, extrapolate
linearly in epochs or in molecules, add the fixed costs, multiply by 1.5 and
round up to the hour. The per-unit costs are the ones measured on a 10-core
Apple M-series laptop CPU and recorded in RUN.md and in
manuscript/experiment-plan.md. Nothing here has been measured on a GPU.

**Every number this prints is a CPU extrapolation.** On a GPU the training
jobs will finish well inside it and the generation jobs will not gain nearly as
much, since generation is a sequential autoregressive loop with a Python-level
graph-to-molecule decode. That asymmetry is why a CPU extrapolation is the
right thing to request: it is an upper bound for the job types that gain most,
and a job killed at 99% has produced nothing but its last checkpoint.

A job whose dataset does not exist yet cannot be sized, and this says so rather
than printing a number derived from a molecule count nobody measured.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Measured CPU costs. Two anchors, because cost per molecule is set by molecule
# size and the two corpora differ by a factor of three in subgraphs per
# molecule: RUN.md's drug-like set at max_nodes 70, and
# manuscript/experiment-plan.md's PROTAC set at max_nodes 108.
ANCHORS = {
    # max_nodes: (subgraphs per molecule, preprocess s/molecule, epoch s/subgraph)
    70: (23.5, 0.045, 0.00266),
    108: (60.8, 0.157, 0.00516),
}

# generate: a fixed start-up plus a flat per-molecule cost, both from RUN.md's
# four-point fit. Scaled by molecule size on the same anchor as the rest.
GENERATE_FIXED_S = 6.0
GENERATE_PER_MOLECULE_S = 0.021

# Decomposition, for a bifunctional dataset only. The adaptive model's measured
# pooled rate (0.483 s per molecule across eight interpreters, 2026-10-19) over
# the subsampled component set. It does not shrink on a GPU node, because the
# splitter runs on CPU in its own interpreters.
SPLITTER_S_PER_MOLECULE = 0.483
COMPONENT_SAMPLE_SIZE = 2000

SAFETY = 1.5

# Bytes of resident memory per loaded subgraph, from the loader's block: the
# 10,000-row block measured at ~1.9 GB on the PROTAC set.
BYTES_PER_SUBGRAPH = 190_000
MEMORY_FLOOR_GB = 8


def anchor_for(max_nodes: int) -> tuple:
    """The measured anchor nearest this molecule size, interpolating between."""
    sizes = sorted(ANCHORS)
    if max_nodes <= sizes[0]:
        return ANCHORS[sizes[0]]
    if max_nodes >= sizes[-1]:
        return ANCHORS[sizes[-1]]
    low, high = sizes[0], sizes[1]
    weight = (max_nodes - low) / (high - low)
    return tuple(
        ANCHORS[low][i] + weight * (ANCHORS[high][i] - ANCHORS[low][i])
        for i in range(3)
    )


def molecules_in(dataset_dir: Path) -> int | None:
    """Training-split molecules, or None when the dataset is not built."""
    path = dataset_dir / "train.smi"
    if not path.exists():
        return None
    return max(0, sum(1 for line in path.read_text().splitlines() if line.strip()) - 1)


def resolve(job: Path) -> dict:
    """config/defaults.yaml, then experiment.yaml, then the job's config.yaml."""
    merged = yaml.safe_load((ROOT / "config" / "defaults.yaml").read_text()) or {}
    experiment = job.parent.parent / "experiment.yaml"
    for path in (experiment, job / "config.yaml"):
        if not path.exists():
            continue
        layer = yaml.safe_load(path.read_text()) or {}
        for key, value in layer.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


def estimate(job: Path) -> dict:
    """Seconds and gigabytes for one job, or the reason it cannot be sized."""
    params = resolve(job)
    job_type = params.get("job_type")
    max_nodes = int(params.get("max_n_nodes") or 70)
    per_molecule_subgraphs, preprocess_s, epoch_s = anchor_for(max_nodes)

    data_root = Path(params.get("data_root") or (ROOT / "data" / "datasets"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    dataset = data_root / str(params.get("dataset_dir") or "")
    n_molecules = molecules_in(dataset)

    needs_dataset = job_type in ("preprocess", "unconditional", "conditional")
    if needs_dataset and n_molecules is None:
        return {
            "missing": f"{dataset.relative_to(ROOT) if ROOT in dataset.parents else dataset}"
            "/train.smi does not exist, so the molecule count the extrapolation "
            "needs is unknown. Build the dataset first."
        }

    n_samples = int(params.get("n_samples") or 0)
    evaluations = 0
    seconds = 0.0
    note = []

    if job_type == "preprocess":
        seconds = n_molecules * preprocess_s
        note.append(f"{n_molecules:,} molecules x {preprocess_s * 1000:.0f} ms")
    elif job_type in ("unconditional", "conditional"):
        subgraphs = n_molecules * per_molecule_subgraphs
        epochs = int(params.get("epochs") or 0)
        seconds = epochs * subgraphs * epoch_s
        note.append(
            f"{epochs} epochs x {subgraphs:,.0f} subgraphs x {epoch_s * 1000:.2f} ms"
        )
        sample_every = int(params.get("sample_every") or 0)
        evaluations = (epochs // sample_every if sample_every else 0) + 1
    elif job_type == "generate":
        seconds = GENERATE_FIXED_S + n_samples * GENERATE_PER_MOLECULE_S * (
            per_molecule_subgraphs / ANCHORS[70][0]
        )
        note.append(f"{n_samples:,} molecules, scaled to max_nodes {max_nodes}")
    elif job_type == "goal_directed":
        steps = int(params.get("rl_steps") or 0)
        batch = int(params.get("rl_batch_size") or params.get("batch_size") or 0)
        seconds = (
            steps
            * batch
            * GENERATE_PER_MOLECULE_S
            * (per_molecule_subgraphs / ANCHORS[70][0])
            * 2
        )
        note.append(f"{steps} steps x {batch} molecules x 2 (agent and best-agent)")
        eval_every = int(params.get("rl_eval_every") or 0)
        evaluations = (steps // eval_every if eval_every else 0) + 1
    else:
        return {"missing": f"no cost model for job_type {job_type!r}"}

    if evaluations:
        evaluation_cost = evaluations * (
            GENERATE_FIXED_S
            + int(params.get("eval_sample_size") or n_samples or 0)
            * GENERATE_PER_MOLECULE_S
            * (per_molecule_subgraphs / ANCHORS[70][0])
        )
        seconds += evaluation_cost
        note.append(f"+{evaluations} evaluation rounds")

    # Decomposition, which every bifunctional job that writes metrics pays once
    # over the subsampled component set. It is not a GPU cost and does not
    # shrink on a GPU node.
    #
    # Counted once, not twice. The reference warheads that `warhead_novelty` is
    # measured against are decomposed from the training split, which the
    # mandated `warhead_scaffold` split already did molecule by molecule, so
    # the splitter cache serves them. On a dataset split some other way the
    # first bifunctional job pays a second 2,000-molecule decomposition and
    # every later one is served from the cache; add an hour to the first job if
    # that is the case.
    if params.get("bifunctional") and job_type in (
        "generate",
        "goal_directed",
        "conditional",
        "unconditional",
    ):
        decomposition = COMPONENT_SAMPLE_SIZE * SPLITTER_S_PER_MOLECULE
        seconds += decomposition
        note.append(
            f"+{decomposition / 3600:.1f} h decomposition "
            f"({COMPONENT_SAMPLE_SIZE} molecules x {SPLITTER_S_PER_MOLECULE} s)"
        )

    hours = max(1, math.ceil(seconds * SAFETY / 3600))

    block = int(params.get("block_size") or 10000)
    memory_gb = max(
        MEMORY_FLOOR_GB,
        math.ceil(block * BYTES_PER_SUBGRAPH * (max_nodes / 108) / 1e9 * 2),
    )
    return {
        "hours": hours,
        "memory_gb": memory_gb,
        "seconds": seconds,
        "note": "; ".join(note),
        "job_type": job_type,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="write the numbers into each config.yaml"
    )
    args = parser.parse_args()

    jobs = sorted((ROOT / "experiments").glob("*/jobs/*/config.yaml"))
    unresolved: list = []
    written = 0
    for config_path in jobs:
        job = config_path.parent
        name = str(job.relative_to(ROOT))
        if job.name.startswith("smoke") or "tutorial" in name:
            # Laptop jobs. They are never submitted, so they have no walltime
            # to derive and the SLURM default would never be reached.
            continue
        result = estimate(job)
        if "missing" in result:
            unresolved.append((name, result["missing"]))
            print(f"UNSIZED  {name}\n         {result['missing']}")
            continue
        print(
            f"{result['hours']:>3d} h  {result['memory_gb']:>3d} GB  {name}\n"
            f"         {result['note']}"
        )
        if args.write:
            written += write_into(config_path, result)

    print(
        f"\n{len(jobs)} job directories, "
        f"{len(unresolved)} could not be sized, {written} written."
    )
    return 0


def write_into(config_path: Path, result: dict) -> int:
    """Replace the job's slurm time and memory, with the derivation beside it."""
    text = config_path.read_text()
    derivation = [
        "# WALLTIME AND MEMORY DERIVED, NOT MEASURED ON THE CLUSTER. By the",
        "# RUN.md Part 1 procedure, from the CPU per-unit costs in RUN.md and",
        "# manuscript/experiment-plan.md, times 1.5, rounded up to the hour:",
        f"#   {result['note']}",
        f"#   = {result['seconds'] / 3600:.2f} h x 1.5 -> {result['hours']} h",
        "# This is a CPU extrapolation. Training will finish well inside it on a",
        "# GPU; generation gains much less, because it is a sequential",
        "# autoregressive loop with a Python-level decode. Re-measure on",
        "# Arrhenius and replace these two numbers with what was observed.",
    ]
    lines = text.splitlines()
    out: list = []
    index = 0
    while index < len(lines):
        line = lines[index]
        # Drop any previous derivation block, however it was worded.
        if line.startswith("# WALLTIME AND MEMORY"):
            while index < len(lines) and lines[index].startswith("#"):
                index += 1
            continue
        if line.rstrip() == "slurm:":
            index += 1
            # Everything indented under slurm:, minus the two keys replaced.
            kept: list = []
            while index < len(lines) and (
                lines[index].startswith((" ", "\t")) or not lines[index].strip()
            ):
                stripped = lines[index].strip()
                if not stripped.startswith(("time:", "memory:")):
                    kept.append(lines[index])
                index += 1
            out.extend(derivation)
            out.append("slurm:")
            out.append(f'  time: "{result["hours"]:02d}:00:00"')
            out.append(f"  memory: {result['memory_gb']}G")
            out.extend(kept)
            continue
        out.append(line)
        index += 1
    config_path.write_text("\n".join(out).rstrip("\n") + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())

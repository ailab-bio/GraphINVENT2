"""
Run goal-directed optimization for every oracle defined in oracles_config.yaml.

Usage (from repository root):
    python experiments/goal_directed/run_all_oracles.py [options]

Options:
    --config  Path to oracles_config.yaml (default: experiments/goal_directed/oracles_config.yaml)
    --pretrained-model  Path to ChEMBL pretrained checkpoint (overrides template default)
    --oracle  Run only this oracle (can be repeated)
    --seed    Run only this seed (can be repeated; default: all seeds from config)
    --dry-run Print commands without executing them

Each (oracle, seed) combination is launched as a separate submit.py call with
its own output directory:
    output/chembl_v34/goal_directed/<oracle>_seed<seed>/

A resolved params.json is written next to each run so you can re-run or inspect it.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE_PATH = REPO_ROOT / "experiments" / "goal_directed" / "rl_params_template.json"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "goal_directed" / "oracles_config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_template() -> dict:
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def load_oracle_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_job_config(
    template: dict,
    oracle_name: str,
    oracle_threshold: float | None,
    seed: int,
    pretrained_model_path: str | None,
) -> dict:
    """Fill in template placeholders for a specific (oracle, seed) combination."""
    cfg = copy.deepcopy(template)

    # Score component
    cfg["job"]["score_components"] = [oracle_name]
    cfg["job"]["score_thresholds"] = [
        oracle_threshold if oracle_threshold is not None else 0.0
    ]

    # Job name encodes oracle + seed for a unique output directory
    job_name = f"{oracle_name.replace(':', '_')}_seed{seed}"
    cfg["submission"]["job_name"] = job_name

    # Seed
    cfg["job"]["seed"] = seed

    # Pretrained model
    if pretrained_model_path:
        cfg["job"]["pretrained_model_path"] = pretrained_model_path

    # Remove template placeholder strings left in the config
    cfg["job"].pop("_oracle_note", None)
    cfg["job"].pop("_pretrained_note", None)
    cfg["job"].pop("_budget_note", None)
    cfg["job"].pop("_rl_note", None)

    return cfg


def write_config(cfg: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "params.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg_path


def run_job(cfg_path: Path, dry_run: bool) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "submit.py"), "--config", str(cfg_path)]
    print(f"  Running: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to oracles_config.yaml",
    )
    p.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Path to ChEMBL pretrained checkpoint (overrides template)",
    )
    p.add_argument(
        "--oracle",
        action="append",
        default=None,
        dest="oracles",
        help="Run only this oracle (repeatable)",
    )
    p.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        dest="seeds",
        help="Run only this seed (repeatable)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    oracle_cfg = load_oracle_config(args.config)
    template = load_template()

    pmo_cfg = oracle_cfg.get("pmo", {})
    all_seeds = args.seeds if args.seeds else pmo_cfg.get("seeds", [42, 123, 456])
    all_oracles = oracle_cfg.get("oracles", [])

    # Filter to requested oracles
    if args.oracles:
        all_oracles = [o for o in all_oracles if o["name"] in args.oracles]
        if not all_oracles:
            print(
                f"No matching oracles found for {args.oracles}. Check oracles_config.yaml."
            )
            sys.exit(1)

    print(f"Oracles:  {[o['name'] for o in all_oracles]}")
    print(f"Seeds:    {all_seeds}")
    print(f"Dry-run:  {args.dry_run}")
    print()

    configs_dir = REPO_ROOT / "experiments" / "goal_directed" / "configs"

    total = len(all_oracles) * len(all_seeds)
    idx = 0

    for oracle in all_oracles:
        oracle_name = oracle["name"]
        threshold = oracle.get("threshold")

        for seed in all_seeds:
            idx += 1
            print(f"[{idx}/{total}] Oracle={oracle_name!r}  seed={seed}")

            cfg = make_job_config(
                template=template,
                oracle_name=oracle_name,
                oracle_threshold=threshold,
                seed=seed,
                pretrained_model_path=args.pretrained_model,
            )

            out_dir = configs_dir / f"{oracle_name.replace(':', '_')}_seed{seed}"
            cfg_path = write_config(cfg, out_dir)
            print(f"  Config:  {cfg_path.relative_to(REPO_ROOT)}")

            try:
                run_job(cfg_path, dry_run=args.dry_run)
                print("  Status:  OK\n")
            except subprocess.CalledProcessError as exc:
                print(f"  Status:  FAILED (return code {exc.returncode})\n")


if __name__ == "__main__":
    main()

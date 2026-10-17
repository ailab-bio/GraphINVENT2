"""
results/report.html: one self-contained page per run, opened by double-clicking.

No external assets. Inline CSS, RDKit SVG inlined directly, no script tags, no
CDN, no sibling image files. It has to render from file:// with no network.

The numbers come from metrics.json and are not recomputed: CLAUDE.md allows one
implementation per metric, and a report that recalculates is a second one that
will drift. The per-molecule annotations under each structure -- weight, logP,
QED, SA -- are the exception, and deliberately so: they are properties of one
drawn molecule, not metrics of the run, and there is nowhere in a flat metrics
dictionary for twenty of them to live.
"""

from __future__ import annotations

import html
import subprocess
from pathlib import Path

from .plot_style import AIME_PALETTE, DARK_SLATE

# How many structures the diversity grid shows, and the seed that fixes which.
GRID_SIZE = 20
PICKER_SEED = 42

# The metric groups, in CLAUDE.md's order. Keys not listed here still appear,
# under "Other", so a metric can never be added and silently not shown.
GROUPS: list = [
    (
        "Core",
        [
            "validity",
            "uniqueness",
            "novelty",
            "vun",
            "near_duplicate_rate",
            "internal_diversity",
            "scaffold_diversity",
            "sa_score_mean",
            "sa_score_median",
            "filter_pass_rate",
            "Ro5_pass_rate",
            "bRo5_pass_rate",
        ],
    ),
    (
        "Unconditional (nll_* are per molecule, summed over construction actions)",
        ["fcd", "nll_train", "nll_validation", "nll_test", "uc_jsd"],
    ),
    (
        "Goal-directed",
        [
            "best_at_1k",
            "best_at_5k",
            "best_at_10k",
            "top10_mean_at_1k",
            "top10_mean_at_5k",
            "top10_mean_at_10k",
            "top100_mean_at_1k",
            "top100_mean_at_5k",
            "top100_mean_at_10k",
            "auc_top10",
            "success_rate",
            "top100_scaffold_count",
            "top100_internal_diversity",
            "quality_filtered_top10_mean",
            "eval_oracle_top10_mean",
            "top100_outside_domain_rate",
            "vs_baseline_top10_at_1k",
            "vs_baseline_top10_at_5k",
            "vs_baseline_top10_at_10k",
            "vs_delta_top10_at_1k",
            "vs_delta_top10_at_5k",
            "vs_delta_top10_at_10k",
        ],
    ),
    (
        "Model efficiency",
        ["n_parameters", "train_wall_clock_s", "inference_s_per_10k", "peak_memory_mb"],
    ),
    (
        "Counts",
        [
            "n_sampled",
            "n_valid",
            "n_unique_valid",
            "n_unique_valid_novel",
            "n_scaffolds",
            "success_threshold",
        ],
    ),
]

CSS = f"""
:root {{ color-scheme: light; }}
body {{
  margin: 0; padding: 2rem 1.5rem; background: #fbfbfa; color: {DARK_SLATE};
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  font-size: 14px; line-height: 1.5;
}}
.wrap {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ font-size: 1.35rem; margin: 0 0 0.25rem; }}
h2 {{ font-size: 1.05rem; margin: 2rem 0 0.5rem; border-bottom: 1.2px solid {DARK_SLATE};
      padding-bottom: 0.25rem; }}
.meta {{ color: #5b5f6b; font-size: 12.5px; margin-bottom: 0.5rem; }}
.meta code {{ background: #efefec; padding: 0.1em 0.35em; border-radius: 3px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 0.5rem; }}
th, td {{ text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid #e3e3df; }}
th {{ font-weight: 600; font-size: 12.5px; color: #5b5f6b; }}
td.value {{ font-variant-numeric: tabular-nums; text-align: right; width: 12rem; }}
td.unavailable {{ color: #8a8d98; font-style: italic; text-align: right; }}
.reason {{ color: #8a8d98; font-size: 12px; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 0.75rem; }}
.cell {{ border: 1.2px solid {DARK_SLATE}; border-radius: 4px; padding: 0.5rem;
        width: 200px; background: #fff; }}
.cell svg {{ display: block; margin: 0 auto; }}
.smiles {{ font-family: ui-monospace, Menlo, monospace; font-size: 10.5px;
          word-break: break-all; color: #5b5f6b; margin: 0.35rem 0 0.2rem; }}
.props {{ font-size: 11.5px; font-variant-numeric: tabular-nums; }}
.score {{ font-size: 12.5px; font-weight: 600; color: {AIME_PALETTE[0]}; }}
.note {{ color: #5b5f6b; font-size: 12.5px; margin: 0.25rem 0 0.75rem; }}
pre.config {{ background: #efefec; border-radius: 4px; padding: 0.75rem;
             font-size: 10.5px; line-height: 1.35; overflow-x: auto;
             white-space: pre-wrap; word-break: break-word; }}
"""


def _git_commit(repo_root: Path) -> str | None:
    """
    The commit, only when the tree is clean.

    A hash next to results from a dirty tree names a state that never existed,
    which is worse than naming nothing.
    """
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if status.returncode != 0 or status.stdout.strip():
            return None
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return rev.stdout.strip() or None if rev.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _format(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join("--" if v is None else f"{v:.3f}" for v in value)
    return html.escape(str(value))


def _metric_rows(metrics: dict, unavailable: dict, keys: list) -> str:
    rows: list = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if value is None:
            reason = html.escape(unavailable.get(key, "undefined for this run"))
            rows.append(
                f"<tr><td>{html.escape(key)}</td>"
                f'<td class="unavailable">null</td>'
                f'<td class="reason">{reason}</td></tr>'
            )
        else:
            rows.append(
                f"<tr><td>{html.escape(key)}</td>"
                f'<td class="value">{_format(value)}</td><td></td></tr>'
            )
    if not rows:
        return ""
    return (
        '<table><tr><th>metric</th><th style="text-align:right">value</th>'
        "<th>note</th></tr>" + "".join(rows) + "</table>"
    )


def _svg(smiles: str) -> str:
    """An inline SVG of one molecule, in the palette's text color."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(184, 140)
    options = drawer.drawOptions()
    options.clearBackground = False
    red, green, blue = (
        int(DARK_SLATE[1:3], 16) / 255,
        int(DARK_SLATE[3:5], 16) / 255,
        int(DARK_SLATE[5:7], 16) / 255,
    )
    options.setSymbolColour((red, green, blue))
    options.setAnnotationColour((red, green, blue))
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _properties(smiles: str) -> str:
    from rdkit import Chem

    from ._properties import logp, mol_weight, qed, sa_score

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return (
        f"MW {mol_weight(mol):.1f} &middot; logP {logp(mol):.2f}<br>"
        f"QED {qed(mol):.3f} &middot; SA {sa_score(mol):.2f}"
    )


def diverse_scaffold_representatives(smiles: list, n: int = GRID_SIZE):
    """
    Up to `n` structurally diverse molecules, and how many scaffolds there were.

    Unique valid molecules, grouped by Bemis-Murcko scaffold, one representative
    per scaffold -- the first in canonical SMILES sort order, so the choice does
    not depend on generation order. When more than `n` scaffolds remain, the
    scaffolds are picked apart with MaxMinPicker over ECFP4 of the scaffolds
    themselves under a fixed seed, so the page is reproducible.
    """
    from rdkit import Chem
    from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

    from ._core import bemis_murcko, ecfp4

    by_scaffold: dict = {}
    for smi in sorted(set(s for s in smiles if s)):
        scaffold = bemis_murcko(smi)
        if scaffold is None:
            continue
        by_scaffold.setdefault(scaffold, []).append(smi)

    scaffolds = sorted(by_scaffold)
    n_scaffolds = len(scaffolds)
    if n_scaffolds <= n:
        chosen = scaffolds
    else:
        fps = []
        usable: list = []
        for scaffold in scaffolds:
            # An acyclic molecule has the empty scaffold, which cannot be
            # fingerprinted; it is kept as a group but cannot be picked apart.
            mol = Chem.MolFromSmiles(scaffold) if scaffold else None
            if mol is None or mol.GetNumAtoms() == 0:
                continue
            fps.append(ecfp4(mol))
            usable.append(scaffold)
        if len(usable) <= n:
            chosen = usable or scaffolds[:n]
        else:
            picker = MaxMinPicker()
            indices = picker.LazyBitVectorPick(fps, len(fps), n, seed=PICKER_SEED)
            chosen = sorted(usable[i] for i in indices)

    return [sorted(by_scaffold[s])[0] for s in chosen], n_scaffolds


def _cells(entries: list) -> str:
    """entries: list of (smiles, extra_html_or_None)."""
    cells: list = []
    for smiles, extra in entries:
        svg = _svg(smiles)
        if not svg:
            continue
        extra_html = f'<div class="score">{extra}</div>' if extra else ""
        cells.append(
            f'<div class="cell">{svg}'
            f'<div class="smiles">{html.escape(smiles)}</div>'
            f"{extra_html}"
            f'<div class="props">{_properties(smiles)}</div></div>'
        )
    return f'<div class="grid">{"".join(cells)}</div>'


def render(
    *,
    payload: dict,
    results_dir: str,
    job_dir: str,
    grid_smiles: list | None = None,
    scored: list | None = None,
    conditions: list | None = None,
) -> Path:
    """
    Write results/report.html and return its path.

    `grid_smiles` are the unique valid generated molecules the diversity grid
    is drawn from. `scored` is (smiles, score) for a goal-directed run, which
    gets a second grid: diverse scaffolds alone hide the optimization outcome,
    and the top-20 grid is where reward hacking becomes visible to the eye.
    `conditions` is (smiles, requested, achieved) for a conditional run.
    """
    run = payload["run"]
    metrics = payload["metrics"]
    unavailable = payload.get("unavailable", {})
    repo_root = Path(__file__).resolve().parents[2]

    parts: list = []
    parts.append(f"<h1>{html.escape(str(run['job_name']))}</h1>")

    commit = _git_commit(repo_root)
    meta = (
        f"<b>{html.escape(str(run['job_type']))}</b> &middot; dataset "
        f"<code>{html.escape(str(run['dataset']))}</code> &middot; seed "
        f"{html.escape(str(run['seed']))} &middot; device "
        f"<code>{html.escape(str(run['device']))}</code><br>"
        f"{html.escape(str(run['timestamp']))}"
    )
    meta += (
        f" &middot; commit <code>{html.escape(commit)}</code>"
        if commit
        else " &middot; no commit recorded (working tree not clean)"
    )
    parts.append(f'<div class="meta">{meta}</div>')

    shown: set = set()
    for title, keys in GROUPS:
        table = _metric_rows(metrics, unavailable, keys)
        shown |= {k for k in keys if k in metrics}
        if table:
            parts.append(f"<h2>{html.escape(title)}</h2>{table}")

    # Anything the groups above do not name, so a new metric cannot go unshown.
    leftover = sorted(set(metrics) - shown)
    conditional = [
        k for k in leftover if k.startswith(("condition_", "baseline_condition_"))
    ]
    categorical = [k for k in leftover if k.startswith(("class_", "knn_", "delta_nll"))]
    components = [k for k in leftover if k.startswith(("warhead_", "split_", "e3_"))]
    other = [k for k in leftover if k not in conditional + categorical + components]
    for title, keys in (
        ("Conditional, continuous", conditional),
        ("Conditional, categorical", categorical),
        ("Component-level", components),
        ("Other", other),
    ):
        table = _metric_rows(metrics, unavailable, keys)
        if table:
            parts.append(f"<h2>{html.escape(title)}</h2>{table}")

    if grid_smiles:
        chosen, n_scaffolds = diverse_scaffold_representatives(grid_smiles)
        note = (
            f"{len(chosen)} of {n_scaffolds} distinct Bemis-Murcko scaffolds, "
            f"picked apart with MaxMinPicker over ECFP4 (seed {PICKER_SEED})."
            if n_scaffolds > GRID_SIZE
            else (
                f"All {n_scaffolds} distinct Bemis-Murcko "
                f"scaffold{'' if n_scaffolds == 1 else 's'} in the sample."
            )
        )
        by_smiles = dict(conditions and [(c[0], c) for c in conditions] or [])
        entries = []
        for smi in chosen:
            extra = None
            if smi in by_smiles:
                _, requested, achieved = by_smiles[smi]
                extra = (
                    f"requested {requested:.2f} &middot; achieved {achieved:.2f}"
                    if requested is not None and achieved is not None
                    else None
                )
            entries.append((smi, extra))
        parts.append(
            f"<h2>Structural diversity</h2>"
            f'<div class="note">{html.escape(note)}</div>{_cells(entries)}'
        )

    if scored:
        top = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:GRID_SIZE]
        parts.append(
            "<h2>Highest oracle score</h2>"
            '<div class="note">The top 20 by oracle score. Compare with the '
            "grid above: overall diversity stays high while a reward-hacked "
            "top set collapses onto one scaffold.</div>"
            + _cells([(smi, f"score {score:.4f}") for smi, score in top])
        )

    config_path = Path(results_dir) / "config.resolved.yaml"
    if config_path.exists():
        parts.append(
            "<h2>Resolved configuration</h2>"
            f'<pre class="config">{html.escape(config_path.read_text())}</pre>'
        )

    document = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(str(run['job_name']))} — GraphINVENT2</title>"
        f'<style>{CSS}</style></head><body><div class="wrap">'
        + "".join(parts)
        + "</div></body></html>\n"
    )
    path = Path(results_dir) / "report.html"
    path.write_text(document)
    return path

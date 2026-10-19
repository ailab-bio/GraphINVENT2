"""
The per-job figures, one function per job type.

Every figure imports the style from `plot_style` and sets nothing else: no
per-plot colors, fonts or sizes, no title inside the axes, panel labels outside
them, all four spines, Dark Slate for text and axes, axis labels with units,
legends without a frame. A categorical distinction is carried by marker or line
style as well as by color, so a figure survives being printed in grayscale.

Each figure is saved into the job's `results/` as both `.png` at 300 dpi and
`.pdf`, and `report.html` embeds the PNG. A figure whose input is missing is
skipped with a line saying what was missing, because a job that cannot draw one
panel should still write the others.
"""

from __future__ import annotations

import json
from pathlib import Path

from .plot_style import DARK_SLATE, add_panel_label, apply, save_figure

# A categorical distinction is carried by marker and line style as well as by
# color, in this order, so two series are told apart in grayscale.
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", ":", "-."]


def _skip(what: str, why: str) -> None:
    print(f"-- Skipping the {what} figure: {why}.", flush=True)


# Inches of figure width per panel. Wide enough that the mandated panel label
# at x = -0.18 in axes coordinates clears the y-axis label; at 4.3 it lands on
# top of it.
PANEL_WIDTH = 4.9
PANEL_HEIGHT = 3.6


def _grid(n_panels: int):
    """
    A figure with exactly `n_panels` axes, or `(None, [])` for none.

    Panels are allocated after it is known which have data, never before. An
    axes left empty because its input was missing renders as a blank box with
    default 0-1 ticks, which reads as a measurement that came out empty rather
    than as one that was not made.
    """
    import matplotlib.pyplot as plt

    if n_panels <= 0:
        return None, []
    # The style is installed here and nowhere else, so no figure in this module
    # can be drawn without it. It used to be a call at the top of each entry
    # point, and `generate_figures` lost its during a refactor: the figure
    # still drew, in default matplotlib blue, and nothing failed.
    apply()
    fig, axes = plt.subplots(
        1, n_panels, figsize=(PANEL_WIDTH * n_panels, PANEL_HEIGHT)
    )
    return fig, list(axes) if n_panels > 1 else [axes]


def _label_and_save(fig, axes, results_dir: str, name: str):
    """Panel labels outside the axes, then PNG and PDF into `results/`."""
    import matplotlib.pyplot as plt

    for ax, label in zip(axes, "abcdef"):
        add_panel_label(ax, f"({label})")
    fig.tight_layout(w_pad=3.0)
    paths = save_figure(fig, results_dir, name)
    plt.close(fig)
    return paths


def _heavy_atoms(smiles: list) -> list:
    from rdkit import Chem

    counts = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            counts.append(mol.GetNumHeavyAtoms())
    return counts


def preprocess_figures(results_dir: str, splits: dict, max_nodes, strategy, seed):
    """
    (a) the heavy-atom distribution with `max_nodes` marked, and (b) the
    molecule count per split, with the strategy and seed in the axis label.

    `max_nodes` is marked because it is the one preprocessing parameter taken
    from the data rather than chosen, and a distribution with a tail past the
    line is molecules the representation cannot hold.
    """

    if not any(splits.values()):
        _skip("preprocessing", "no split file held any molecule")
        return []
    fig, axes = _grid(2)

    counts = _heavy_atoms([s for split in splits.values() for s in split])
    axes[0].hist(counts, bins=40)
    if max_nodes:
        axes[0].axvline(float(max_nodes), linestyle="--", color=DARK_SLATE)
        axes[0].annotate(
            f"max_nodes = {int(max_nodes)}",
            xy=(float(max_nodes), 0.9),
            xycoords=("data", "axes fraction"),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            fontsize=9.5,
            color=DARK_SLATE,
        )
    axes[0].set_xlabel("heavy atoms per molecule (count)")
    axes[0].set_ylabel("molecules (count)")

    names = list(splits)
    axes[1].bar(names, [len(splits[n]) for n in names])
    for index, name in enumerate(names):
        axes[1].annotate(
            f"{len(splits[name]):,}",
            xy=(index, len(splits[name])),
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=DARK_SLATE,
        )
    axes[1].set_xlabel(f"split ({strategy}, seed {seed})")
    axes[1].set_ylabel("molecules (count)")

    return _label_and_save(fig, axes, results_dir, "preprocessing")


def training_figure(results_dir: str, convergence_log: str, control_log=None):
    """
    Train and validation loss against epoch, on one axes.

    One axes and not two, because the gap between the two curves is the thing
    being read and two panels make it a comparison between figures. A transfer
    job overlays the from-scratch control, which is the only way the transfer
    claim is legible: the same final loss reached in fewer epochs is the claim,
    and a curve on its own cannot show it.
    """

    series = _read_convergence(convergence_log)
    if not series:
        _skip("training-curve", f"no readable convergence log at {convergence_log}")
        return []
    fig, (ax,) = _grid(1)

    for index, (label, values) in enumerate(series.items()):
        ax.plot(
            values["x"],
            values["y"],
            label=label,
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            markevery=max(1, len(values["x"]) // 12),
        )
    control = _read_convergence(control_log) if control_log else {}
    for index, (label, values) in enumerate(control.items(), start=len(series)):
        ax.plot(
            values["x"],
            values["y"],
            label=f"{label}, from scratch",
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            markevery=max(1, len(values["x"]) // 12),
        )

    ax.set_xlabel("epoch (count)")
    ax.set_ylabel("loss (nats per action)")
    ax.legend()
    return _label_and_save(fig, [ax], results_dir, "training_curves")


def _read_convergence(path) -> dict:
    """
    `{series: {x, y}}` from a convergence log, or `{}` when there is none.

    The supervised and RL logs differ in their first column and in whether they
    carry a validation loss, so both shapes are read here rather than at two
    call sites.
    """
    if not path or not Path(path).exists():
        return {}
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    header = [h.strip() for h in lines[0].split(",")]
    rows: list = []
    for line in lines[1:]:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != len(header):
            continue
        rows.append(fields)
    if not rows:
        return {}

    steps = []
    for row in rows:
        token = row[0].split()[-1]
        try:
            steps.append(float(token))
        except ValueError:
            steps.append(float(len(steps) + 1))

    out: dict = {}
    for name, label in (
        ("avg_train_loss", "training"),
        ("avg_valid_loss", "validation"),
    ):
        if name not in header:
            continue
        column = header.index(name)
        pairs = []
        for step, row in zip(steps, rows):
            try:
                pairs.append((step, float(row[column])))
            except ValueError:
                continue
        if pairs:
            out[label] = {"x": [p[0] for p in pairs], "y": [p[1] for p in pairs]}
    return out


def generate_figures(
    results_dir: str,
    metrics: dict,
    generated_smiles: list,
    training_smiles: list,
    test_smiles: list,
):
    """
    (a) the validity-uniqueness-novelty funnel with counts annotated, (b) the
    distribution of each generated molecule's maximum Tanimoto to the training
    split, and (c) the generated against test distributions of MW, logP and SA.

    The funnel is a bar chart and not three numbers because the counts are what
    make the rates readable: a novelty of 1.0 over four surviving molecules and
    over nine thousand are the same number and not the same result. Panel (b)
    is where memorization shows: a mass piled against 1.0 is a model
    reproducing its training set while scoring well on binary novelty.
    """
    stages = [
        ("sampled", metrics.get("n_sampled")),
        ("valid", metrics.get("n_valid")),
        ("unique valid", metrics.get("n_unique_valid")),
        ("unique valid novel", metrics.get("n_unique_valid_novel")),
    ]
    stages = [(name, value) for name, value in stages if value is not None]
    if not stages:
        _skip("generation funnel", "metrics.json carries none of the funnel counts")

    similarities = _max_similarities(generated_smiles, training_smiles)
    if not similarities:
        _skip(
            "training-similarity",
            "no unique valid generated molecule or no training reference",
        )

    properties = _property_values(generated_smiles, test_smiles)
    panels = [bool(stages), bool(similarities), bool(properties)]
    fig, axes = _grid(sum(panels))
    if fig is None:
        return []
    axes = iter(axes)

    if stages:
        ax = next(axes)
        ax.bar([s[0] for s in stages], [s[1] for s in stages])
        for index, (_, value) in enumerate(stages):
            ax.annotate(
                f"{int(value):,}",
                xy=(index, value),
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=DARK_SLATE,
            )
        ax.set_xlabel("funnel stage")
        ax.set_ylabel("molecules (count)")
        ax.tick_params(axis="x", labelrotation=20)

    if similarities:
        ax = next(axes)
        ax.hist(similarities, bins=30, range=(0.0, 1.0))
        ax.axvline(0.9, linestyle="--", color=DARK_SLATE)
        ax.annotate(
            "near-duplicate\nthreshold",
            xy=(0.89, 0.72),
            xycoords=("data", "axes fraction"),
            ha="right",
            fontsize=9.5,
            color=DARK_SLATE,
        )
        ax.set_xlabel("maximum ECFP4 Tanimoto to the training split (unitless)")
        ax.set_ylabel("unique valid molecules (count)")

    if properties:
        _draw_property_panel(next(axes), properties)

    return _label_and_save(fig, fig.axes, results_dir, "generation")


def _max_similarities(generated: list, reference: list) -> list:
    """Each generated molecule's maximum ECFP4 Tanimoto to the reference set."""
    from rdkit import Chem, DataStructs

    from ._core import ecfp4

    reference_fps = []
    for smi in reference:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            reference_fps.append(ecfp4(mol))
    if not reference_fps:
        return []
    out = []
    for smi in sorted(set(generated)):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        out.append(max(DataStructs.BulkTanimotoSimilarity(ecfp4(mol), reference_fps)))
    return out


PROPERTY_PANEL = [
    ("mol_weight", "MW (Da)"),
    ("logp", "logP (unitless)"),
    ("sa_score", "SA score (1-10)"),
]


def _property_values(generated: list, reference: list) -> list:
    """
    `(label, position, offset, z-scores)` for each property and each side, or
    `[]` when nothing can be computed.

    Standardized against the test split's own mean and standard deviation,
    because the three properties differ by two orders of magnitude: plotted raw
    on one axes, MW sets the scale and logP and SA collapse to flat lines. On a
    z-scale the reference distribution is centered on zero with unit width by
    construction, so the panel shows exactly the question being asked --
    whether the generated set sits on top of the reference set for all three at
    once, since a model that drifts drifts on all of them.
    """
    import numpy as np
    from rdkit import Chem

    from ._properties import PROPERTY_REGISTRY

    if not generated or not reference:
        _skip("property-distribution", "no generated sample or no test split")
        return []

    def values_of(fn, smiles):
        out = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            try:
                out.append(float(fn(mol)))
            except (ValueError, TypeError):
                continue
        return out

    rows: list = []
    for index, (name, unit) in enumerate(PROPERTY_PANEL):
        fn = PROPERTY_REGISTRY.get(name)
        if fn is None:
            continue
        reference_values = values_of(fn, reference)
        if len(reference_values) < 2:
            continue
        center = float(np.mean(reference_values))
        spread = float(np.std(reference_values)) or 1.0
        for offset, smiles in enumerate((sorted(set(generated)), reference)):
            scaled = [(v - center) / spread for v in values_of(fn, smiles)]
            if scaled:
                rows.append((unit, index * 3 + offset, offset, scaled))
    if not rows:
        _skip("property-distribution", "no property could be computed on either set")
    return rows


def _draw_property_panel(ax, rows: list) -> None:
    """The violins, with generated and reference told apart by color and fill."""
    from .plot_style import AIME_PALETTE

    parts = ax.violinplot(
        [row[3] for row in rows],
        positions=[row[1] for row in rows],
        widths=0.8,
        showmedians=True,
    )
    # Color *and* fill, since a violin carries no marker to distinguish the two
    # sides in grayscale.
    for body, row in zip(parts["bodies"], rows):
        body.set_facecolor(AIME_PALETTE[0] if row[2] == 0 else AIME_PALETTE[1])
        body.set_alpha(0.85 if row[2] == 0 else 0.35)
    ax.axhline(0.0, linestyle=":", color=DARK_SLATE, linewidth=1.0)
    ax.set_xticks([index * 3 + 0.5 for index in range(len(PROPERTY_PANEL))])
    ax.set_xticklabels([unit for _, unit in PROPERTY_PANEL], fontsize=9.5)
    ax.set_xlabel("property, generated (filled) against test split (open)")
    ax.set_ylabel("z-score against the test split")


def conditional_figures(results_dir: str, metrics: dict, conditions: list):
    """
    For a continuous condition, (a) requested against achieved with the
    identity line and (b) the decile coverage. For a categorical one, the
    per-class bars with both baselines beside each class.

    The identity line is the whole figure: a model that ignores its condition
    produces a horizontal cloud at the corpus mean, which is visible against
    the diagonal and invisible in a success rate. An empty decile is drawn as a
    gap and never as a zero, because no request fell into it and zero would
    read as total failure there.
    """

    table_path = Path(results_dir) / "condition_per_class.json"
    if table_path.exists():
        return _categorical_figure(results_dir, table_path)

    coverage_keys = sorted(k for k in metrics if k.startswith("condition_coverage_"))
    if not conditions and not coverage_keys:
        _skip(
            "conditional",
            "no requested-against-achieved pairs and no decile coverage",
        )
        return []

    if not conditions:
        _skip("requested-against-achieved", "no molecule carried both values")
    drawable = [
        k for k in coverage_keys if any(v is not None for v in metrics[k] or [])
    ]
    if coverage_keys and not drawable:
        _skip("decile-coverage", "no decile carried a value for any property")

    fig, axes = _grid(bool(conditions) + bool(drawable))
    if fig is None:
        return []
    axes = iter(axes)

    if conditions:
        ax = next(axes)
        requested = [c[1] for c in conditions]
        achieved = [c[2] for c in conditions]
        ax.plot(requested, achieved, marker="o", linestyle="none", alpha=0.6)
        low, high = min(requested + achieved), max(requested + achieved)
        ax.plot([low, high], [low, high], linestyle="--", color=DARK_SLATE)
        ax.set_xlabel("requested value (property's own units)")
        ax.set_ylabel("achieved value (property's own units)")

    if drawable:
        ax = next(axes)
        for index, key in enumerate(drawable):
            deciles = metrics.get(key) or []
            # An empty decile is a gap and never a zero: no request fell into
            # it, and zero would read as total failure there. A None in the
            # series is what matplotlib draws as a break in the line.
            xs = [i + 1 for i in range(len(deciles))]
            ys = [v if v is not None else float("nan") for v in deciles]
            ax.plot(
                xs,
                ys,
                label=key.replace("condition_coverage_", ""),
                marker=MARKERS[index % len(MARKERS)],
                linestyle=LINESTYLES[index % len(LINESTYLES)],
            )
        ax.set_xlabel("decile of the training property distribution (1-10)")
        ax.set_ylabel("success rate within tolerance (fraction)")
        ax.set_ylim(0.0, 1.0)
        ax.legend()

    return _label_and_save(fig, fig.axes, results_dir, "conditioning")


def _categorical_figure(results_dir: str, table_path: Path):
    """
    The per-class categorical metrics with both baselines beside each class.

    Three bars per class, in the order conditional, lookup, post-hoc, because
    the comparison is the finding: beating post-hoc is the claim and matching
    lookup on novelty is the failure.
    """
    import json as _json

    table = _json.loads(table_path.read_text())
    classes = list(table.get("class_panel") or [])
    blocks = [
        ("conditional", table.get("conditional") or {}),
        ("lookup baseline", table.get("baseline_lookup") or {}),
        ("post-hoc baseline", table.get("baseline_posthoc") or {}),
    ]
    if not classes or not any(rows for _, rows in blocks):
        _skip("categorical conditioning", f"{table_path} names no class with a value")
        return []

    fig, axes = _grid(2)
    width = 0.26
    for metric_index, (key, label) in enumerate(
        (
            ("class_selectivity_margin", "class selectivity margin (Tanimoto)"),
            ("novelty", "novelty (fraction)"),
        )
    ):
        ax = axes[metric_index]
        highest = 0.0
        for block_index, (name, rows) in enumerate(blocks):
            values = [(rows.get(c) or {}).get(key) for c in classes]
            offsets = [i + (block_index - 1) * width for i in range(len(classes))]
            # A class with no value is drawn as a gap and annotated, never as a
            # zero bar: nothing was measured there, and a zero reads as a
            # measurement that came out at the floor.
            ax.bar(
                offsets,
                [v if v is not None else float("nan") for v in values],
                width=width,
                label=name,
                hatch=("", "//", "..")[block_index],
            )
            for offset, value in zip(offsets, values):
                if value is None:
                    ax.annotate(
                        "n/a",
                        xy=(offset, 0.0),
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        rotation=90,
                        color=DARK_SLATE,
                    )
                else:
                    highest = max(highest, float(value))
        # Headroom for the legend, which otherwise lands on the tallest bars.
        ax.set_ylim(0.0, (highest or 1.0) * 1.35)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=9.5)
        ax.set_xlabel("requested class")
        ax.set_ylabel(label)
    axes[0].legend()
    return _label_and_save(fig, axes, results_dir, "conditioning")


def goal_directed_figures(
    results_dir: str, metrics: dict, score_log: str, optimization_log: list
):
    """
    (a) the oracle score against learning step, (b) the budget curve with the
    virtual screening baseline on the same axes, and (c)
    `top100_warhead_scaffold_count` against learning step where a bifunctional
    dataset records it.

    The screening baseline goes on the same axes and not in a table, because
    the headline comparison is whether the generative curve is above the flat
    line a random library gives at the same budget, and in separate figures it
    is not a comparison. Panel (c) is the reward-hacking detector: a score
    rising while the warhead scaffold count falls toward 1 is the failure mode,
    and only the trajectory shows it.
    """
    steps = _read_score_log(score_log)
    if not steps:
        _skip("score-trajectory", f"no readable score log at {score_log}")
    curve = _budget_curve(optimization_log)
    if not curve:
        _skip("budget-curve", "results/optimization_log.jsonl is absent or empty")
    scaffolds = _read_scaffold_trajectory(results_dir)
    if not scaffolds:
        _skip(
            "scaffold-collapse",
            "results/top100_scaffolds.jsonl is absent, which is expected off a "
            "bifunctional dataset",
        )

    fig, axes = _grid(sum(map(bool, (steps, curve, scaffolds))))
    if fig is None:
        return []
    axes = iter(axes)

    if steps:
        ax = next(axes)
        ax.plot(
            steps["x"], steps["y"], marker="o", markevery=max(1, len(steps["x"]) // 12)
        )
        ax.set_xlabel("learning step (count)")
        ax.set_ylabel("mean composite score (unitless)")

    if curve:
        ax = next(axes)
        ax.plot(
            curve["x"],
            curve["y"],
            marker="o",
            linestyle="-",
            markevery=max(1, len(curve["x"]) // 12),
            label="generative",
        )
        for index, (budget, value) in enumerate(_baseline_points(metrics)):
            ax.plot(
                [budget],
                [value],
                marker=MARKERS[(index + 1) % len(MARKERS)],
                linestyle="none",
                label="virtual screening" if index == 0 else None,
            )
        ax.set_xlabel("unique molecules submitted to the oracle (count)")
        ax.set_ylabel("top-10 mean oracle score (unitless)")
        ax.legend()

    if scaffolds:
        ax = next(axes)
        ax.plot(
            scaffolds["x"],
            scaffolds["y"],
            marker="s",
            linestyle="--",
            markevery=max(1, len(scaffolds["x"]) // 12),
        )
        ax.set_xlabel("learning step (count)")
        ax.set_ylabel("distinct warhead scaffolds in the top 100 (count)")

    return _label_and_save(fig, fig.axes, results_dir, "optimization")


def _read_score_log(path) -> dict:
    """`{x, y}` from logs/score.log, or `{}`."""
    if not path or not Path(path).exists():
        return {}
    xs, ys = [], []
    for line in Path(path).read_text().splitlines()[1:]:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            xs.append(float(fields[0].split()[-1]))
            ys.append(float(fields[1]))
        except ValueError:
            continue
    return {"x": xs, "y": ys} if xs else {}


def _budget_curve(optimization_log: list) -> dict:
    """The running top-10 mean against oracle calls, from the optimization log."""
    if not optimization_log:
        return {}
    from . import _budget

    points = _budget.top10_curve(optimization_log)
    return {"x": [p[0] for p in points], "y": [p[1] for p in points]} if points else {}


def _baseline_points(metrics: dict) -> list:
    """`(budget, top-10 mean)` for each declared virtual screening budget."""
    points = []
    for key, value in metrics.items():
        if not key.startswith("vs_baseline_top10_at_") or value is None:
            continue
        token = key.rsplit("_", 1)[-1]
        scale = {"k": 1000}.get(token[-1], 1)
        try:
            points.append((float(token.rstrip("k")) * scale, float(value)))
        except ValueError:
            continue
    return sorted(points)


def _read_scaffold_trajectory(results_dir: str) -> dict:
    """
    `{x, y}` from results/top100_scaffolds.jsonl, written per evaluation step by
    a bifunctional goal-directed run, or `{}` when there is none.
    """
    path = Path(results_dir) / "top100_scaffolds.jsonl"
    if not path.exists():
        return {}
    xs, ys = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            xs.append(float(row["step"]))
            ys.append(float(row["top100_warhead_scaffold_count"]))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return {"x": xs, "y": ys} if xs else {}

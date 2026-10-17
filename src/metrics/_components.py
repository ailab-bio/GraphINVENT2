"""
Component-level metrics for bifunctional molecules.

Whole-molecule statistics for a bifunctional compound are dominated by the
linker, which is the least informative part, so CLAUDE.md recomputes the
conditional and goal-directed metrics over the decomposed target-binding
component. That needs a decomposition, which comes from PROTAC-Splitter
(Ribes et al., J. Cheminform. 18:30, 2026;
github.com/ribesstefano/PROTAC-Splitter), declared as the `components` extra.

`split_success_rate` is always reported beside the component metrics, and the
component metrics are computed only over the molecules that split: a falling
split rate otherwise masquerades as improving component quality.
"""

from __future__ import annotations

from ._core import ECFP4_RADIUS, compute_core_metrics, ecfp4


class MissingInput(Exception):
    """A metric's input is absent. The message says which and where from."""


def decompose(smiles_list: list) -> list:
    """
    Decompose each molecule into (warhead, linker, e3_ligand), or None when the
    decomposition fails.

    One function, so the dependency has one call site. Fails loudly when the
    extra is not installed rather than returning nothing and letting every
    component metric quietly report zero.
    """
    try:
        import protac_splitter  # noqa: F401
    except ImportError:
        raise MissingInput(
            "Component-level metrics need PROTAC-Splitter, which is the "
            "'components' extra:\n"
            "    uv sync --extra components\n"
            "It decomposes a bifunctional molecule into warhead, linker and E3 "
            "ligand (Ribes et al., J. Cheminform. 18:30, 2026). Without it "
            "split_success_rate and every warhead_* metric are undefined."
        ) from None

    # The import above is the contract; the call below is the only place the
    # library's own interface is touched.
    from protac_splitter import split_protac

    out: list = []
    for smi in smiles_list:
        try:
            result = split_protac(smi)
        except Exception:
            out.append(None)
            continue
        out.append(result if result else None)
    return out


def compute_component_metrics(
    generated_smiles: list,
    n_sampled: int,
    training_warheads: list,
    known_e3_ligands: list,
    *,
    e3_similarity_floor: float,
) -> dict:
    """
    `split_success_rate`, the `warhead_*` recomputation, and
    `e3_recognizable_rate`.

    Parameters
    ----------
    training_warheads
        Target-binding components of the training molecules, the reference set
        for the warhead-level novelty and near-duplicate numbers.
    known_e3_ligands
        The reference set for `e3_recognizable_rate`.
    e3_similarity_floor
        A predicted E3 ligand counts as recognizable above this ECFP4 Tanimoto
        to any known E3 ligand. CLAUDE.md calls this a validity check, not a
        diversity analysis, so the floor is stated by the caller and recorded.
    """
    decomposed = decompose(generated_smiles)
    split_ok = [d for d in decomposed if d is not None]

    results: dict = {
        "split_success_rate": len(split_ok) / n_sampled if n_sampled else None,
        "n_split_successfully": len(split_ok),
        "e3_similarity_floor": e3_similarity_floor,
        "ecfp4_radius": ECFP4_RADIUS,
    }
    if not split_ok:
        results["e3_recognizable_rate"] = None
        return results

    # The core table, recomputed over target-binding components only. The
    # whole-molecule core metrics stay whole-molecule and are reported
    # separately; these carry the warhead_ prefix CLAUDE.md tabulates.
    warheads = [d[0] for d in split_ok]
    warhead_core = compute_core_metrics(
        warheads, n_sampled=len(warheads), training_smiles=training_warheads
    )
    for key, value in warhead_core.items():
        results[f"warhead_{key}"] = value

    # e3_recognizable_rate, over successfully split molecules.
    from rdkit import Chem, DataStructs

    known_fps = []
    for smi in known_e3_ligands:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            known_fps.append(ecfp4(mol))
    if not known_fps:
        raise MissingInput(
            "e3_recognizable_rate needs a set of known E3 ligands to compare "
            "against, and got none."
        )

    recognizable = 0
    compared = 0
    for d in split_ok:
        e3 = d[2] if len(d) > 2 else None
        mol = Chem.MolFromSmiles(e3) if e3 else None
        if mol is None:
            continue
        compared += 1
        sims = DataStructs.BulkTanimotoSimilarity(ecfp4(mol), known_fps)
        if sims and max(sims) > e3_similarity_floor:
            recognizable += 1
    results["e3_recognizable_rate"] = recognizable / compared if compared else None
    return results

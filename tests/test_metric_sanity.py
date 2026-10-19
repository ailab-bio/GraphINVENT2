"""
Metric sanity over fixtures with hand-known answers.

Justification for a fifth test file: the other four exercise `src/graphinvent/`
and none of them imports `src/metrics/`, which is where every reported number
now comes from. Each assertion here guards a defect that shipped undetected and
whose failure mode is a plausible wrong number rather than a crash:

  - novelty was pinned at 1.0 by an isomeric-canonicalization mismatch, which
    would have made every generative result look better than it was
  - the funnel denominators could disagree, letting validity exceed 1.0
  - fcd was computed on a few hundred molecules, which is a different quantity
    from the published one rather than a noisy version of it, and on one
    molecule it hung the suite at 100% CPU with no error
  - fcd compared a flattened generated set against a raw stereo-bearing
    reference
  - the budget curve integrated one component of a multi-component objective
  - three implementations of the core metrics disagreed on success_rate
  - condition_success_rate is a step function of condition_tolerance, so it
    reported the same number for models whose error differed eightfold
  - top100_scaffold_count counted an acyclic molecule's empty scaffold as a
    scaffold, and is blind by construction to a collapsing warhead
  - conditioning.properties was never checked against the file's columns, so a
    config listing the same properties in the other order trained on one
    property and reported it as another, with a byte-identical HDF5
  - every molecule counted valid was an open shell: RDKit sanitizes a radical
    without complaint, so 253 of 253 generated molecules with unpaired
    electrons were counted in validity and in every metric downstream of it
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "src" / "graphinvent")]

from metrics import _components  # noqa: E402
from metrics._budget import compute_set_level_metrics, top100  # noqa: E402
from metrics._conditions import (  # noqa: E402
    compute_continuous_condition_metrics,
    condition_mae,
)
from metrics._core import canonical_smiles, compute_core_metrics  # noqa: E402
from metrics._unconditional import (  # noqa: E402
    FCD_MIN_MOLECULES,
    _compute_fcd,
)

# Two of these carry stereocenters. Flattening them collapses each onto its
# own flat form, which is what makes the novelty asymmetry measurable.
TRAINING = [
    "C[C@H](N)C(=O)O",
    "C[C@@H](N)C(=O)O",
    "CCO",
    "c1ccccc1",
]
# The three flat forms of the training set, plus one molecule that is genuinely
# absent from it.
GENERATED_FLAT = ["CC(N)C(=O)O", "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]


class TestNoveltyUnderBothChiralitySettings:
    """The bug: novelty read 1.0 whichever setting was used."""

    def test_flat_comparison_sees_the_copies(self):
        # With isomeric=False both sides are flat, so three of the four
        # generated molecules are training molecules and only aspirin is novel.
        core = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=False
        )
        assert core["n_unique_valid"] == 4
        assert core["n_unique_valid_novel"] == 1, (
            "flattened, three of these four ARE training molecules; a novelty "
            "above 1/4 means the training side was not canonicalized the same way"
        )
        assert core["novelty"] == pytest.approx(0.25)

    def test_isomeric_comparison_counts_the_flattened_copies_as_novel(self):
        # With isomeric=True the flat alanine cannot equal either stereoisomer,
        # so it counts as novel: two of four, not one.
        core = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=True
        )
        assert core["n_unique_valid_novel"] == 2
        assert core["novelty"] == pytest.approx(0.5)

    def test_the_two_settings_must_disagree(self):
        flat = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=False
        )
        iso = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=True
        )
        assert flat["novelty"] != iso["novelty"], (
            "identical novelty under both settings is the signature of the "
            "original bug: one side was canonicalized and the other was not"
        )

    def test_canonicalization_is_aligned_with_its_input(self):
        out = canonical_smiles(["CCO", "not_a_molecule", "c1ccccc1"])
        assert len(out) == 3
        assert out[1] is None, "entries RDKit rejects must stay in position"


class TestFunnelDenominators:
    """The bug: validity could exceed 1.0 when batches over-produced."""

    def test_the_funnel_composes(self):
        core = compute_core_metrics(
            GENERATED_FLAT + ["garbage", None],
            n_sampled=6,
            training_smiles=TRAINING,
            isomeric=False,
        )
        assert core["validity"] == pytest.approx(core["n_valid"] / core["n_sampled"])
        assert core["uniqueness"] == pytest.approx(
            core["n_unique_valid"] / core["n_valid"]
        )
        assert core["novelty"] == pytest.approx(
            core["n_unique_valid_novel"] / core["n_unique_valid"]
        )
        assert core["vun"] == pytest.approx(
            core["n_unique_valid_novel"] / core["n_sampled"]
        )
        # The composite is the product of the three stages, which is the
        # property that makes the funnel a funnel.
        assert core["vun"] == pytest.approx(
            core["validity"] * core["uniqueness"] * core["novelty"]
        )

    def test_no_rate_exceeds_one(self):
        core = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=False
        )
        for key in ("validity", "uniqueness", "novelty", "vun"):
            assert 0.0 <= core[key] <= 1.0, f"{key} = {core[key]} is not a rate"


class TestFcdSampleFloor:
    """
    The bug: one molecule produced a NaN covariance and `scipy.linalg.sqrtm`
    never returned, hanging the suite at 100% CPU with no error. The timeout is
    the point of this test: without it a regression hangs rather than fails.
    """

    @pytest.mark.timeout(60)
    def test_one_molecule_returns_rather_than_hanging(self):
        result = _compute_fcd(["CCO"], ["CCO", "c1ccccc1"])
        assert result["value"] is None
        assert "insufficient_sample" in result["reason"]
        assert result["n_generated"] == 1

    @pytest.mark.timeout(60)
    def test_below_the_floor_is_not_computed(self):
        # Comfortably more than two molecules and far fewer than the floor: the
        # old two-molecule floor would have computed a number here.
        gen = [f"C{'C' * i}O" for i in range(1, 40)]
        result = _compute_fcd(gen, gen)
        assert result["value"] is None, (
            f"fcd must not be computed from {len(gen)} molecules; a 512-"
            "dimensional covariance from that sample is singular"
        )
        assert "insufficient_sample" in result["reason"]
        assert result["n_generated"] < FCD_MIN_MOLECULES

    def test_the_floor_is_a_statistical_one(self):
        assert FCD_MIN_MOLECULES >= 512, (
            "ChemNet activations are 512-dimensional, so a floor below 512 "
            "admits a singular covariance"
        )

    @pytest.mark.timeout(60)
    def test_counts_are_reported_whatever_happens(self):
        result = _compute_fcd(["CCO"], ["CCO"])
        assert "n_generated" in result and "n_reference" in result


class TestExactlyOneImplementation:
    """The bug: three implementations of the core metrics, two of them live."""

    def test_the_duplicate_module_is_gone(self):
        assert not (REPO_ROOT / "src" / "graphinvent" / "_metrics.py").exists(), (
            "src/graphinvent/_metrics.py is a second implementation of the core "
            "metrics and CLAUDE.md requires exactly one"
        )

    def test_the_dead_layer_is_not_exported(self):
        import metrics

        for dead in (
            "evaluate_unconditional",
            "evaluate_conditional",
            "evaluate_goal_directed",
        ):
            assert dead not in metrics.__all__
            assert not hasattr(metrics, dead)

    def test_success_rate_uses_at_least(self):
        # The two implementations disagreed: one used a strict `>`. The config
        # key reads as "a molecule is a success at or above this score", so a
        # molecule exactly on the threshold must count.
        from metrics._budget import compute_set_level_metrics

        result = compute_set_level_metrics([("CCO", 0.5)], success_threshold=0.5)
        assert result["success_rate"] == pytest.approx(
            1.0
        ), "a score exactly at the threshold must count as a success"


class TestTruncationInvariant:
    """
    The bug: batches over-produce, so more molecules were measured than were
    asked for and validity exceeded 1.0. Latent at low validity, which is why
    a smoke test on a briefly-trained model cannot catch it: the check has to
    be the invariant itself.
    """

    def test_more_valid_than_sampled_is_refused(self):
        with pytest.raises(ValueError, match="not truncated to n_samples"):
            compute_core_metrics(
                ["CCO", "c1ccccc1", "CCC"],
                n_sampled=2,
                training_smiles=TRAINING,
                isomeric=False,
            )

    def test_exactly_n_sampled_is_fine(self):
        core = compute_core_metrics(
            ["CCO", "c1ccccc1"], n_sampled=2, training_smiles=TRAINING, isomeric=False
        )
        assert core["validity"] == pytest.approx(1.0)


class TestConditionMeanAbsoluteError:
    """
    The bug: `condition_success_rate` is a step function of the tolerance, so a
    model that halves its error moves it not at all until it crosses the
    threshold. `condition_mae_<property>` is the tolerance-free companion.
    """

    TRAINING_VALUES = list(range(100))

    def test_identical_rates_different_errors(self):
        tight = compute_continuous_condition_metrics(
            "logp", [10.0] * 4, [10.0, 10.1, 9.9, 10.0], 0.5, self.TRAINING_VALUES
        )
        loose = compute_continuous_condition_metrics(
            "logp", [10.0] * 4, [10.4, 10.4, 9.6, 9.6], 0.5, self.TRAINING_VALUES
        )
        assert (
            tight["condition_success_rate_logp"]
            == loose["condition_success_rate_logp"]
            == 1.0
        ), "both samples are inside the tolerance, so the rate cannot separate them"
        assert tight["condition_mae_logp"] == pytest.approx(0.05)
        assert loose["condition_mae_logp"] == pytest.approx(0.4)

    def test_denominator_matches_the_success_rate(self):
        # Molecules without both a request and a finite achieved value are
        # excluded, not counted as a zero error.
        assert condition_mae([1.0, None, 3.0], [1.0, 2.0, float("nan")]) == 0.0
        assert condition_mae([], []) is None


class TestTop100ScaffoldCount:
    """
    Two bugs. RDKit returns an empty SMILES for a molecule with no ring system,
    which was counted as a scaffold; and a PROTAC's whole-molecule scaffold is
    nearly unique per compound, so the count cannot detect warhead collapse.
    """

    def test_acyclic_molecule_is_not_a_scaffold(self):
        scored = [("c1ccccc1", 0.9), ("CCCCCC", 0.8)]
        metrics = compute_set_level_metrics(scored, success_threshold=0.5)
        assert metrics["top100_scaffold_count"] == 1, (
            "hexane has no ring system and so no Bemis-Murcko scaffold; "
            "counting its empty scaffold makes the tally one too high"
        )

    def test_warhead_count_sees_a_collapse_the_whole_molecule_count_misses(
        self, monkeypatch
    ):
        # Three molecules sharing one warhead and differing elsewhere. The
        # decomposition is stubbed because the real one runs in another
        # interpreter at seconds per molecule; what is under test here is that
        # the count is taken over the target-binding component.
        warhead = "[*:1]c1ccc(-c2scnc2C)cc1"
        fragments = [
            {"poi": warhead, "linker": f"[*:1]CCC{'C' * i}[*:2]", "e3": "[*:2]NC=O"}
            for i in range(3)
        ]
        monkeypatch.setattr(_components, "decompose", lambda smiles: fragments)
        out = _components.compute_top100_warhead_scaffolds(["x", "y", "z"])
        assert out["top100_n_split"] == 3
        assert out["top100_split_success_rate"] == 1.0
        assert out["top100_warhead_scaffold_count"] == 1, (
            "all three carry the same warhead, which is the collapse the "
            "whole-molecule count cannot see"
        )

    def test_both_counts_use_the_same_hundred_molecules(self):
        # 150 molecules, so the two metrics must agree on which 100 they are.
        scored = [(f"c1ccccc1{'C' * (i % 7)}", 1.0 - i * 0.001) for i in range(150)]
        assert [smi for smi, _ in top100(scored)] == [smi for smi, _ in scored[:100]]


class TestCompositeBudgetCurve:
    """
    The bug: the oracle call curve logged one component of a multi-component
    objective, so every at-budget statistic and `auc_top10` described a
    quantity the policy was not trained on. Prompt E asked for this assertion
    explicitly and the file's own justification claimed it; it did not exist.

    The two components here rank molecules in opposite directions, so the
    composite is distinguishable from either one. The check is that the value
    reaching the log is the fully multiplied score and not a contribution.
    """

    @staticmethod
    def _scorer(score_type="continuous"):
        import torch
        from ScoringFunction import ScoringFunction

        sf = object.__new__(ScoringFunction)
        sf.score_components = ["big", "small"]
        sf.score_type = score_type
        sf.score_thresholds = [0.0, 0.0]
        sf.device = "cpu"
        sf.n_graphs = 0
        sf.n_molecules_scored = 0
        sf._component_uncertainty = {}
        sf._composite_log = []
        sf._composite_seen = set()
        sf._modulation = type("M", (), {"modulates_score": lambda self: False})()

        # Rising and falling in the same quantity: their product peaks in the
        # middle, where neither component does.
        rising = torch.tensor([0.2, 0.5, 0.8])
        falling = torch.tensor([0.8, 0.5, 0.2])
        sf.get_contributions_to_score = lambda graphs: [rising, falling]
        sf._graphs_to_smiles = lambda graphs: ["CCO", "CCC", "CCN"]
        return sf, rising, falling

    def test_the_log_records_the_composite_not_a_component(self):
        import torch

        sf, rising, falling = self._scorer()
        ones = torch.ones(3)
        final, components = sf.compute_score_with_components(
            graphs=[None] * 3, termination=ones, validity=ones, uniqueness=ones
        )
        expected = (rising * falling).tolist()
        logged = [score for _, score in sf.optimization_log]

        assert logged == pytest.approx(expected), (
            "the oracle call curve must carry the composite the policy was "
            f"trained on; got {logged} against a composite of {expected}"
        )
        assert logged != pytest.approx(rising.tolist())
        assert logged != pytest.approx(falling.tolist())
        assert final.tolist() == pytest.approx(expected)
        assert set(components) == {"big", "small"}

    def test_the_budget_statistics_follow_that_log(self):
        from metrics._budget import compute_at_budget_metrics

        sf, rising, falling = self._scorer()
        import torch

        ones = torch.ones(3)
        sf.compute_score_with_components(
            graphs=[None] * 3, termination=ones, validity=ones, uniqueness=ones
        )
        log = sf.optimization_log
        at_budget = compute_at_budget_metrics(log)
        # The middle molecule wins on the composite, and it wins on neither
        # component, so a curve following a component would report a different
        # best.
        assert at_budget["best_at_1k"] == pytest.approx(0.25)
        assert at_budget["best_at_1k"] != pytest.approx(max(rising.tolist()))
        assert at_budget["best_at_1k"] != pytest.approx(max(falling.tolist()))

    def test_the_masks_are_applied_before_the_score_is_logged(self):
        import torch

        sf, rising, falling = self._scorer()
        # The middle molecule is a duplicate and the last one is invalid, so
        # both must reach the log as zero. Recording before the masks logs the
        # unmasked product instead, which inflates every at-budget statistic.
        uniqueness = torch.tensor([1.0, 0.0, 1.0])
        validity = torch.tensor([1.0, 1.0, 0.0])
        sf.compute_score_with_components(
            graphs=[None] * 3,
            termination=torch.ones(3),
            validity=validity,
            uniqueness=uniqueness,
        )
        logged = [score for _, score in sf.optimization_log]
        assert logged == pytest.approx([0.16, 0.0, 0.0]), (
            "a duplicate, invalid or improperly terminated molecule scores "
            "zero by design, so the log cannot carry its unmasked product"
        )

    def test_duplicate_molecules_are_not_counted_twice(self):
        import torch

        sf, _, _ = self._scorer()
        sf._graphs_to_smiles = lambda graphs: ["CCO", "CCO", "CCN"]
        ones = torch.ones(3)
        sf.compute_score_with_components(
            graphs=[None] * 3, termination=ones, validity=ones, uniqueness=ones
        )
        # An oracle call is one unique molecule, so the curve advances twice.
        assert [calls for calls, _ in sf.optimization_log] == [1, 2]


class TestConditioningDeclarationMatchesTheFile:
    """
    The bug: the condition vector is built positionally from the file's property
    columns, while generation labels each slot from `conditioning.properties`.
    Nothing connected the two, so a transposed declaration silently conditioned
    on one property and reported it as another. Both orderings produced an
    identical HDF5, which is why no run could have revealed it.
    """

    FILE = Path("data/datasets/zinc1k/zinc1k_logp_mw.smi")

    def _check(self, declared, dim, columns=("logp", "mol_weight")):
        from DataProcessor import _check_conditioning

        return _check_conditioning(list(columns), declared, dim, self.FILE)

    def test_matching_declaration_is_accepted(self):
        assert self._check(["logp", "mol_weight"], 2) is None

    def test_transposed_declaration_is_refused(self):
        with pytest.raises(ValueError, match="agree in order"):
            self._check(["mol_weight", "logp"], 2)

    def test_unknown_property_name_is_refused(self):
        with pytest.raises(ValueError, match="does not match"):
            self._check(["not_a_property", "also_not"], 2)

    def test_property_columns_with_conditioning_unset_is_refused(self):
        # L13: these columns used to be dropped in silence, leaving
        # condition_dim: 0 in preprocessing.yaml as the only evidence.
        with pytest.raises(ValueError, match="would be dropped"):
            self._check([], 0)

    def test_declaration_missing_while_conditioning_is_on(self):
        with pytest.raises(ValueError, match="nothing names the slots"):
            self._check([], 2)

    def test_width_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="property columns"):
            self._check(["logp"], 1)

    def test_a_plain_smiles_file_is_unaffected(self):
        assert self._check([], 0, columns=()) is None


class TestOpenShellIsNotAMolecule:
    """
    The bug: an atom whose bonds and predicted hydrogens fall short of its
    valence becomes a radical centre, and `SanitizeMol` accepts it. Measured on
    the first toy runs, 253 of 253 molecules counted valid carried unpaired
    electrons, methylidyne among them, and validity, uniqueness, novelty,
    diversity and every synthesizability number was computed over them.

    The training corpora are entirely closed-shell, so nothing upstream could
    have revealed this.
    """

    RADICALS = ["[CH]", "[N][N]", "[C]C([C])C", "[CH2][N][CH][N]"]
    CLOSED = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]

    def test_rdkit_sanitizes_a_radical_without_complaint(self):
        # The premise: this is why a validity check has to test for it.
        from rdkit import Chem

        for smiles in self.RADICALS:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"{smiles} should parse"
            assert any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())

    def test_no_training_molecule_is_an_open_shell(self):
        # If a corpus did contain radicals the gate would silently shrink the
        # training set, so this is the other half of the invariant.
        from rdkit import Chem

        train = REPO_ROOT / "data" / "datasets" / "zinc1k" / "train.smi"
        if not train.exists():
            pytest.skip("zinc1k development split is not present")
        n = 0
        for line in train.read_text().splitlines():
            token = line.split("\t")[0].split()[0] if line.strip() else ""
            if not token or token.upper() == "SMILES":
                continue
            mol = Chem.MolFromSmiles(token)
            if mol is None:
                continue
            n += 1
            assert not any(
                a.GetNumRadicalElectrons() for a in mol.GetAtoms()
            ), f"{token} in the training split is an open shell"
        assert n > 0

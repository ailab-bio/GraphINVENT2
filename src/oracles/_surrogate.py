"""
Oracles backed by a user-supplied model or function.

These are the intended route for target-specific objectives.  Rather than
depending on a curated third-party oracle collection, a user trains a surrogate
on their own assay data, saves it, and names it in the job configuration; the
provenance and quality of the model are then theirs to control and to report.

Two mechanisms cover almost everything:

``SklearnOracle``
    A pickled scikit-learn estimator over molecular fingerprints.  This covers
    the common case of a classifier or regressor trained on ChEMBL or in-house
    activity data.

``PythonOracle``
    Any importable callable taking a list of SMILES and returning a list of
    floats.  This is the escape hatch for a PyTorch model, a web service, a
    physics calculation, or anything else, with no requirement that it fit the
    scikit-learn interface.
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ._base import BaseOracle


def _featurize(
    smiles: Sequence[Optional[str]],
    radius: int,
    n_bits: int,
    use_counts: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    """
    Morgan-fingerprint feature matrix for the parseable inputs.

    Returns the matrix together with the indices it corresponds to, so the
    caller can place predictions back into the right positions without
    featurising unparseable molecules or silently shifting the output.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    rows: List[np.ndarray] = []
    valid_idx: List[int] = []
    for i, smi in enumerate(smiles):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if use_counts:
            fp = generator.GetCountFingerprintAsNumPy(mol)
        else:
            fp = generator.GetFingerprintAsNumPy(mol)
        rows.append(np.asarray(fp, dtype=np.float64))
        valid_idx.append(i)

    if not rows:
        return np.zeros((0, n_bits), dtype=np.float64), []
    return np.vstack(rows), valid_idx


class SklearnOracle(BaseOracle):
    """
    Wraps a pickled scikit-learn estimator over molecular fingerprints.

    The pickle may hold either the estimator itself or a dict containing it;
    for a dict, ``model_key`` names the entry (``"classifier_sv"`` by default,
    matching the format used by the surrogates shipped under ``data/surrogates``).

    ``output`` selects what counts as the native value:

    ``"proba"``
        ``predict_proba(X)[:, 1]``, the positive-class probability.  Use for a
        binary activity classifier.
    ``"predict"``
        ``predict(X)``, for a regressor reporting pIC50, logS, or similar.  Pair
        it with a transform, since the raw value is not a desirability.
    ``"decision"``
        ``decision_function(X)``, for an SVM without calibrated probabilities.

    Uncertainty is available when the estimator is an ensemble exposing
    ``estimators_`` (random forest, extra trees, bagging): the spread of the
    per-estimator predictions is a cheap, well-understood proxy for how much
    the training data constrains a prediction.  It is not a calibrated
    posterior, and out-of-domain molecules can still receive a confidently
    wrong consensus, so treat it as a relative signal rather than an absolute one.

    Parameters
    ----------
    path
        Path to the pickle.
    radius, n_bits
        Morgan fingerprint parameters.  These must match whatever the model was
        trained on; there is no way to verify that from the pickle, so a
        mismatch shows up as uniformly poor predictions rather than an error.
    invalid_value
        Native value assigned to molecules RDKit cannot parse.  Defaults to 0.0,
        which for a probability means "inactive".  Set it explicitly for a
        regressor whose scale makes 0.0 a good score.
    """

    def __init__(
        self,
        name: str,
        path: str,
        transform: Optional[dict] = None,
        direction: str = "maximize",
        radius: int = 2,
        n_bits: int = 2048,
        use_counts: bool = False,
        output: str = "proba",
        model_key: str = "classifier_sv",
        invalid_value: float = 0.0,
    ) -> None:
        super().__init__(name=name, transform=transform, direction=direction)
        self.radius = radius
        self.n_bits = n_bits
        self.use_counts = use_counts
        self.output = output
        self.invalid_value = invalid_value

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Surrogate model for oracle '{name}' not found at '{path}'."
            )
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        if isinstance(loaded, dict):
            if model_key not in loaded:
                raise KeyError(
                    f"Pickle at '{path}' is a dict without key '{model_key}'; "
                    f"it contains {sorted(loaded)}. Set 'model_key' to the "
                    "entry holding the estimator."
                )
            self.model = loaded[model_key]
        else:
            self.model = loaded

        if output not in ("proba", "predict", "decision"):
            raise ValueError(
                f"Unknown output '{output}' for oracle '{name}'. "
                "Choose from: proba, predict, decision."
            )

    def _raw_predict(self, features: np.ndarray) -> np.ndarray:
        if self.output == "proba":
            return np.asarray(self.model.predict_proba(features))[:, 1]
        if self.output == "decision":
            return np.asarray(self.model.decision_function(features)).ravel()
        return np.asarray(self.model.predict(features)).ravel()

    def predict(self, smiles: Sequence[Optional[str]]) -> List[float]:
        values = [self.invalid_value] * len(smiles)
        features, valid_idx = _featurize(
            smiles, self.radius, self.n_bits, self.use_counts
        )
        if len(valid_idx) == 0:
            return values
        predictions = self._raw_predict(features)
        for position, prediction in zip(valid_idx, predictions):
            values[position] = float(prediction)
        return values

    @property
    def _sub_estimators(self) -> list:
        """Component estimators of an ensemble, or an empty list."""
        return list(getattr(self.model, "estimators_", []) or [])

    @property
    def supports_uncertainty(self) -> bool:
        """
        True only for an ensemble estimator.

        The base class infers support from whether the method is overridden,
        which is too coarse here: this class always overrides it but can only
        deliver a spread when the wrapped model has component estimators.
        Advertising support it cannot honour would make the caller take the
        uncertainty path and hit NotImplementedError mid-run.
        """
        return bool(self._sub_estimators)

    def predict_with_uncertainty(
        self, smiles: Sequence[Optional[str]]
    ) -> Tuple[List[float], List[float]]:
        """
        Native values plus the standard deviation across ensemble members.

        Unparseable molecules receive zero uncertainty alongside
        ``invalid_value``: their score is already the worst available, and
        inflating its uncertainty would let reward modulation partially undo
        the penalty.
        """
        members = self._sub_estimators
        if not members:
            raise NotImplementedError(
                f"Oracle '{self.name}' wraps a non-ensemble estimator "
                f"({type(self.model).__name__}) and cannot report uncertainty."
            )

        values = [self.invalid_value] * len(smiles)
        uncertainties = [0.0] * len(smiles)
        features, valid_idx = _featurize(
            smiles, self.radius, self.n_bits, self.use_counts
        )
        if len(valid_idx) == 0:
            return values, uncertainties

        per_member = []
        for member in members:
            if self.output == "proba" and hasattr(member, "predict_proba"):
                per_member.append(np.asarray(member.predict_proba(features))[:, 1])
            else:
                per_member.append(np.asarray(member.predict(features)).ravel())
        stacked = np.vstack(per_member)

        means = stacked.mean(axis=0)
        stds = stacked.std(axis=0)
        for position, mean, std in zip(valid_idx, means, stds):
            values[position] = float(mean)
            uncertainties[position] = float(std)
        return values, uncertainties


class PythonOracle(BaseOracle):
    """
    Wraps any importable callable ``f(list[str]) -> list[float]``.

    The escape hatch for scoring logic that does not fit a pickled estimator:
    a PyTorch model, an in-house web service, a free-energy calculation.  The
    callable is named as ``"package.module:function"`` and imported at
    construction time, so a bad path fails when the job starts rather than
    after the first batch has been generated.

    The callable is responsible for handling ``None`` and unparseable SMILES;
    it must return exactly one float per input, in order.  That contract is
    checked on every call, because a length mismatch would otherwise
    misalign scores with molecules and corrupt the reward silently.
    """

    def __init__(
        self,
        name: str,
        target: str,
        transform: Optional[dict] = None,
        direction: str = "maximize",
        kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(name=name, transform=transform, direction=direction)
        self.target = target
        self.kwargs = kwargs or {}
        self.fn: Callable = self._import_target(target)

    @staticmethod
    def _import_target(target: str) -> Callable:
        if ":" not in target:
            raise ValueError(
                f"Oracle target '{target}' must be 'module.path:callable_name'."
            )
        module_name, attr = target.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Could not import module '{module_name}' for oracle target "
                f"'{target}'. Ensure it is on PYTHONPATH."
            ) from exc
        try:
            fn = getattr(module, attr)
        except AttributeError as exc:
            raise AttributeError(
                f"Module '{module_name}' has no attribute '{attr}'."
            ) from exc
        if not callable(fn):
            raise TypeError(f"Oracle target '{target}' is not callable.")
        return fn

    def predict(self, smiles: Sequence[Optional[str]]) -> List[float]:
        values = self.fn(list(smiles), **self.kwargs)
        if len(values) != len(smiles):
            raise ValueError(
                f"Oracle '{self.name}' returned {len(values)} values for "
                f"{len(smiles)} molecules; it must return exactly one value "
                "per input, in order."
            )
        return [float(v) for v in values]

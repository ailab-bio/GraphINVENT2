"""Builds oracle instances from the ``oracles`` block of a job configuration."""

from __future__ import annotations

from typing import Dict, Type

from ._base import BaseOracle
from ._cache import CachedOracle
from ._surrogate import PythonOracle, SklearnOracle
from ._vina import VinaOracle

#: Maps the ``type`` field of an oracle spec to its class.  Register a new
#: oracle class here to make it available from configuration; nothing else in
#: the codebase needs to change.
ORACLE_TYPES: Dict[str, Type[BaseOracle]] = {
    "sklearn": SklearnOracle,
    "python": PythonOracle,
    "vina": VinaOracle,
}


class OracleFactory:
    """
    Creates oracles from configuration.

    An oracle spec is a dict whose ``type`` selects the class and whose
    remaining keys are passed to that class's constructor, so the configuration
    schema is the constructor signature and does not have to be maintained
    twice.  ``transform`` and ``direction`` are understood by every oracle.

    Examples
    --------
    A selectivity objective -- bind one kinase, avoid a second -- is two
    oracles over the same kind of quantity with opposite directions:

    >>> specs = {
    ...     "GSK3B": {"type": "sklearn", "path": "data/surrogates/gsk3b.pkl"},
    ...     "JNK3":  {"type": "sklearn", "path": "data/surrogates/jnk3.pkl",
    ...               "direction": "minimize"},
    ... }
    >>> oracles = OracleFactory.from_specs(specs)      # doctest: +SKIP
    """

    @staticmethod
    def create(name: str, spec: dict) -> BaseOracle:
        """
        Build a single oracle from its spec.

        Raises
        ------
        ValueError
            If ``type`` is missing or unknown.  Failing here means a typo in a
            config surfaces before any molecules are generated, rather than as
            a "score component is not defined" error part-way into a run.
        TypeError
            If the spec carries keys the oracle class does not accept; the
            message names them, since a silently ignored key would mean the run
            optimises something other than what was asked for.
        """
        if not isinstance(spec, dict):
            raise ValueError(
                f"Oracle '{name}' must be defined by a dict, got {type(spec).__name__}."
            )
        spec = dict(spec)
        # Underscore-prefixed keys are inline comments in params.json.
        spec = {k: v for k, v in spec.items() if not k.startswith("_")}

        oracle_type = spec.pop("type", None)
        if oracle_type is None:
            raise ValueError(
                f"Oracle '{name}' has no 'type'. "
                f"Choose from: {sorted(ORACLE_TYPES)}."
            )
        if oracle_type not in ORACLE_TYPES:
            raise ValueError(
                f"Unknown oracle type '{oracle_type}' for oracle '{name}'. "
                f"Choose from: {sorted(ORACLE_TYPES)}."
            )

        cls = ORACLE_TYPES[oracle_type]
        try:
            return cls(name=name, **spec)
        except TypeError as exc:
            raise TypeError(
                f"Could not construct oracle '{name}' of type '{oracle_type}': "
                f"{exc}. Check the keys in its config block."
            ) from exc

    @staticmethod
    def create_cached(name: str, spec: dict) -> CachedOracle:
        """
        Build an oracle wrapped in the deduplicating cache.

        Caching matters more here than convenience: a converging RL agent
        re-proposes the same molecules repeatedly, and for a docking oracle
        each repeat would otherwise cost another pose search.  The cache is
        also what makes the oracle-call budget meaningful, since it counts
        unique molecules evaluated.
        """
        return CachedOracle(OracleFactory.create(name, spec))

    @staticmethod
    def from_specs(specs: dict) -> Dict[str, CachedOracle]:
        """Build every oracle in an ``{name: spec}`` mapping."""
        if not specs:
            return {}
        return {
            name: OracleFactory.create_cached(name, spec)
            for name, spec in specs.items()
            if not name.startswith("_")
        }

    @staticmethod
    def from_config(config: dict) -> Dict[str, CachedOracle]:
        """
        Build oracles from a full params.json dict, reading ``job.oracles``.
        """
        job = config.get("job", config)
        return OracleFactory.from_specs(job.get("oracles", {}))

    @staticmethod
    def known_types() -> list:
        """Registered oracle type names."""
        return sorted(ORACLE_TYPES)

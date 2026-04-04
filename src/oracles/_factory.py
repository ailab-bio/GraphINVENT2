"""OracleFactory: create oracle instances by name from a config dict."""

from __future__ import annotations

from ._cache import CachedOracle
from ._tdc import ORACLE_REGISTRY, TDCOracle


class OracleFactory:
    """
    Creates oracle instances by name.

    Oracles are created as :class:`CachedOracle` wrappers around
    :class:`TDCOracle` by default.  For the 5 named PMO oracles (SA, DRD2,
    GSK3B, JNK3, celecoxib_rediscovery) the TDC name is looked up from
    :data:`~src.oracles._tdc.ORACLE_REGISTRY`; any other name is forwarded
    to TDC verbatim, making the entire TDC catalogue available without code
    changes.

    Examples
    --------
    Single oracle:

    >>> oracle = OracleFactory.create_cached("DRD2")
    >>> scores = oracle(["CCO", "c1ccccc1"])

    From a params.json config:

    >>> import json
    >>> config = json.load(open("jobs/rl/params.json"))
    >>> oracles = OracleFactory.from_config(config)
    >>> for name, oracle in oracles.items():
    ...     print(name, oracle(["CCO"]))
    """

    @staticmethod
    def create(name: str) -> TDCOracle:
        """
        Create a raw (uncached) TDC oracle by name.

        Parameters
        ----------
        name : str
            Oracle name (see :data:`~src.oracles._tdc.ORACLE_REGISTRY` for
            the 5 built-in names; any TDC oracle name also works).

        Returns
        -------
        TDCOracle
        """
        return TDCOracle(name=name)

    @staticmethod
    def create_cached(name: str) -> CachedOracle:
        """
        Create a cached TDC oracle by name.

        The returned :class:`CachedOracle` deduplicates queries and tracks
        cumulative oracle calls for AUC Top-k computation.

        Parameters
        ----------
        name : str
            Oracle name.

        Returns
        -------
        CachedOracle
        """
        return CachedOracle(TDCOracle(name=name))

    @staticmethod
    def known_names() -> list:
        """
        Return the list of built-in oracle names (keys in ORACLE_REGISTRY).

        Any name not in this list is still valid — it is forwarded to TDC
        directly — but these are the five names with first-class support.
        """
        return list(ORACLE_REGISTRY.keys())

    @staticmethod
    def from_config(config: dict) -> dict:
        """
        Instantiate all oracles specified in a job config dict.

        Reads ``config["job"]["oracle"]`` (str, single oracle) or
        ``config["job"]["oracles"]`` (list of str, multi-objective).  Returns
        an empty dict if neither key is present.

        Parameters
        ----------
        config : dict
            Top-level config as loaded from params.json.

        Returns
        -------
        dict mapping oracle name -> CachedOracle
        """
        job = config.get("job", {})
        oracles: dict = {}

        if "oracle" in job:
            name = job["oracle"]
            oracles[name] = OracleFactory.create_cached(name)
        elif "oracles" in job:
            for name in job["oracles"]:
                oracles[name] = OracleFactory.create_cached(name)

        return oracles

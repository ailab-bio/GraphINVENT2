# Contributing to GraphINVENT2

Contributions are welcome as issues or pull requests. Bug reports are most useful when they
include the `params.json` of the failing job and the `run_info` block from `params_all.json`,
which records the library versions, device, and git commit the run used.

## Code style

Formatting is black at line length 88, and linting is ruff with pyflakes, pycodestyle, and
isort rules enabled. Both read their configuration from `pyproject.toml`, so running them
without arguments produces the same result as CI. mypy is configured but not enforced, since
tensor-heavy code produces more noise than signal under strict settings.

All importable code lives under `src/`; `submit.py`, `visualize.py`, and `cleanup.py` are at
the repository root and are not part of any package.

```bash
python -m black src/ tests/ submit.py visualize.py cleanup.py
python -m ruff check src/ tests/ submit.py visualize.py cleanup.py
python -m mypy src/graphinvent --ignore-missing-imports
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

black, ruff with `--fix`, and the standard whitespace, YAML, JSON, and debug-statement hooks
then run on every commit. To run them across the whole tree:

```bash
pre-commit run --all-files
```

## Type hints

New public functions should carry PEP 484 annotations. For tensor-heavy code, annotate at
least the non-tensor arguments and use `torch.Tensor` for the rest; a more precise shape
annotation is not currently expressible and pretending otherwise makes the signature harder to
read rather than easier.

```python
def write_likelihoods(likelihood_filename: str, likelihoods: torch.Tensor) -> None:
    ...
```

## Docstrings

NumPy style for public classes and functions:

```python
def foo(x: int, y: float) -> str:
    """
    One-line summary.

    Parameters
    ----------
    x : int
        Description of x.
    y : float
        Description of y.

    Returns
    -------
    str
        Description of the return value.
    """
```

Comments should explain why the code is the way it is. A comment restating what the next line
does is worse than no comment, because it has to be maintained and it will drift.

## Tests

Tests live in `tests/` and use pytest without `unittest.TestCase` subclasses. They fall into two
groups. `test_model.py`, `test_metrics.py`, `test_metrics_module.py`, `test_scoring.py`,
`test_oracles.py`, `test_uncertainty.py`, `test_graph_roundtrip.py`, and
`test_conditioning.py` are self-contained and run anywhere.
`test_preprocessing.py` verifies a *completed preprocessing job*, so it needs
`tests/config.py` to point at a dataset directory that already holds the `.smi` and `.h5`
files; it defaults to `data/datasets/debug`.

```bash
pytest tests/ -v
pytest tests/test_preprocessing.py::TestSMILESReconstruction -v
```

No test needs network access. The oracle tests build their models in-process or inject a
stub docking backend through `VinaOracle(dock_fn=...)`, so a contributor can run the whole
suite without a docking installation or any downloaded model.

## Development install

Install PyTorch first, from pytorch.org for your platform and CUDA version, then:

```bash
pip install -e ".[dev]"     # black, ruff, mypy, pytest, pre-commit
pre-commit install
```

Note that the editable install does not reliably put `src/` on the import path when the
repository lives under a path containing spaces, such as an iCloud Drive directory; setuptools
writes a `.pth` file that is then not honoured. The symptoms are `ModuleNotFoundError: No
module named 'metrics'` and a `graphinvent-submit` console script that fails on import. Run
scripts from the repository root and add `sys.path.insert(0, "src")` where you need the
`metrics` or `oracles` packages, and invoke jobs as `python submit.py` rather than through the
console script.

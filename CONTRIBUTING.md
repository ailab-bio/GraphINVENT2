# Contributing to GraphINVENT2

## Code style

All Python code must pass **black** (formatting) and **ruff** (linting) before merging.

| Tool | Role | Config |
|------|------|--------|
| [black](https://black.readthedocs.io) | Auto-formatter, line length 88 | `[tool.black]` in `pyproject.toml` |
| [ruff](https://docs.astral.sh/ruff/) | Linter (pyflakes + pycodestyle + isort) | `[tool.ruff]` in `pyproject.toml` |
| [mypy](https://mypy.readthedocs.io) | Optional static type checking | `[tool.mypy]` in `pyproject.toml` |

### Running the checks manually

```bash
# Format
python -m black --line-length 88 graphinvent/ tests/ submit.py visualize.py

# Lint
python -m ruff check graphinvent/ tests/ submit.py visualize.py

# Type check (optional, best-effort for ML code)
python -m mypy graphinvent/ --ignore-missing-imports
```

### Pre-commit hooks

Install once after cloning:

```bash
pip install pre-commit
pre-commit install
```

After that, black and ruff run automatically on every `git commit`. To run them manually on all files:

```bash
pre-commit run --all-files
```

## Type hints

New public functions should include PEP 484 type annotations. For tensor-heavy code, annotate at least the non-tensor arguments; use `torch.Tensor` for tensor inputs/outputs. Example:

```python
def write_likelihoods(likelihood_filename: str, likelihoods: torch.Tensor) -> None:
    ...
```

## Docstrings

Use **NumPy-style** docstrings for all public classes and functions:

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

## Tests

Tests live in `tests/` and use **pytest** (no `unittest.TestCase` subclasses).

```bash
# Run all tests (edit tests/config.py to point at a preprocessed dataset first)
pytest tests/ -v

# Run a single test
pytest tests/test_preprocessing.py::test_valid_smiles -v
```

## Development install

```bash
pip install torch           # from pytorch.org for your CUDA version
pip install -e ".[dev]"     # installs ruff, pyright, pytest
pip install black pre-commit mypy
pre-commit install
```

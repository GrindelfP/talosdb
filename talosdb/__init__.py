"""
talosdb — lightweight experiment storage library.
Named after Talos I station.

Hierarchy:
    TalosDB  →  Experiment  →  Run
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_value(v: Any) -> str:
    """Format a parameter value for use in a folder name."""
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def _params_to_dirname(params: dict[str, Any]) -> str:
    """{'A': 0.5, 'beta': 1.0} → 'A=0.5_beta=1'"""
    return "_".join(f"{k}={_format_value(v)}" for k, v in sorted(params.items()))


def _dirname_to_params(dirname: str) -> dict[str, Any]:
    """
    'A=0.5_beta=1' → {'A': 0.5, 'beta': 1.0}

    Values: int if possible, then float, otherwise str.

    Fix: uses a regex split so parameter *values* containing underscores
    (e.g. ``method=runge_kutta``) are parsed correctly.
    """
    # Split at every '_' that is immediately followed by 'word=',
    # i.e. the start of a new key=value pair.
    parts = re.split(r"_(?=[^_=]+=)", dirname)
    result: dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            result[k] = int(v)
        except ValueError:
            try:
                result[k] = float(v)
            except ValueError:
                result[k] = v
    return result


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _load_run_params(run_path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    """
    Load parameters from ``params.json`` if it exists, otherwise return
    *fallback* (params parsed from the directory name).

    Using the JSON file preserves the original Python types (e.g. a float
    that was stored as 1.0 won't be silently parsed as int 1).
    """
    p = run_path / Run.PARAMS_FILE
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return fallback


# ---------------------------------------------------------------------------
# .dat format
# ---------------------------------------------------------------------------

def _save_dat(path: Path, array: np.ndarray) -> None:
    """
    Save a numpy array as a human-readable .dat file.

    Format::

        # shape: d0 d1 ...
        # dtype: float64
        # (1D / 2D) plain TSV rows
        # (3D+)     2D slices separated by blank lines,
        #           each preceded by '# slice [i, j, ...]'

    The shape header allows _load_dat() to reconstruct the exact array.
    """
    array = np.asarray(array)
    lines: list[str] = []

    # --- header ---
    shape_str = " ".join(str(d) for d in array.shape)
    lines.append(f"# shape: {shape_str}")
    lines.append(f"# dtype: {array.dtype}")

    if array.ndim == 0:
        # scalar
        lines.append(str(array.item()))

    elif array.ndim <= 2:
        # 1D → treat as single-column; 2D → plain TSV
        mat = array.reshape(-1, 1) if array.ndim == 1 else array
        for row in mat:
            lines.append("\t".join(str(x) for x in row))

    else:
        # 3D+: iterate over all leading indices, dump 2D slices
        leading_shape = array.shape[:-2]
        rows, cols = array.shape[-2], array.shape[-1]
        flat_slices = array.reshape(-1, rows, cols)
        leading_indices = list(
            itertools.product(*[range(d) for d in leading_shape])
        )

        for idx, slice_2d in zip(leading_indices, flat_slices):
            idx_str = ", ".join(str(i) for i in idx)
            lines.append(f"# slice [{idx_str}]")
            for row in slice_2d:
                lines.append("\t".join(str(x) for x in row))
            lines.append("")  # blank line between slices

    path.write_text("\n".join(lines), encoding="utf-8")


def _load_dat(path: Path) -> np.ndarray:
    """
    Load a .dat file saved by _save_dat() and return the original numpy array.
    Shape and dtype are reconstructed from the header.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    shape: tuple[int, ...] | None = None
    dtype_str: str = "float64"
    data_lines: list[str] = []

    for line in lines:
        line = line.rstrip()
        if line.startswith("# shape:"):
            shape = tuple(int(x) for x in line.split(":", 1)[1].split())
        elif line.startswith("# dtype:"):
            dtype_str = line.split(":", 1)[1].strip()
        elif line.startswith("#") or line == "":
            continue  # skip slice markers and blank lines
        else:
            data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No data found in {path}")

    rows = [list(map(float, row.split("\t"))) for row in data_lines]
    flat = np.array(rows, dtype=dtype_str)

    if shape is not None:
        return flat.reshape(shape)

    # Fallback (no shape header): squeeze single-column arrays to 1D.
    # Guard against flat being unexpectedly 1D before indexing axis 1.
    if flat.ndim == 2 and flat.shape[1] == 1:
        return flat.ravel()
    return flat


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class Run:
    """
    Represents a single simulation run inside an Experiment.

    Directory layout::

        <experiment_dir>/<A=0.5_beta=1.0>/
            data.dat      ← saved by run.save()
            params.json   ← saved by run.save_params()
            plot.png      ← saved by run.save_plot()
            failed.json   ← written on exception (context manager only)
    """

    DATA_FILE   = "data.dat"
    PARAMS_FILE = "params.json"
    PLOT_FILE   = "plot.png"

    def __init__(self, path: Path, name_params: dict[str, Any]) -> None:
        self._path        = path
        self._name_params = name_params
        self._path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name_params(self) -> dict[str, Any]:
        return dict(self._name_params)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, result: np.ndarray) -> None:
        """
        Save a numpy array as data.dat (human-readable TSV with header).

        Supports scalar, 1D, 2D, and arbitrary ND arrays.
        """
        _save_dat(self._path / self.DATA_FILE, np.asarray(result))

    def save_params(self, extra: Optional[dict[str, Any]] = None) -> None:
        """
        Save parameters to params.json.

        The name_params (those encoded in the folder name) are always
        included. *extra* adds any additional parameters
        (fixed constants, solver settings, etc.).

        All params are merged into a single flat JSON object so the file
        is self-contained and readable without knowing the folder name.
        """
        merged = dict(self._name_params)
        if extra:
            merged.update(extra)
        with open(self._path / self.PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    def save_plot(
        self,
        plot_fn: Callable[[Any, Path], None],
        result: Any,
        filename: str = PLOT_FILE,
    ) -> None:
        """
        Call ``plot_fn(result, save_path)`` to produce and save a figure.

        Contract for ``plot_fn``::

            def plot_fn(result, save_path):
                fig, ax = plt.subplots()
                ax.plot(result[:, 0], result[:, 1])
                fig.savefig(save_path)
                plt.close(fig)

        talosdb does not import matplotlib — all rendering is the caller's
        responsibility. The library only provides the save path.
        """
        plot_fn(result, self._path / filename)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_data(self) -> np.ndarray:
        """Load data.dat and return the original numpy array."""
        p = self._path / self.DATA_FILE
        if not p.exists():
            raise FileNotFoundError(f"No data file in run: {self._path}")
        return _load_dat(p)

    def load_params(self) -> dict[str, Any]:
        """Load params.json and return a dict."""
        p = self._path / self.PARAMS_FILE
        if not p.exists():
            raise FileNotFoundError(f"No params file in run: {self._path}")
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def is_failed(self) -> bool:
        """Return True if this run has a failed.json marker."""
        return (self._path / "failed.json").exists()

    def load_failure(self) -> dict[str, Any]:
        """Return the failure info dict, or raise FileNotFoundError."""
        p = self._path / "failed.json"
        if not p.exists():
            raise FileNotFoundError(f"No failure marker in run: {self._path}")
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """On exception — write failed.json inside the run folder."""
        if exc_type is not None:
            self._mark_failed(exc_val)
        return False  # re-raise the exception

    def _mark_failed(self, exc: BaseException) -> None:
        marker = {
            "status":    "failed",
            "error":     type(exc).__name__,
            "message":   str(exc),
            "timestamp": _now_stamp(),
        }
        with open(self._path / "failed.json", "w", encoding="utf-8") as f:
            json.dump(marker, f, indent=2)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Run(path={self._path})"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """
    A named collection of Runs inside a TalosDB.

    Metadata is stored in ``experiment.json`` at the experiment root::

        {
          "name":    "my_experiment",
          "created": "2025-06-10_14-30-00"
        }

    Note: the ``runs`` list that older versions stored in experiment.json has
    been removed. Ground truth is always the actual subdirectories on disk,
    which avoids the meta file going stale if folders are moved or deleted
    manually.
    """

    META_FILE = "experiment.json"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)
        self._meta = self._load_or_create_meta()

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def _load_or_create_meta(self) -> dict:
        meta_path = self._path / self.META_FILE
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        meta = {
            "name":    self._path.name,
            "created": _now_stamp(),
        }
        self._write_meta(meta)
        return meta

    def _write_meta(self, meta: Optional[dict] = None) -> None:
        if meta is None:
            meta = self._meta
        with open(self._path / self.META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._meta["name"]

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def run(
        self,
        params: dict[str, Any],
        overwrite: bool = False,
    ) -> Run:
        """
        Create (or reopen) a Run for the given parameter combination.

        The folder name is derived from *params*, e.g.
        ``{"A": 0.5, "beta": 1.0}`` → ``A=0.5_beta=1``.

        Parameters
        ----------
        params:
            Key-value pairs that identify this run.
        overwrite:
            If False (default) and the run folder already exists, a
            warning is issued so you know you're reopening an existing
            run. Pass ``overwrite=True`` to suppress the warning (e.g.
            when intentionally re-running and overwriting results).

        Supports use as a context manager::

            with exp.run({"A": 0.5, "beta": 1.0}) as run:
                run.save(result)
                run.save_params({"T": 300})
        """
        dirname  = _params_to_dirname(params)
        run_path = self._path / dirname

        if run_path.exists() and not overwrite:
            warnings.warn(
                f"Run '{dirname}' already exists — reopening. "
                "Pass overwrite=True to suppress this warning.",
                stacklevel=2,
            )

        return Run(run_path, params)

    # ------------------------------------------------------------------
    # Query / load
    # ------------------------------------------------------------------

    def load(self, params: dict[str, Any]) -> Run:
        """
        Load the Run that exactly matches *params*.

        Raises FileNotFoundError if not found.
        """
        dirname  = _params_to_dirname(params)
        run_path = self._path / dirname
        if not run_path.exists():
            raise FileNotFoundError(
                f"No run matching {params} in experiment '{self.name}'"
            )
        # Load full params from params.json so types are preserved.
        full_params = _load_run_params(run_path, params)
        return Run(run_path, full_params)

    def query(self, filters: dict[str, Any]) -> list[Run]:
        """
        Return all Runs whose parameters contain *filters* (subset match).

        Parameters are read from ``params.json`` when available, so type
        comparisons are reliable even for values that look like ints in
        the folder name.

        Example::

            runs = exp.query({"beta": 1.0})   # all runs where beta == 1.0
        """
        matches = []
        for run_path in self._iter_run_paths():
            name_params  = _dirname_to_params(run_path.name)
            full_params  = _load_run_params(run_path, name_params)
            if all(full_params.get(k) == v for k, v in filters.items()):
                matches.append(Run(run_path, full_params))
        return matches

    def all_runs(self) -> list[Run]:
        """Return all Runs in this experiment."""
        result = []
        for run_path in self._iter_run_paths():
            name_params = _dirname_to_params(run_path.name)
            full_params = _load_run_params(run_path, name_params)
            result.append(Run(run_path, full_params))
        return result

    def _iter_run_paths(self) -> Iterator[Path]:
        """Yield Path objects for each existing run subdirectory."""
        for child in sorted(self._path.iterdir()):
            if child.is_dir():
                yield child

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = sum(1 for _ in self._iter_run_paths())
        return f"Experiment(name={self.name!r}, runs={n})"


# ---------------------------------------------------------------------------
# TalosDB
# ---------------------------------------------------------------------------

class TalosDB:
    """
    Root database object. Points to a directory on disk.

    Usage::

        db  = TalosDB("~/science/results")
        exp = db.experiment("vc_sweep_2025")

        for A, beta in product(A_grid, beta_grid):
            result = simulate(A, beta)
            with exp.run({"A": A, "beta": beta}) as run:
                run.save(result)
                run.save_params({"gamma": 0.1, "T": 300})
                run.save_plot(my_plot_fn, result)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def experiment(self, name: Optional[str] = None) -> Experiment:
        """
        Create or open an experiment.

        If *name* is omitted, a datetime-stamped name is generated.
        """
        exp_name = name if name else _now_stamp()
        return Experiment(self._path / exp_name)

    def list_experiments(self) -> list[str]:
        """Return names of all experiment directories."""
        return sorted(d.name for d in self._path.iterdir() if d.is_dir())

    def delete_experiment(self, name: str, confirm: bool = False) -> None:
        """
        Delete an experiment and all its runs.

        *confirm=True* is required to prevent accidental deletion.
        """
        if not confirm:
            raise ValueError(
                "Pass confirm=True to delete_experiment — this is irreversible."
            )
        target = self._path / name
        if not target.exists():
            raise FileNotFoundError(f"Experiment '{name}' not found.")
        shutil.rmtree(target)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TalosDB(path={self._path}, "
            f"experiments={len(self.list_experiments())})"
        )
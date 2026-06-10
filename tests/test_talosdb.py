"""
Tests for talosdb.
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from talosdb import TalosDB, Experiment, Run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    return TalosDB(tmp_path / "testdb")


@pytest.fixture
def exp(db):
    return db.experiment("test_exp")


# ---------------------------------------------------------------------------
# _params_to_dirname / _dirname_to_params round-trip
# ---------------------------------------------------------------------------

class TestDirnameParsing:
    def test_basic_roundtrip(self):
        from talosdb import _params_to_dirname, _dirname_to_params
        params = {"A": 0.5, "beta": 1.0}
        assert _dirname_to_params(_params_to_dirname(params)) == params

    def test_int_value(self):
        from talosdb import _params_to_dirname, _dirname_to_params
        params = {"N": 100, "mode": 2}
        result = _dirname_to_params(_params_to_dirname(params))
        assert result == params

    def test_string_value(self):
        from talosdb import _params_to_dirname, _dirname_to_params
        params = {"solver": "euler", "A": 1.0}
        result = _dirname_to_params(_params_to_dirname(params))
        assert result == params

    def test_underscore_in_value(self):
        """Values containing underscores must not confuse the parser."""
        from talosdb import _params_to_dirname, _dirname_to_params
        params = {"method": "runge_kutta", "A": 0.5}
        dirname = _params_to_dirname(params)
        parsed  = _dirname_to_params(dirname)
        assert parsed["method"] == "runge_kutta"
        assert math.isclose(parsed["A"], 0.5)

    def test_sorted_keys(self):
        from talosdb import _params_to_dirname
        d1 = _params_to_dirname({"beta": 1.0, "A": 0.5})
        d2 = _params_to_dirname({"A": 0.5, "beta": 1.0})
        assert d1 == d2


# ---------------------------------------------------------------------------
# .dat round-trip
# ---------------------------------------------------------------------------

class TestDatFormat:
    def _roundtrip(self, arr, tmp_path):
        from talosdb import _save_dat, _load_dat
        p = tmp_path / "test.dat"
        _save_dat(p, arr)
        return _load_dat(p)

    def test_1d(self, tmp_path):
        arr = np.linspace(0, 1, 50)
        result = self._roundtrip(arr, tmp_path)
        assert result.shape == arr.shape
        np.testing.assert_array_almost_equal(result, arr)

    def test_2d(self, tmp_path):
        arr = np.random.rand(20, 3)
        result = self._roundtrip(arr, tmp_path)
        assert result.shape == arr.shape
        np.testing.assert_array_almost_equal(result, arr)

    def test_3d(self, tmp_path):
        arr = np.random.rand(4, 5, 6)
        result = self._roundtrip(arr, tmp_path)
        assert result.shape == arr.shape
        np.testing.assert_array_almost_equal(result, arr)

    def test_4d(self, tmp_path):
        arr = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
        result = self._roundtrip(arr, tmp_path)
        assert result.shape == arr.shape
        np.testing.assert_array_equal(result, arr)

    def test_human_readable(self, tmp_path):
        """The .dat file must be readable as plain text."""
        from talosdb import _save_dat
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = tmp_path / "test.dat"
        _save_dat(p, arr)
        text = p.read_text()
        assert "# shape:" in text
        assert "# dtype:" in text
        assert "1.0\t2.0" in text

    def test_integer_dtype(self, tmp_path):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = self._roundtrip(arr, tmp_path)
        assert result.dtype == arr.dtype
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class TestRun:
    def test_save_and_load_data(self, exp):
        run = exp.run({"A": 0.5, "beta": 1.0})
        data = np.random.rand(10, 2)
        run.save(data)
        loaded = run.load_data()
        np.testing.assert_array_almost_equal(loaded, data)

    def test_save_params_merges(self, exp):
        run = exp.run({"A": 0.5, "beta": 1.0})
        run.save_params({"gamma": 0.1, "T": 300})
        params = run.load_params()
        assert params["A"] == 0.5
        assert params["beta"] == 1.0
        assert params["gamma"] == 0.1
        assert params["T"] == 300

    def test_save_params_no_extra(self, exp):
        run = exp.run({"A": 1.0})
        run.save_params()
        params = run.load_params()
        assert params == {"A": 1.0}

    def test_save_plot(self, exp, tmp_path):
        run = exp.run({"A": 0.5})
        data = np.eye(3)
        saved_paths = []

        def fake_plot(result, save_path):
            save_path = Path(save_path)
            save_path.write_text("fake_image")
            saved_paths.append(save_path)

        run.save_plot(fake_plot, data)
        assert len(saved_paths) == 1
        assert saved_paths[0].exists()

    def test_load_data_missing_raises(self, exp):
        run = exp.run({"A": 99.0})
        with pytest.raises(FileNotFoundError):
            run.load_data()

    def test_load_params_missing_raises(self, exp):
        run = exp.run({"A": 99.0})
        with pytest.raises(FileNotFoundError):
            run.load_params()

    def test_context_manager_success(self, exp):
        data = np.arange(5, dtype=float)
        with exp.run({"A": 1.0}) as run:
            run.save(data)
        assert not run.is_failed()

    def test_context_manager_failure(self, exp):
        with pytest.raises(RuntimeError):
            with exp.run({"A": 2.0}) as run:
                raise RuntimeError("simulation exploded")
        assert run.is_failed()
        info = run.load_failure()
        assert info["error"] == "RuntimeError"
        assert "simulation exploded" in info["message"]

    def test_is_failed_false_by_default(self, exp):
        run = exp.run({"A": 3.0})
        assert not run.is_failed()

    def test_overwrite_warning(self, exp):
        exp.run({"A": 0.5, "beta": 1.0})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exp.run({"A": 0.5, "beta": 1.0})
            assert len(w) == 1
            assert "already exists" in str(w[0].message)

    def test_overwrite_no_warning(self, exp):
        exp.run({"A": 0.5, "beta": 1.0})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exp.run({"A": 0.5, "beta": 1.0}, overwrite=True)
            assert len(w) == 0


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class TestExperiment:
    def test_name(self, exp):
        assert exp.name == "test_exp"

    def test_meta_file_created(self, exp):
        assert (exp.path / "experiment.json").exists()

    def test_load_exact(self, exp):
        params = {"A": 0.5, "beta": 1.0}
        run = exp.run(params)
        run.save_params()
        loaded = exp.load(params)
        assert loaded.path == run.path

    def test_load_missing_raises(self, exp):
        with pytest.raises(FileNotFoundError):
            exp.load({"A": 999.0})

    def test_query_subset(self, exp):
        for beta in [1.0, 2.0]:
            for A in [0.5, 1.0]:
                r = exp.run({"A": A, "beta": beta})
                r.save_params()
        results = exp.query({"beta": 1.0})
        assert len(results) == 2
        assert all(r.load_params()["beta"] == 1.0 for r in results)

    def test_query_no_match(self, exp):
        exp.run({"A": 0.5}).save_params()
        assert exp.query({"A": 999.0}) == []

    def test_all_runs(self, exp):
        for i in range(3):
            exp.run({"i": i}).save_params()
        runs = exp.all_runs()
        assert len(runs) == 3

    def test_repr(self, exp):
        assert "test_exp" in repr(exp)


# ---------------------------------------------------------------------------
# TalosDB
# ---------------------------------------------------------------------------

class TestTalosDB:
    def test_creates_directory(self, tmp_path):
        path = tmp_path / "newdb"
        db = TalosDB(path)
        assert db.path.exists()

    def test_experiment_creates_subdir(self, db):
        exp = db.experiment("alpha")
        assert (db.path / "alpha").is_dir()

    def test_experiment_auto_name(self, db):
        exp = db.experiment()
        assert exp.path.exists()

    def test_list_experiments(self, db):
        db.experiment("a")
        db.experiment("b")
        db.experiment("c")
        names = db.list_experiments()
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_delete_experiment_requires_confirm(self, db):
        db.experiment("to_delete")
        with pytest.raises(ValueError, match="confirm=True"):
            db.delete_experiment("to_delete")

    def test_delete_experiment(self, db):
        db.experiment("to_delete")
        db.delete_experiment("to_delete", confirm=True)
        assert "to_delete" not in db.list_experiments()

    def test_delete_experiment_missing_raises(self, db):
        with pytest.raises(FileNotFoundError):
            db.delete_experiment("ghost", confirm=True)

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        db = TalosDB("~/mydb")
        assert db.path.exists()
        assert str(tmp_path) in str(db.path)

    def test_repr(self, db):
        assert "TalosDB" in repr(db)

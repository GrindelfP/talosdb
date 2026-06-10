# talosdb

Lightweight experiment storage library for scientific simulations.
Named after Talos I station.

## Installation

```bash
pip install src
```

## Quickstart

```python
from src import TalosDB
from itertools import product

db = TalosDB("~/science/results")
exp = db.experiment("vc_sweep_2025")

A_grid = [0.5, 1.0, 1.5]
beta_grid = [1.0, 2.0]

for A, beta in product(A_grid, beta_grid):
    result = simulate(A, beta)  # numpy array
    with exp.run({"A": A, "beta": beta}) as run:
        run.save(result)
        run.save_params({"gamma": 0.1, "T": 300})  # extra constants
        run.save_plot(my_plot_fn, result)
```

## File layout

```
db_root/
└── vc_sweep_2025/
    ├── experiment.json          # metadata: name, creation date
    ├── A=0.5_beta=1.0/
    │   ├── data.dat             # human-readable TSV (numpy array)
    │   ├── params.json          # all parameters (name + extra)
    │   └── plot.png             # if save_plot() was called
    └── A=1.0_beta=2.0/
        ├── data.dat
        └── params.json
```

## API reference

### TalosDB

```python
db = TalosDB("path/to/db")        # creates root folder if absent
db.experiment("name")              # create / open experiment
db.experiment()                    # name = datetime stamp
db.list_experiments()              # → list[str]
db.delete_experiment("name", confirm=True)
```

### Experiment

```python
exp = db.experiment("my_exp")

# Create / open a run
run  = exp.run({"A": 0.5, "beta": 1.0})
# or as context manager (marks failed.json on exception):
with exp.run({"A": 0.5, "beta": 1.0}) as run:
    ...

# Load
run  = exp.load({"A": 0.5, "beta": 1.0})   # exact match
runs = exp.query({"beta": 1.0})             # subset match → list[Run]
runs = exp.all_runs()                        # every run
```

### Run

```python
run.save(result)                            # numpy array → data.dat
run.save_params({"gamma": 0.1, "T": 300})  # extra params → params.json
run.save_plot(plot_fn, result)              # calls plot_fn(result, path)

result = run.load_data()                    # → np.ndarray
params = run.load_params()                  # → dict
run.is_failed()                             # → bool
run.load_failure()                          # → dict with error info
```

### plot_fn contract

```python
def my_plot_fn(result, save_path):
    fig, ax = plt.subplots()
    ax.plot(result[:, 0], result[:, 1])
    fig.savefig(save_path)
    plt.close(fig)
```

talosdb does not import matplotlib — rendering is entirely the caller's responsibility.

## .dat format

Arrays are stored as human-readable TSV with a small header:

```
# shape: 100 2
# dtype: float64
0.0	0.001
0.1	0.043
...
```

3D+ arrays are split into labelled 2D slices:

```
# shape: 2 3 4
# dtype: float64
# slice [0]
1.0	2.0	3.0	4.0
5.0	6.0	7.0	8.0
9.0	10.0	11.0	12.0

# slice [1]
...
```

The shape header ensures exact reconstruction on load regardless of dimensionality.

## License

MIT

# Copilot Instructions for trajan

## Build, test, and lint

```bash
# Run all tests (includes doctests in source files)
pytest

# Run a single test
pytest tests/test_repr.py::test_repr_1d

# Run with verbose output (as in CI)
pytest -vs --log-cli-level=debug

# Run slow / very slow tests (skipped by default)
pytest --run-slow
pytest --run-very-slow

# Show plots interactively during tests
pytest --plot

# Build package
poetry build

# Build docs
cd docs && make html
```

Doctests live inside the source modules and run automatically via `--doctest-modules` (set in `pyproject.toml`). Always check that docstring examples remain valid after changes.

## Architecture

`trajan` exposes a single xarray Dataset accessor `ds.traj`. Accessing it triggers auto-detection logic in `trajan/accessor.py` that inspects the dataset shape and returns one of three concrete subclasses of `Traj` (`trajan/traj.py`):

| Class | When used | Key trait |
|---|---|---|
| `Traj1d` | Shared time axis across trajectories | `time` dim is 1-D |
| `Traj2d` | Per-trajectory time, 2-D arrays | `time` or `obs` dim is 2-D |
| `ContiguousRagged` | CF contiguous ragged array (1-D storage) | `index` dim, `rowsize` variable |

`ContiguousRagged` converts to `Traj2d` internally for most operations. `Traj1d` and `Traj2d` can convert to each other via `to_1d()` / `to_2d()`.

Plotting lives in `trajan/plot/__init__.py`, accessed as `ds.traj.plot`. It auto-selects Cartopy GeoAxes for geographic data, plain matplotlib for Cartesian data.

Animation lives in `trajan/animation/__init__.py`, accessed as `ds.traj.animate()`. It uses a builder/chaining pattern — configure with method calls then call `.show()` or `.save()`. `FuncAnimation` is built with `blit=True`; all animated artists must have `animated=True` and be returned from the frame function.

## Key conventions

- **`xr.set_options(keep_attrs=True)`** is set globally in `accessor.py`. Dataset attributes are preserved through all operations.
- **CF conventions throughout** — stay true to CF trajectory conventions. Prefer `cf_xarray` for accessing variables and dimensions (e.g. `ds.cf['trajectory_id']`, `ds.cf[['time']]`) over hard-coded variable names.
- **Use pandas for datetime handling** — timedeltas, date ranges, and time arithmetic should use pandas. Use `.total_seconds()` on `pd.Timedelta`, not `int()` (removed in newer pandas).
- **Uniform interface across layouts** — `traj.py` is the accessor and defines the shared API. `Traj1d`, `Traj2d`, and `ContiguousRagged` should expose the same interface and hide the complexity of their different data layouts from the caller.
- **`Traj2d` is the most capable representation** — it is the most flexible and retains the highest degree of information. Prefer it when in doubt.
- **Polyfill pattern** — methods not natively supported by a layout can often be implemented by converting to `Traj1d` (via `gridtime` or `to_1d`), applying the operation, then collecting results back into the original or 2D dataset.
- **Animated matplotlib artists must be inside the axes** — with `blit=True`, only artists that are children of the axes are correctly restored after a window resize/move. Use `ax.text()` with `animated=True` for per-frame text; do not use `ax.set_title()` for updating content. Return all animated artists from the frame function.
- **Test fixtures** are defined in `tests/fixtures.py` and imported into all tests via `conftest.py`. Example datasets live in `examples/` (`barents.nc.xz`, `openoil.nc`). Add new shared fixtures there.
- **Slow tests** should be marked `@pytest.mark.slow` or `@pytest.mark.veryslow` so they are skipped in normal runs.

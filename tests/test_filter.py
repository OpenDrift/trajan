import numpy as np
import pytest
import xarray as xr
import trajan as _

from trajan.traj1d import _nsigma_sliding_filter
from trajan.readers.omb import sliding_filter_nsigma


@pytest.fixture
def barents_with_spike(barents):
    """Barents dataset with a position spike injected into trajectory 0."""
    ds = barents.copy(deep=True)
    ds['lon'].values[0, 100] += 50.0
    ds['lat'].values[0, 100] += 10.0
    return ds


def test_nsigma_sliding_filter_matches_omb():
    """_nsigma_sliding_filter must produce the same result as sliding_filter_nsigma in omb.py."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(200)
    arr[50] = 100.0   # outlier

    result_new = _nsigma_sliding_filter(arr.copy(), nsigma=5.0, side_half_width=2)
    result_omb = sliding_filter_nsigma(arr.copy(), nsigma=5.0, side_half_width=2)

    np.testing.assert_array_equal(result_new, result_omb)


def test_nsigma_sliding_filter_removes_spike():
    arr = np.zeros(20)
    arr[10] = 1000.0  # clear outlier
    result = _nsigma_sliding_filter(arr, nsigma=5.0, side_half_width=2)
    assert np.isnan(result[10])
    # boundary points are untouched
    assert not np.isnan(result[0])
    assert not np.isnan(result[-1])


def test_filter_speed_masks_spike(barents_with_spike):
    """Speed filter must mask the injected spike and leave no high-speed residuals."""
    max_speed = 3.0
    filtered = barents_with_spike.traj.filter(method='speed', max_speed=max_speed)

    # The spike at lon[0, 100] must be masked (new algo masks the destination, not source)
    assert np.isnan(filtered.lon.values[0, 100]), \
        "Speed filter did not mask the injected position spike"

    # No speed in the filtered dataset should exceed the threshold
    filtered_speed = filtered.traj.speed()
    assert float(filtered_speed.max(skipna=True)) <= max_speed, \
        f"High-speed positions remain after speed filter: max {float(filtered_speed.max(skipna=True)):.1f} m/s"


def test_filter_speed_clears_stuck_gps_run(barents):
    """Speed filter must clear an entire run of stuck GPS positions (e.g. (0,0) no-fix values),
    not just the boundary points, and must NOT mask the valid positions on either side."""
    import copy
    ds = barents.copy(deep=True)
    # Inject a run of 20 stuck-at-zero positions in the middle of trajectory 0
    run_start, run_end = 200, 220
    ds['lon'].values[0, run_start:run_end] = 1e-7
    ds['lat'].values[0, run_start:run_end] = 1e-7
    # shift times to include a big gap (simulate no-fix period)
    for i in range(run_start, run_end):
        ds['time'].values[0, i] = (
            ds['time'].values[0, run_start - 1] +
            np.timedelta64(int((i - run_start + 1) * 60), 's')
        )

    filtered = ds.traj.filter(method='speed', max_speed=3.0)

    # All 20 stuck positions must be NaN
    stuck = filtered.lon.values[0, run_start:run_end]
    assert np.all(np.isnan(stuck)), \
        f"Speed filter left {(~np.isnan(stuck)).sum()} stuck positions unmasked"

    # Valid positions immediately before and after must NOT be masked
    assert not np.isnan(filtered.lon.values[0, run_start - 1]), \
        "Speed filter falsely masked the valid position before the bad run"
    assert not np.isnan(filtered.lon.values[0, run_end]), \
        "Speed filter falsely masked the valid position after the bad run"


def test_filter_speed_removes_real_outliers(barents):
    """Speed filter on the raw barents data must remove all known high-speed positions."""
    max_speed = 3.0
    # Raw data has positions with > 3 m/s
    raw_speed = barents.traj.speed()
    assert float(raw_speed.max(skipna=True)) > max_speed, \
        "Test precondition: barents raw data should contain positions above threshold"

    filtered = barents.traj.filter(method='speed', max_speed=max_speed)
    filtered_speed = filtered.traj.speed()
    assert float(filtered_speed.max(skipna=True)) <= max_speed, \
        f"High-speed positions remain after speed filter: max {float(filtered_speed.max(skipna=True)):.1f} m/s"


def test_filter_speed_no_false_positives(barents):
    """Speed filter with a very high threshold should leave the clean dataset unchanged."""
    filtered = barents.traj.filter(method='speed', max_speed=1e6)
    np.testing.assert_array_equal(
        np.isnan(filtered.lon.values),
        np.isnan(barents.lon.values),
    )


def test_filter_nsigma_matches_omb_on_barents(barents):
    """nsigma_sliding filter result must match per-trajectory application of sliding_filter_nsigma."""
    filtered = barents.traj.filter(method='nsigma_sliding', nsigma=5.0, side_half_width=2)

    for ti in range(barents.sizes['trajectory']):
        lat_expected = sliding_filter_nsigma(
            barents.lat.values[ti], nsigma=5.0, side_half_width=2)
        lon_expected = sliding_filter_nsigma(
            barents.lon.values[ti], nsigma=5.0, side_half_width=2)
        np.testing.assert_array_equal(filtered.lat.values[ti], lat_expected)
        np.testing.assert_array_equal(filtered.lon.values[ti], lon_expected)


def test_filter_nsigma_masks_spike(barents_with_spike):
    filtered = barents_with_spike.traj.filter(method='nsigma_sliding', nsigma=5.0, side_half_width=2)
    assert np.isnan(filtered.lon.values[0, 100]), \
        "nsigma_sliding filter did not mask the injected position spike"


def test_filter_unknown_method_raises(barents):
    with pytest.raises(ValueError, match="Unknown filter method"):
        barents.traj.filter(method='unknown')

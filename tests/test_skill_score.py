import pytest
import xarray as xr
import pandas as pd
import trajan as ta
import numpy as np
from datetime import timedelta


def test_barents_self(barents):
    # this should be full score..
    barents = barents.traj.gridtime('1h')
    print(barents)
    skill = barents.traj.skill(barents)
    print(skill)

    np.testing.assert_allclose(skill.values, 1.)

def test_barents_trajs(barents):
    barents = barents.traj.gridtime('1h')
    b0 = barents.isel(trajectory=0).dropna('time')
    b1 = barents.isel(trajectory=1).sel(time=slice(b0.time[0], b0.time[-1]))
    b1 = b1.traj.gridtime(b0.time)
    skill = b0.traj.skill(b1, tolerance_threshold=1)
    print(skill)
    np.testing.assert_allclose(skill.values, 0.543, atol=0.001)

def test_barents_trajs_noround(barents):
    barents = barents.traj.gridtime('1h', round=False)
    b0 = barents.isel(trajectory=0).dropna('time')
    b1 = barents.isel(trajectory=1).sel(time=slice(b0.time[0], b0.time[-1]))
    b1 = b1.traj.gridtime(b0.time)
    skill = b0.traj.skill(b1, tolerance_threshold=1)
    print(skill)
    np.testing.assert_allclose(skill.values, 0.544782, atol=0.001)

def test_barents_align(barents):
    barents = barents.traj.gridtime('1h')
    b0 = barents.isel(trajectory=0)

    # assert b0.sizes['trajectory'] == 1
    assert barents.sizes['trajectory'] == 2

    (b01, _) = xr.broadcast(b0, barents)
    b01 = b01.transpose('trajectory', ...)

    np.testing.assert_allclose(b01.isel(trajectory=0).lon, barents.isel(trajectory=0).lon)
    np.testing.assert_allclose(b01.isel(trajectory=1).lon, barents.isel(trajectory=0).lon)

    skill = b01.traj.skill(barents)
    print(skill)


@pytest.mark.xfail(reason='Need opendrift version with test data')
def test_opendrift(plot, tmpdir):
    from opendrift.models.oceandrift import OceanDrift
    from opendrift.models.physics_methods import wind_drift_factor_from_trajectory, distance_between_trajectories, skillscore_liu_weissberg

    ot = OceanDrift(loglevel=50)
    ot.add_readers_from_list([
        ot.test_data_folder() +
        '16Nov2015_NorKyst_z_surface/norkyst800_subset_16Nov2015.nc',
        ot.test_data_folder() +
        '16Nov2015_NorKyst_z_surface/arome_subset_16Nov2015.nc'
    ],
                             lazy=False)
    ot.seed_elements(lon=4,
                     lat=60,
                     number=1,
                     time=ot.readers[list(ot.readers)[0]].start_time,
                     wind_drift_factor=0.033)

    ot.set_config('drift:horizontal_diffusivity', 10)
    ot.run(duration=timedelta(hours=12), time_step=600)

    drifter_lons = ot.history['lon'][0]
    drifter_lats = ot.history['lat'][0]
    drifter_times = ot.get_time_array()[0]
    drifter = {
        'lon': drifter_lons,
        'lat': drifter_lats,
        'time': drifter_times,
        'linewidth': 2,
        'color': 'b',
        'label': 'Synthetic drifter'
    }

    dds = ta.from_dataframe(pd.DataFrame(drifter), name='label')
    dds = dds.drop_vars(['linewidth', 'color'])
    dds = dds.traj.gridtime(dds.time.isel(trajectory=0).values) # convert to 1d dataset
    print(dds)

    o = OceanDrift(loglevel=50)
    o.add_readers_from_list([
        o.test_data_folder() +
        '16Nov2015_NorKyst_z_surface/norkyst800_subset_16Nov2015.nc',
        o.test_data_folder() +
        '16Nov2015_NorKyst_z_surface/arome_subset_16Nov2015.nc'
    ],
                            lazy=False)

    outf = str(tmpdir / 'o.nc')

    wdf = np.linspace(0.0, 0.05, 100)
    o.seed_elements(lon=4,
                    lat=60,
                    time=o.readers[list(o.readers)[0]].start_time,
                    wind_drift_factor=wdf,
                    number=len(wdf))
    o.run(duration=timedelta(hours=12), time_step=600, outfile=outf)

    if plot:
        o.plot(linecolor='wind_drift_factor', drifter=drifter)

    skillscore = o.skillscore_trajectory(drifter_lons,
                                         drifter_lats,
                                         drifter_times,
                                         tolerance_threshold=1)
    od_truth = [
        0.23451455, 0.24566153, 0.25828114, 0.27480025, 0.29290334, 0.31228188,
        0.33277579, 0.35437205, 0.37564148, 0.39638561, 0.41455774, 0.43061849,
        0.44466504, 0.45651809, 0.46716261, 0.47737433, 0.48706077, 0.49638797,
        0.50652886, 0.51707627, 0.52794453, 0.53965483, 0.55196713, 0.56465838,
        0.57759507, 0.59052492, 0.60278208, 0.61424894, 0.6251237, 0.63510257,
        0.64456457, 0.65318072, 0.66084812, 0.66814688, 0.67515132, 0.68315515,
        0.69175573, 0.70038038, 0.70939529, 0.7186845, 0.72782826, 0.73692918,
        0.74632541, 0.75522697, 0.76454528, 0.77396618, 0.78327117, 0.79279034,
        0.80185191, 0.81121632, 0.82047978, 0.8298343, 0.83933016, 0.84886072,
        0.85834259, 0.86815226, 0.87806759, 0.88724654, 0.89628481, 0.90559992,
        0.91461977, 0.92308472, 0.9318062, 0.94039675, 0.94861431, 0.95576439,
        0.96024486, 0.96192987, 0.96116929, 0.95807075, 0.95259218, 0.94448303,
        0.93478274, 0.92433786, 0.9138931, 0.90293765, 0.89201611, 0.88173708,
        0.87158623, 0.86122774, 0.85066312, 0.84014473, 0.82965122, 0.8193531,
        0.80900096, 0.79877646, 0.78823299, 0.77755203, 0.76701245, 0.75641498,
        0.74542862, 0.73478708, 0.72428215, 0.71395678, 0.70378888, 0.69367207,
        0.68359826, 0.67347744, 0.66335169, 0.65338939
    ]

    np.testing.assert_allclose(skillscore, od_truth)

    ods = xr.open_dataset(outf)
    print(ods)
    print(dds)

    dds = dds.isel(trajectory=0).broadcast_like(ods)
    tskill = dds.traj.skill(ods)
    print(tskill)

    np.testing.assert_allclose(tskill.values, od_truth)



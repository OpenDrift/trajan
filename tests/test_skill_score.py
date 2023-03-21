import xarray as xr
import trajan as _
import numpy as np
from datetime import timedelta


def test_barents_self(barents):
    # this should be full score..
    barents = barents.traj.gridtime('1H')
    print(barents)
    skill = barents.traj.skill(barents)
    print(skill)

    np.testing.assert_allclose(skill.values, 1.)


def test_barents_trajs(barents):
    barents = barents.traj.gridtime('1H')
    b0 = barents.isel(trajectory=0).dropna('time')
    b1 = barents.isel(trajectory=1).sel(time=slice(b0.time[0], b0.time[-1]))
    b1 = b1.traj.gridtime(b0.time)
    skill = b0.traj.skill(b1, tolerance_threshold=100)
    print(skill)
    np.testing.assert_allclose(skill.values, 0.)


def test_opendrift(plot):
    from opendrift.models.oceandrift import OceanDrift
    from opendrift.models.physics_methods import wind_drift_factor_from_trajectory, distance_between_trajectories, skillscore_liu_weissberg

    ot = OceanDrift(loglevel=0)
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

    o = OceanDrift(loglevel=0)
    o.add_readers_from_list([
        o.test_data_folder() +
        '16Nov2015_NorKyst_z_surface/norkyst800_subset_16Nov2015.nc',
        o.test_data_folder() +
        '16Nov2015_NorKyst_z_surface/arome_subset_16Nov2015.nc'
    ],
                            lazy=False)

    wdf = np.linspace(0.0, 0.05, 100)
    o.seed_elements(lon=4,
                    lat=60,
                    time=o.readers[list(o.readers)[0]].start_time,
                    wind_drift_factor=wdf,
                    number=len(wdf))
    o.run(duration=timedelta(hours=12), time_step=600)

    if plot:
        o.plot(linecolor='wind_drift_factor', drifter=drifter)

    skillscore = o.skillscore_trajectory(drifter_lons, drifter_lats, drifter_times, tolerance_threshold=1)
    print(skillscore)



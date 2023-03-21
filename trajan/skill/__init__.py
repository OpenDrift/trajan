import pyproj
import numpy as np

def distance_between_trajectories(lon1, lat1, lon2, lat2):
    '''Calculate the distances [m] between two trajectories'''

    assert len(lon1) == len(lat1) == len(lat1) == len(lat2)
    geod = pyproj.Geod(ellps='WGS84')
    _azimuth_forward, _a2, distance = geod.inv(lon1, lat1, lon2, lat2)

    return distance

def distance_along_trajectory(lon, lat):
    '''Calculate the distances [m] between points along a trajectory'''

    geod = pyproj.Geod(ellps='WGS84')
    _azimuth_forward, _a2, distance = geod.inv(lon[1:], lat[1:], lon[0:-1], lat[0:-1])

    return distance

def liu_weissberg(lon_obs, lat_obs, lon_model, lat_model, tolerance_threshold=1):
    '''
    Calculate skill score from normalized cumulative seperation distance. Liu and Weisberg 2011.

    Returns:

        Skill score between 0. and 1.
    '''

    lon_obs = np.array(lon_obs)
    lat_obs = np.array(lat_obs)
    lon_model = np.array(lon_model)
    lat_model = np.array(lat_model)
    d = distance_between_trajectories(lon_obs, lat_obs, lon_model, lat_model)
    l = distance_along_trajectory(lon_obs, lat_obs)
    s = np.nansum(d) / np.nansum(np.nancumsum(l))
    if tolerance_threshold==0:
        skillscore = 0
    else:
        skillscore = max(0, 1 - s/tolerance_threshold)

    return skillscore


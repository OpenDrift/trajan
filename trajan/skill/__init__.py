import logging
import pyproj
import numpy as np

logger = logging.getLogger(__name__)


def distance_between_trajectories(lon1, lat1, lon2, lat2):
    '''Calculate the distances [m] between two trajectories'''

    assert len(lon1) == len(lat1) == len(lon2) == len(lat2)
    geod = pyproj.Geod(ellps='WGS84')
    _azimuth_forward, _a2, distance = geod.inv(lon1, lat1, lon2, lat2)

    return distance

def distance_along_trajectory(lon, lat):
    '''Calculate the distances [m] between points along a trajectory'''

    geod = pyproj.Geod(ellps='WGS84')
    _azimuth_forward, _a2, distance = geod.inv(lon[1:], lat[1:], lon[0:-1], lat[0:-1])

    return distance

def liu_weissberg(lon_obs, lat_obs, lon_model, lat_model, tolerance_threshold=1, cumulative=False):
    '''
    Calculate skill score from normalized cumulative separation distance. Liu and Weisberg 2011.

    Parameters:
        cumulative: If True, return an array of skill scores at each time step
                    (shape same as input, time first axis), with NaN at t=0.
                    At step k the score is max(0, 1 - s_k/n₀) where
                    s_k = (Σ_{i=0}^{k} d_i) / (Σ_{i=1}^{k} L_i), i.e. the
                    Liu-Weisberg formula evaluated on the data up to step k.
                    The last value therefore equals the non-cumulative score.
                    The score decreases at step k when d_k/L_k exceeds the
                    running average s_{k-1} (accelerating divergence).

    Returns:

        Skill score between 0. and 1., or array thereof if cumulative=True.
    '''

    lon_obs = np.array(lon_obs)
    lat_obs = np.array(lat_obs)
    lon_model = np.array(lon_model)
    lat_model = np.array(lat_model)

    d = distance_between_trajectories(lon_obs, lat_obs, lon_model, lat_model)
    l = distance_along_trajectory(lon_obs, lat_obs)

    if cumulative:
        # Running Liu-Weisberg: at step k, evaluate the formula on data [0..k].
        # Numerator: Σ d_i, Denominator: Σ L_i (sum of cumulative arc lengths).
        # At k=0 the denominator is 0 → NaN; at k=N-1 the value equals the
        # non-cumulative score.  Score decreases when d_k/L_k > running average
        # (accelerating divergence).
        cum_d = np.nancumsum(d, axis=0)            # shape (N, ...)
        cum_L = np.nancumsum(np.nancumsum(l, axis=0), axis=0)  # shape (N-1, ...)
        # s[k] = Σ_{0..k+1} d / Σ_{1..k+1} L  →  shape (N-1, ...)
        s = cum_d[1:] / cum_L
        if tolerance_threshold == 0:
            skillscore = np.where(np.isfinite(s), 0.0, np.nan)
        else:
            skillscore = np.maximum(0, 1 - s / tolerance_threshold)
        # Prepend NaN for t=0 (denominator undefined, no arc length yet)
        nan_row = np.full((1,) + skillscore.shape[1:], np.nan)
        skillscore = np.concatenate([nan_row, skillscore], axis=0)
    else:
        s = np.nansum(d, axis=0) / np.nansum(np.nancumsum(l, axis=0), axis=0)
        if tolerance_threshold == 0:
            skillscore = 0
        else:
            skillscore = np.maximum(0, 1 - s / tolerance_threshold)

    return skillscore

def darpa(lon1, lat1, lon2, lat2):
    '''Scoring algorithm used for DARPA float challenge 2021.

    Copied from implementation made by Jean Rabault.
    Assuming 6 positions, separated by 2 days, where first pos (start) are identical.
    Arrays can be multidimensional, but first dimension must be time (along trajectory).
    '''

    assert len(lon1) == 6
    distances = distance_between_trajectories(lon1, lat1, lon2, lat2)/1000
    if distances[0] != 0:
        logger.warning('DARPA score: first position is not identical '
                       '(distance: %s km)' % distances[0])
    distances = distances[1:]  # For the remaining, we ignore starting position

    dict_DARPA_points = {
        4: 5,
        8: 2,
        16: 1,
        32: 0.25 }
    distance_thresholds = sorted(list(dict_DARPA_points.keys()))

    dict_DARPA_scoring_multiplicator = {
        0: 1,
        1: 2,
        2: 5,
        3: 10,
        4: 20 }

    DARPA_points = 0
    for crrt_ind, crrt_distance in enumerate(distances):
        if crrt_distance >= distance_thresholds[-1]:
            break

        for crrt_threshold in distance_thresholds:
            if crrt_distance < crrt_threshold:
                crrt_points = dict_DARPA_points[crrt_threshold]
                break

        crrt_multiplicator = dict_DARPA_scoring_multiplicator[crrt_ind]

        DARPA_points += int(crrt_points * crrt_multiplicator)

    return DARPA_points


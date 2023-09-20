import unittest
import pytest
import numpy as np
import math
try:
    from parcels import FieldSet
    has_parcels = True
except:
    has_parcels = False


@pytest.fixture(scope='session')
@unittest.skipIf(has_parcels is False,
                 'Parcels library is not installed')
def moving_eddies_fieldset(xdim=200, ydim=350, mesh='flat'):
    """
    From: https://oceanparcels.org/examples-data/MovingEddies_data/
    """
    # Set Parcels FieldSet variables
    time = np.arange(0., 8. * 86400., 86400., dtype=np.float64)

    # Coordinates of the test fieldset (on A-grid in m)
    if mesh is 'spherical':
        lon = np.linspace(0, 4, xdim, dtype=np.float32)
        lat = np.linspace(45, 52, ydim, dtype=np.float32)
    else:
        lon = np.linspace(0, 4.e5, xdim, dtype=np.float32)
        lat = np.linspace(0, 7.e5, ydim, dtype=np.float32)

    # Grid spacing in m
    def cosd(x):
        return math.cos(math.radians(float(x)))

    dx = (lon[1] - lon[0]) * 1852 * 60 * cosd(
        lat.mean()) if mesh is 'spherical' else lon[1] - lon[0]
    dy = (lat[1] -
          lat[0]) * 1852 * 60 if mesh is 'spherical' else lat[1] - lat[0]

    # Define arrays U (zonal), V (meridional), and P (sea surface height) on A-grid
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    # Some constants
    corio_0 = 1.e-4  # Coriolis parameter
    h0 = 1  # Max eddy height
    sig = 0.5  # Eddy e-folding decay scale (in degrees)
    g = 10  # Gravitational constant
    eddyspeed = 0.1  # Translational speed in m/s
    dX = eddyspeed * 86400 / dx  # Grid cell movement of eddy max each day
    dY = eddyspeed * 86400 / dy  # Grid cell movement of eddy max each day

    [x, y] = np.mgrid[:lon.size, :lat.size]
    for t in range(time.size):
        hymax_1 = lat.size / 7.
        hxmax_1 = .75 * lon.size - dX * t
        hymax_2 = 3. * lat.size / 7. + dY * t
        hxmax_2 = .75 * lon.size - dX * t

        P[:, :, t] = h0 * np.exp(-(x - hxmax_1)**2 / (sig * lon.size / 4.)**2 -
                                 (y - hymax_1)**2 / (sig * lat.size / 7.)**2)
        P[:, :,
          t] += h0 * np.exp(-(x - hxmax_2)**2 / (sig * lon.size / 4.)**2 -
                            (y - hymax_2)**2 / (sig * lat.size / 7.)**2)

        V[:-1, :, t] = -np.diff(P[:, :, t], axis=0) / dx / corio_0 * g
        V[-1, :, t] = V[-2, :, t]  # Fill in the last column

        U[:, :-1, t] = np.diff(P[:, :, t], axis=1) / dy / corio_0 * g
        U[:, -1, t] = U[:, -2, t]  # Fill in the last row

    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lon': lon, 'lat': lat, 'time': time}

    #     fieldset = moving_eddies_fieldset()
    #     filename = 'moving_eddies'
    #     fieldset.write(filename)

    return FieldSet.from_data(data, dimensions, transpose=True, mesh=mesh)

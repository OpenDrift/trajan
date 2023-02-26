import trajan as ta
import xarray as xr
import cf_xarray as _
import matplotlib.pyplot as plt

def test_barents_detect_lonlat(barents, plot):
    print(barents.traj.crs)

    crs = barents.traj.crs
    assert crs.to_cf()['grid_mapping_name'] == 'latitude_longitude'
    print(crs.to_cf())



def test_barents_set_crs(barents, plot):
    crs = barents.traj.crs
    barents.traj.set_crs(crs)

    assert 'latitude_longitude' in barents
    assert len(barents.cf.grid_mapping_names) > 0
    assert barents.traj.crs == crs

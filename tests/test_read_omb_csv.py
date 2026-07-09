import xarray as xr
import trajan as ta
from trajan.readers.omb import read_omb_csv


def test_read_csv_omb_default_waves():
    ds = read_omb_csv(ta.DATA_DIR + 'omb/omb1.csv')

    # 1668207637
    assert ds.attrs['time_coverage_start'] == '2022-11-12T00:00:37'
    assert ds.attrs['time_coverage_end'] == '2022-11-12T02:30:27'

    assert ds.sizes['trajectory'] == 2
    print(ds.time)

    # grid dataset
    # ds = ds.traj.gridtime('1h')
    # print(ds)

    ds.to_netcdf(ta.DATA_DIR + 'test.nc')

    ds2 = xr.open_dataset(ta.DATA_DIR + 'test.nc', decode_cf=True)
    assert ds2.attrs['time_coverage_start'] == '2022-11-12T00:00:37'
    assert ds2.attrs['time_coverage_end'] == '2022-11-12T02:30:27'

    assert ds2.sizes['trajectory'] == 2

    assert ds == ds2


def test_read_csv_omb_modified_waves():

    dict_wave_packet_params = {
        "_BD_YWAVE_PACKET_MIN_BIN": 9,
        "_BD_YWAVE_PACKET_MAX_BIN": 94,
        "LENGTH_FROM_SERIAL_OUTPUT": 198,
    }

    xr_result = read_omb_csv(ta.DATA_DIR + 'omb/omb2.csv', modified_wave_packet_properties=dict_wave_packet_params)

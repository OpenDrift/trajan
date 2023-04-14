import pandas as pd
from trajan.readers.omb import read_omb_csv


def test_read_csv_omb_default_waves(test_data, tmpdir):
    path_to_test_data = test_data / 'csv' / 'omb1.csv'
    ds = read_omb_csv(path_to_test_data)

    # 1668207637
    assert ds.attrs['time_coverage_start'] == '2022-11-12T00:00:37+00:00'
    assert ds.attrs['time_coverage_end'] == '2022-11-12T02:30:27+00:00'

    assert ds.dims['trajectory'] == 2
    print(ds.time)

    # grid dataset
    # ds = ds.traj.gridtime('1h')
    # print(ds)

    ds.to_netcdf(tmpdir / 'test.nc')


def test_read_csv_omb_modified_waves(test_data):
    path_to_test_data = test_data / 'csv' / 'omb2.csv'

    dict_wave_packet_params = {
        "_BD_YWAVE_PACKET_MIN_BIN": 9,
        "_BD_YWAVE_PACKET_MAX_BIN": 94,
        "LENGTH_FROM_SERIAL_OUTPUT": 198,
    }

    xr_result = read_omb_csv(path_to_test_data, modified_wave_packet_properties=dict_wave_packet_params)

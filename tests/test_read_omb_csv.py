from pathlib import Path
from trajan.readers.omb import read_omb_csv


def test_read_csv_omb_default_waves(test_data):
    path_to_test_data = test_data / 'csv/omb1.csv'
    xr_result = read_omb_csv(path_to_test_data)


def test_read_csv_omb_modified_waves(test_data):
    path_to_test_data = test_data / 'csv/omb2.csv'

    dict_wave_packet_params = {
        "_BD_YWAVE_PACKET_MIN_BIN": 9,
        "_BD_YWAVE_PACKET_MAX_BIN": 94,
        "LENGTH_FROM_SERIAL_OUTPUT": 198,
    }

    xr_result = read_omb_csv(path_to_test_data, modified_wave_packet_properties=dict_wave_packet_params)

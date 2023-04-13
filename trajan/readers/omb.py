
from pathlib import Path
import xarray as xr
import logging
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np

from omb_decoder import decode_message
from typing import Union
from omb_decoder import GNSS_Metadata, Waves_Metadata, Thermistors_Metadata, GNSS_Packet, Waves_Packet, Thermistors_Packet, _BD_YWAVE_NBR_BINS

logger = logging.getLogger(__name__)


@dataclass
class ParsedIridiumMessage:
    device_from: str
    kind: str
    meta: Union[GNSS_Metadata, Waves_Metadata, Thermistors_Metadata]
    data: Union[GNSS_Packet, Waves_Packet, Thermistors_Packet]


def append_dict_with_entry(dict_in: dict, parsed_entry: ParsedIridiumMessage):
    if parsed_entry.device_from not in dict_in:
        dict_in[parsed_entry.device_from] = {}
        dict_in[parsed_entry.device_from]["G"] = []
        dict_in[parsed_entry.device_from]["Y"] = []
    dict_in[parsed_entry.device_from][parsed_entry.kind].append(parsed_entry)


def read_omb_csv(path_in: Path) -> xr.Dataset:
    logger.debug("read path to pandas")

    ########################################
    # generic pandas read to be able to open from a variety of files

    omb_dataframe = pd.read_csv(path_in)

    ########################################
    # check this is actually a Rock7 data file

    columns = omb_dataframe.columns.to_list()
    expected_columns = [
        'Date Time (UTC)',
        'Device',
        'Direction',
        'Payload',
        'Approx Lat/Lng',
        'Payload (Text)',
        'Length (Bytes)',
        'Credits'
    ]

    if columns != expected_columns:
        raise RuntimeError(f"does not look like a Rock7 file; got colmns {columns}, expected {expected_columns}")

    ########################################
    # decode

    dict_entries = {}
    number_valid_entries = 0
    number_pd_entries = len(omb_dataframe)

    frequencies = _BD_YWAVE_NBR_BINS * [np.nan]
    frequencies_set = False

    # decode each entry;
    # only consider messages from the buoy to owner
    # only consider non empty messages
    # be verbose about messages that cannot be decoded: something is seriously wrong then!
    for pd_index, crrt_data in tqdm(omb_dataframe.iterrows(), total=number_pd_entries):
        # only use data from the buoy
        if crrt_data.Direction != "MO":
            logger.debug(f"omb_dataframe at index {pd_index} is:\n{crrt_data}\nthis is not a from buoy (Direction: MO) message, drop")
            continue

        # only use non empty data frames
        if getattr(crrt_data, "Length (Bytes)") == 0:
            logger.debug(f"omb_dataframe at index {pd_index} is:\n{crrt_data}\nthis is empty (Length (Bytes) is 0), drop")
            continue

        # we should only have valid dataframes at this point; attempt to decode
        try:
            crrt_kind, crrt_meta, crrt_list_packets = decode_message(crrt_data.Payload, print_decoded=False)
            number_valid_entries += 1

            # a GNSS packet may contain several data entries; split it here for simplicity
            if crrt_kind == "G":
                for crrt_fix in crrt_list_packets:
                    crrt_parsed = ParsedIridiumMessage(
                        device_from = crrt_data.Device,
                        kind = crrt_kind,
                        meta = crrt_meta,
                        data = crrt_fix,
                    )

                    append_dict_with_entry(dict_entries, crrt_parsed)

            # other packets contain a single entry: add as is
            else:
                if crrt_kind == "Y" and not frequencies_set:
                    frequencies_set = True
                    frequencies = crrt_list_packets[0].list_frequencies

                crrt_parsed = ParsedIridiumMessage(
                    device_from = crrt_data.Device,
                    kind = crrt_kind,
                    meta = crrt_meta,
                    data = crrt_list_packets[0],
                )

                append_dict_with_entry(dict_entries, crrt_parsed)

        except Exception as e:
            logger.warning(f"attempt to decode entry at index {pd_index}, Payload equal to: {crrt_data.Payload} failed with exception:\n{e}")
            continue

    if number_valid_entries == 0:
        logger.warning("got no valid decoded payload in the whole csv file; are you sure this is an OMB csv iridium file?")

    ########################################
    # turn into a trajan compatible format

    # determine the size for trajectory, obs, imu_obs, frequencies
    trajectory = len(dict_entries)

    obs_gnss = max([len(dict_entries[crrt_instr]["G"]) for crrt_instr in dict_entries])

    obs_waves_imu = max([len(dict_entries[crrt_instr]["Y"]) for crrt_instr in dict_entries])

    frequencies_waves_imu = _BD_YWAVE_NBR_BINS

    list_instruments = sorted(list(dict_entries.keys()))

    int64_fill = -(2**63 - 0)

    # create and fill the xarray dataset
    xr_result = xr.Dataset(
        {
            # meta vars
            #
            'drifter_names': xr.DataArray(
                data=list_instruments,
                dims=['trajectory'],
                attrs={
                    "cf_role": "trajectory_id",
                    "standard_name": "platform_id",
                }
            ).astype(str),
            #
            'frequencies_waves_imu': xr.DataArray(
                data=frequencies,
                dims=["frequencies_waves_imu"],
                attrs={
                    "_FillValue": "NaN",
                }
            ),
            #
            # gnss position vars
            #
            'time': xr.DataArray(
                dims=["trajectory", "obs"],
                data=int64_fill*np.ones((trajectory, obs_gnss), dtype=np.int64),
                attrs={
                    "_FillValue": str(int64_fill),
                    "standard_name": "time",
                    "unit": "seconds since 1970-01-01T00:00:00+00:00",
                    "time_calendar": "proleptic_gregorian",
                }
            ),
            #
            'lat': xr.DataArray(
                dims=["trajectory", "obs"],
                data=np.nan*np.ones((trajectory, obs_gnss)),
                attrs={
                    "_FillValue": "NaN",
                    "standard_name": "latitude",
                    "unit": "degree_north",
                }
            ),
            #
            'lon': xr.DataArray(
                dims=["trajectory", "obs"],
                data=np.nan*np.ones((trajectory, obs_gnss)),
                attrs={
                    "_FillValue": "NaN",
                    "standard_name": "longitude",
                    "unit": "degree_east",
                }
            ),
            #
            # imu waves vars
            #
            'accel_energy_spectrum': xr.DataArray(
                dims=["trajectory", "obs_waves_imu", "frequencies_waves_imu"],
                data=np.nan*np.ones((trajectory, obs_waves_imu, frequencies_waves_imu)),
                attrs={
                    "_FillValue": "NaN",
                }
            ),
            #
            'elevation_energy_spectrum': xr.DataArray(
                dims=["trajectory", "obs_waves_imu", "frequencies_waves_imu"],
                data=np.nan*np.ones((trajectory, obs_waves_imu, frequencies_waves_imu)),
                attrs={
                    "_FillValue": "NaN",
                }
            ),
            #
            'processed_elevation_energy_spectrum': xr.DataArray(
                dims=["trajectory", "obs_waves_imu", "frequencies_waves_imu"],
                data=np.nan*np.ones((trajectory, obs_waves_imu, frequencies_waves_imu)),
                attrs={
                    "_FillValue": "NaN",
                }
            ),
            #
            'pcutoff': xr.DataArray(
                dims=["trajectory", "obs_waves_imu"],
                data=np.nan*np.ones((trajectory, obs_waves_imu)),
                attrs={
                    "_FillValue": "NaN",
                }
            ),
        },
        attrs={
            "Conventions": "CF-1.10",
            "featureType": "trajectory",
        }
    )

    # actually fill the data
    for crrt_instrumnent_idx, crrt_instrument in enumerate(list_instruments):
        # gnss position data
        #
        list_time = [int(crrt_packet.data.datetime_fix.timestamp()) for crrt_packet in dict_entries[crrt_instrument]["G"]]
        xr_result["time"][crrt_instrumnent_idx, 0:len(list_time)] = list_time
        #
        list_lat = [crrt_packet.data.latitude for crrt_packet in dict_entries[crrt_instrument]["G"]]
        xr_result["lat"][crrt_instrumnent_idx, 0:len(list_lat)] = list_lat
        #
        list_lon = [crrt_packet.data.longitude for crrt_packet in dict_entries[crrt_instrument]["G"]]
        xr_result["lon"][crrt_instrumnent_idx, 0:len(list_lon)] = list_lon

        # wave data

    return xr_result


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.ERROR)

    print("start main")

    path_to_test_data = Path.cwd().parent.parent / "tests" / "test_data" / "csv" / "omb1.csv"
    xr_result = read_omb_csv(path_to_test_data)

    print(xr_result)
    xr_result.to_netcdf("test.nc")

    print("done main")

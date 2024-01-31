from pathlib import Path
import xarray as xr
import logging
import pandas as pd
from dataclasses import dataclass
import numpy as np
import datetime

from .omb_decoder import decode_message
from typing import Union
from .omb_decoder import GNSS_Metadata, Waves_Metadata, Thermistors_Metadata, GNSS_Packet, Waves_Packet, Thermistors_Packet, _BD_YWAVE_NBR_BINS

logger = logging.getLogger(__name__)


@dataclass
class ParsedIridiumMessage:
    device_from: str
    kind: str
    meta: Union[GNSS_Metadata, Waves_Metadata, Thermistors_Metadata]
    data: Union[GNSS_Packet, Waves_Packet, Thermistors_Packet]


def sliding_filter_nsigma(np_array_in, nsigma=5.0, side_half_width=2):
    """Perform a sliding filter, on points of indexes
    [idx-side_half_width; idx+side_half_width], to remove outliers. I.e.,
    the [idx] point gets removed if it is more than nsigma deviations away
    from the mean of the whole segment.

    np_array_in should have a shape (nbr_of_entries,)"""

    np_array = np.copy(np_array_in)
    array_len = np_array.shape[0]

    middle_point_index_start = side_half_width
    middle_point_index_end = array_len - side_half_width - 1

    for crrt_middle_index in range(middle_point_index_start,
                                   middle_point_index_end + 1, 1):
        crrt_left_included = crrt_middle_index - side_half_width
        crrt_right_included = crrt_middle_index + side_half_width
        crrt_array_data = np.concatenate([
            np_array_in[crrt_left_included:crrt_middle_index],
            np_array_in[crrt_middle_index + 1:crrt_right_included + 1]
        ])
        mean = np.mean(crrt_array_data)
        std = np.std(crrt_array_data)
        if np.abs(np_array[crrt_middle_index] - mean) > nsigma * std:
            logger.debug("found outlier in sliding_filter_nsigma")
            np_array[crrt_middle_index] = np.nan

    return np_array


def append_dict_with_entry(dict_in: dict, parsed_entry: ParsedIridiumMessage):
    if parsed_entry.device_from not in dict_in:
        dict_in[parsed_entry.device_from] = {}
        dict_in[parsed_entry.device_from]["G"] = []
        dict_in[parsed_entry.device_from]["Y"] = []
        dict_in[parsed_entry.device_from]["T"] = []
    dict_in[parsed_entry.device_from][parsed_entry.kind].append(parsed_entry)


def read_omb_csv(path_in: Path,
                 dict_instruments_params: dict = None,
                 modified_wave_packet_properties: dict = None) -> xr.Dataset:
    logger.debug("read path to pandas")

    if modified_wave_packet_properties is None:
        nbr_bins_waves = _BD_YWAVE_NBR_BINS
    else:
        nbr_bins_waves = modified_wave_packet_properties[
            "_BD_YWAVE_PACKET_MAX_BIN"] - modified_wave_packet_properties[
                "_BD_YWAVE_PACKET_MIN_BIN"]

    ########################################
    # generic pandas read to be able to open from a variety of files

    omb_dataframe = pd.read_csv(path_in)

    ########################################
    # check this is actually a Rock7 data file

    columns = omb_dataframe.columns.to_list()
    expected_columns = [
        'Date Time (UTC)', 'Device', 'Direction', 'Payload', 'Approx Lat/Lng',
        'Payload (Text)', 'Length (Bytes)', 'Credits'
    ]

    if not set(expected_columns).issubset(set(columns)):
        raise RuntimeError(
            f"does not look like a Rock7 file; got colmns {columns}, expected {expected_columns}, missing: {set(expected_columns) - set(columns)}"
        )

    ########################################
    # decode

    dict_entries = {}
    number_valid_entries = 0

    frequencies = nbr_bins_waves * [np.nan]
    frequencies_set = False

    # decode each entry;
    # only consider messages from the buoy to owner
    # only consider non empty messages
    # be verbose about messages that cannot be decoded: something is seriously wrong then!
    for pd_index, crrt_data in omb_dataframe.iterrows():
        # only use data from the buoy
        if crrt_data.Direction != "MO":
            logger.debug(
                f"omb_dataframe at index {pd_index} is:\n{crrt_data}\nthis is not a from buoy (Direction: MO) message, drop"
            )
            continue

        # only use non empty data frames
        if getattr(crrt_data, "Length (Bytes)") == 0:
            logger.debug(
                f"omb_dataframe at index {pd_index} is:\n{crrt_data}\nthis is empty (Length (Bytes) is 0), drop"
            )
            continue

        # only use data that are after the start time for the current buoy
        crrt_start_time = None

        if dict_instruments_params is not None:
            crrt_instrument = getattr(crrt_data, "Device")
            if crrt_instrument in dict_instruments_params:
                if "start_time" in dict_instruments_params[crrt_instrument]:
                    crrt_start_time = dict_instruments_params[crrt_instrument]["start_time"]

        # we should only have valid dataframes at this point; attempt to decode
        # hard to catch exceptions in a fine grain way, so cath all, but only on the decoding itself
        # however, in practice, all messages should be decodable, and if not this is a serious issue; don t be silent
        try:
            crrt_kind, crrt_meta, crrt_list_packets = decode_message(
                crrt_data.Payload,
                print_decoded=False,
                dict_wave_packet_params=modified_wave_packet_properties)

        except AssertionError as e:
            logger.warning(
                f"attempt to decode entry at index {pd_index}, Payload equal to: {crrt_data.Payload} failed with exception:\n{e}"
            )
            continue

        number_valid_entries += 1

        # a GNSS packet may contain several data entries; split it here for simplicity
        if crrt_kind == "G":
            for crrt_fix in crrt_list_packets:
                if crrt_start_time is None or crrt_fix.datetime_fix > crrt_start_time:
                    crrt_parsed = ParsedIridiumMessage(
                        device_from=crrt_data.Device,
                        kind=crrt_kind,
                        meta=crrt_meta,
                        data=crrt_fix,
                    )

                    append_dict_with_entry(dict_entries, crrt_parsed)

                else:
                    logger.info(f"buoy {crrt_instrument}: ignore fix {crrt_fix}, since before {crrt_start_time}")

        # other packets contain a single entry: add as is
        else:
            if crrt_kind == "Y" and not frequencies_set:
                frequencies_set = True
                frequencies = crrt_list_packets[0].list_frequencies

            if crrt_start_time is None or crrt_list_packets[0].datetime_fix > crrt_start_time:
                crrt_parsed = ParsedIridiumMessage(
                    device_from=crrt_data.Device,
                    kind=crrt_kind,
                    meta=crrt_meta,
                    data=crrt_list_packets[0],
                )

                append_dict_with_entry(dict_entries, crrt_parsed)

            else:
                logger.info(f"buoy {crrt_instrument}: ignore spectrum with timestamp {crrt_list_packets[0].datetime_fix}, since before {crrt_start_time}")

    if number_valid_entries == 0:
        logger.warning(
            "got no valid decoded payload in the whole csv file; are you sure this is an OMB csv iridium file?"
        )

    ########################################
    # turn into a trajan compatible format

    # determine the size for trajectory, obs, imu_obs, frequencies
    trajectory = len(dict_entries)

    obs_gnss = max(
        [len(dict_entries[crrt_instr]["G"]) for crrt_instr in dict_entries])

    obs_waves_imu = max(
        [len(dict_entries[crrt_instr]["Y"]) for crrt_instr in dict_entries])

    frequencies_waves_imu = nbr_bins_waves

    list_instruments = sorted(list(dict_entries.keys()))

    empty_time = np.full((trajectory, obs_gnss),
                         np.datetime64('nat'),
                         dtype='datetime64[ns]')
    empty_time_waves_imu = np.full((trajectory, obs_waves_imu),
                                   np.datetime64('nat'),
                                   dtype='datetime64[ns]')

    # create and fill the xarray dataset
    xr_result = xr.Dataset(
        {
            # meta vars
            #
            'trajectory':
            xr.DataArray(data=list_instruments,
                         dims=['trajectory'],
                         attrs={
                             "cf_role": "trajectory_id",
                             "standard_name": "platform_id",
                         }).astype(str),
            #
            'frequencies_waves_imu':
            xr.DataArray(data=frequencies,
                         dims=["frequencies_waves_imu"],
                         attrs={
                             "_FillValue": "NaN",
                             "unit": "Hz",
                         }),
            #
            # gnss position vars
            #
            'time':
            xr.DataArray(dims=["trajectory", "obs"],
                         data=empty_time,
                         attrs={
                             "standard_name": "time",
                         }),
            #
            'lat':
            xr.DataArray(dims=["trajectory", "obs"],
                         data=np.nan * np.ones((trajectory, obs_gnss)),
                         attrs={
                             "_FillValue": "NaN",
                             "standard_name": "latitude",
                             "unit": "degree_north",
                         }),
            #
            'lon':
            xr.DataArray(dims=["trajectory", "obs"],
                         data=np.nan * np.ones((trajectory, obs_gnss)),
                         attrs={
                             "_FillValue": "NaN",
                             "standard_name": "longitude",
                             "unit": "degree_east",
                         }),
            #
            # imu waves vars
            #
            'time_waves_imu':
            xr.DataArray(dims=["trajectory", "obs_waves_imu"],
                         data=empty_time_waves_imu,
                         attrs={
                             "standard_name": "time",
                         }),
            #
            'accel_energy_spectrum':
            xr.DataArray(
                dims=["trajectory", "obs_waves_imu", "frequencies_waves_imu"],
                data=np.nan * np.ones(
                    (trajectory, obs_waves_imu, frequencies_waves_imu)),
                attrs={
                    "_FillValue": "NaN",
                }),
            #
            'elevation_energy_spectrum':
            xr.DataArray(
                dims=["trajectory", "obs_waves_imu", "frequencies_waves_imu"],
                data=np.nan * np.ones(
                    (trajectory, obs_waves_imu, frequencies_waves_imu)),
                attrs={
                    "_FillValue": "NaN",
                }),
            #
            'processed_elevation_energy_spectrum':
            xr.DataArray(
                dims=["trajectory", "obs_waves_imu", "frequencies_waves_imu"],
                data=np.nan * np.ones(
                    (trajectory, obs_waves_imu, frequencies_waves_imu)),
                attrs={
                    "_FillValue": "NaN",
                }),
            #
            'pcutoff':
            xr.DataArray(dims=["trajectory", "obs_waves_imu"],
                         data=np.nan * np.ones((trajectory, obs_waves_imu)),
                         attrs={
                             "_FillValue": "NaN",
                         }),
            #
            'pHs0':
            xr.DataArray(
                dims=["trajectory", "obs_waves_imu"],
                data=np.nan * np.ones((trajectory, obs_waves_imu)),
                attrs={
                    "_FillValue":
                    "NaN",
                    "definition":
                    "4 * math.sqrt(m0) of low freq cutoff elevation spectrum"
                }),
            #
            'pT02':
            xr.DataArray(
                dims=["trajectory", "obs_waves_imu"],
                data=np.nan * np.ones((trajectory, obs_waves_imu)),
                attrs={
                    "_FillValue":
                    "NaN",
                    "definition":
                    "math.sqrt(m0 / m2) of low freq cutoff elevation spectrum"
                }),
            #
            'pT24':
            xr.DataArray(
                dims=["trajectory", "obs_waves_imu"],
                data=np.nan * np.ones((trajectory, obs_waves_imu)),
                attrs={
                    "_FillValue":
                    "NaN",
                    "definition":
                    "math.sqrt(m2 / m4) of low freq cutoff elevation spectrum"
                }),
            #
            'Hs0':
            xr.DataArray(dims=["trajectory", "obs_waves_imu"],
                         data=np.nan * np.ones((trajectory, obs_waves_imu)),
                         attrs={
                             "_FillValue":
                             "NaN",
                             "definition":
                             "4 * math.sqrt(m0) of full elevation spectrum"
                         }),
            #
            'T02':
            xr.DataArray(dims=["trajectory", "obs_waves_imu"],
                         data=np.nan * np.ones((trajectory, obs_waves_imu)),
                         attrs={
                             "_FillValue":
                             "NaN",
                             "definition":
                             "math.sqrt(m0 / m2) of full elevation spectrum"
                         }),
            #
            'T24':
            xr.DataArray(dims=["trajectory", "obs_waves_imu"],
                         data=np.nan * np.ones((trajectory, obs_waves_imu)),
                         attrs={
                             "_FillValue":
                             "NaN",
                             "definition":
                             "math.sqrt(m2 / m4) of full elevation spectrum"
                         }),
        }, )

    # actually fill the data
    for crrt_instrument_idx, crrt_instrument in enumerate(list_instruments):
        ####################
        # gnss position data

        list_time = [
            crrt_packet.data.datetime_posix
            for crrt_packet in dict_entries[crrt_instrument]["G"]
        ]
        list_lat = [
            crrt_packet.data.latitude
            for crrt_packet in dict_entries[crrt_instrument]["G"]
        ]
        list_lon = [
            crrt_packet.data.longitude
            for crrt_packet in dict_entries[crrt_instrument]["G"]
        ]

        # sort in time
        argsort_time_idx = list(np.argsort(np.array(list_time)))
        list_time = [list_time[i] for i in argsort_time_idx]
        list_lat = [list_lat[i] for i in argsort_time_idx]
        list_lon = [list_lon[i] for i in argsort_time_idx]

        # reject outliers
        logger.debug("start applying sliding_filter_nsigma")
        np_latitude = sliding_filter_nsigma(np.array(list_lat))
        np_longitude = sliding_filter_nsigma(np.array(list_lon))
        logger.debug("done applying sliding_filter_nsigma")

        xr_result["time"][crrt_instrument_idx,
                          0:len(list_time)] = pd.to_datetime(list_time,
                                                             utc=True,
                                                             unit='s')
        xr_result["lat"][crrt_instrument_idx, 0:len(list_lat)] = np_latitude
        xr_result["lon"][crrt_instrument_idx, 0:len(list_lon)] = np_longitude

        ####################
        # wave data

        list_parsed_waves_messages = dict_entries[crrt_instrument]["Y"]

        crrt_list_times_waves = [
            crrt_wave_data.data.datetime_posix
            for crrt_wave_data in list_parsed_waves_messages
        ]
        argsort_time_idx = list(np.argsort(np.array(crrt_list_times_waves)))
        list_parsed_waves_messages = [
            list_parsed_waves_messages[i] for i in argsort_time_idx
        ]

        for crrt_wave_idx, crrt_wave_data in enumerate(
                list_parsed_waves_messages):
            xr_result["time_waves_imu"][crrt_instrument_idx, crrt_wave_idx] = \
                pd.to_datetime(crrt_wave_data.data.datetime_posix, utc=True, unit='s')

            xr_result["pcutoff"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.low_frequency_index_cutoff

            xr_result["accel_energy_spectrum"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.list_acceleration_energies

            xr_result["elevation_energy_spectrum"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.list_elevation_energies

            xr_result["processed_elevation_energy_spectrum"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.processed_list_elevation_energies

            xr_result["pHs0"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.processed_Hs

            xr_result["pT02"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.processed_Tz

            xr_result["pT24"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.processed_Tc

            xr_result["Hs0"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.Hs

            xr_result["T02"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.Tz

            xr_result["T24"][crrt_instrument_idx, crrt_wave_idx] = \
                crrt_wave_data.data.Tc

    xr_result = xr_result.traj.assign_cf_attrs(
        creator_name="XX:TODO",
        creator_email="XX:TODO",
        title="XX:TODO",
        summary="XX:TODO",
        history=
        "created with trajan.reader.omb from a Rock7 Iridium CSV file of OMB transmissions"
    )

    return xr_result

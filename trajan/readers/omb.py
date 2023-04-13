
from pathlib import Path
import xarray as xr
import logging
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass

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
                crrt_parsed = ParsedIridiumMessage(
                    device_from = crrt_data.Device,
                    kind = crrt_kind,
                    meta = crrt_meta,
                    data = crrt_list_packets,
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

    obs = max([len(dict_entries[crrt_instr]["G"]) for crrt_instr in dict_entries])

    imu_obs = max([len(dict_entries[crrt_instr]["Y"]) for crrt_instr in dict_entries])

    frequencies = _BD_YWAVE_NBR_BINS

    # create the xarray dataset

    # fill the xarray dataset



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("start main")

    path_to_test_data = Path.cwd().parent.parent / "tests" / "test_data" / "csv" / "omb1.csv"
    read_omb_csv(path_to_test_data)

    print("done main")

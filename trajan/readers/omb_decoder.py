"""
The OMB data decoder; slightly curated from:

https://github.com/jerabaul29/OpenMetBuoy-v2021a/tree/main/legacy_firmware/decoder

For issues related to the decoder, please discuss on the OMB repository:

https://github.com/jerabaul29/OpenMetBuoy-v2021a/issues
"""

import binascii
import struct
import datetime
from dataclasses import dataclass
import math
import numpy as np
import scipy.signal as signal
import logging

logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------
# a few module constants

_BD_VERSION_NBR = "2.1"

####################
# derived properties of the GNSS packets
# 14; 1 byte start, posix, 2 longs, 1 byte end
_BD_GNSS_PACKET_LENGTH = 14

####################
# properties of the wave spectra packets with 2048 FFT_LEN
_BD_YWAVE_PACKET_MIN_BIN = 9
_BD_YWAVE_PACKET_MAX_BIN = 64
_BD_YWAVE_PACKET_SCALER = 65000
_BD_YWAVE_PACKET_SAMPLING_FREQ_HZ = 10.0
_BD_YWAVE_PACKET_NBR_SAMPLES_PER_SEGMENT = 2**11

# derived properties of the wave packer
_BD_YWAVE_NBR_BINS = _BD_YWAVE_PACKET_MAX_BIN - _BD_YWAVE_PACKET_MIN_BIN
_BD_YWAVE_PACKET_FRQ_RES = _BD_YWAVE_PACKET_SAMPLING_FREQ_HZ / _BD_YWAVE_PACKET_NBR_SAMPLES_PER_SEGMENT

# 138 in total length; 1 byte start, posix, int, 4 floats, uint36 array, 2 bytes for alignment, 1 byte end
_BD_YWAVE_PACKET_LENGTH = \
    1 + 4 + 4 + 4*4 + _BD_YWAVE_NBR_BINS * 2 + 2 + 1
LENGTH_FROM_SERIAL_OUTPUT = 138
assert _BD_YWAVE_PACKET_LENGTH == LENGTH_FROM_SERIAL_OUTPUT, "the arduino printout indicates that wave packets have length {}".format(
    LENGTH_FROM_SERIAL_OUTPUT)

####################
# properties of the thermistors packets
_BD_THERM_MSG_FIXED_LENGTH = 3  # start byte, byte metadata nbr packets, byte end
_BD_THERM_MSG_NBR_THERMISTORS = 6
_BD_THERM_PACKET_NBR_BYTES_PER_THERMISTOR = 3
_BD_THERM_PACKET_LENGTH = 1 + 1 * 4 + _BD_THERM_PACKET_NBR_BYTES_PER_THERMISTOR * _BD_THERM_MSG_NBR_THERMISTORS * 1 + 6 * 1
# these are from the params of the tracker
_BD_THERM_ROLL_FLOAT_TO_INT8_FACTOR = 0.7
_BD_THERM_PITCH_FLOAT_TO_INT8_FACTOR = 1.4
# factor for converting bin 12 bits signed int to temperature; from thermistor datasheet
_BD_THERM_12BITS_TO_FLOAT_TEMPERATURE_FACTOR = 1.0 / 16.0

#--------------------------------------------------------------------------------
# misc


def get_version():
    return _BD_VERSION_NBR


#--------------------------------------------------------------------------------
# helper functions for unpacking binary data


def byte_to_char(crrt_byte):
    return (chr(crrt_byte))


def one_byte_to_int(crrt_byte):
    return (struct.unpack('B', bytes(crrt_byte))[0])


def one_byte_to_signed_int(crrt_byte):
    return (struct.unpack('b', bytes(crrt_byte))[0])


def four_bytes_to_long(crrt_four_bytes):
    res = struct.unpack('<l', bytes(crrt_four_bytes))
    return res[0]


def four_bytes_to_int(crrt_four_bytes):
    res = struct.unpack('<i', bytes(crrt_four_bytes))
    return res[0]


def four_bytes_to_unsignedint(crrt_four_bytes):
    res = struct.unpack('<I', bytes(crrt_four_bytes))
    return res[0]


def four_bytes_to_float(crrt_four_bytes):
    res = struct.unpack('<f', bytes(crrt_four_bytes))
    return res[0]


#--------------------------------------------------------------------------------
# custom data classes to store data packets


@dataclass
class Spectral_Moments:
    m0: float
    m2: float
    m4: float


@dataclass
class GNSS_Packet:
    datetime_fix: datetime.datetime
    datetime_posix: int
    latitude: float
    longitude: float
    is_valid: bool


@dataclass
class GNSS_Metadata:
    nbr_gnss_fixes: int


@dataclass
class Waves_Packet:
    # the decoded part: this is just decoding / textbook operations
    datetime_fix: datetime.datetime
    datetime_posix: int
    spectrum_number: int
    Hs: float
    Tz: float
    Tc: float
    _array_max_value: float
    _array_uint16: float
    list_frequencies: list
    list_acceleration_energies: list
    frequency_resolution: float
    list_elevation_energies: list
    wave_spectral_moments: Spectral_Moments
    is_valid: bool
    # the processed part: this relies on some more advanced operations
    processed_list_frequencies: list
    processed_list_elevation_energies: list
    processed_wave_spectral_moments: Spectral_Moments
    processed_Hs: float
    processed_Tz: float
    processed_Tc: float
    low_frequency_index_cutoff: int
    # processed_quality_index: str  # todo: add a quality explanation string


@dataclass
class Waves_Metadata:
    None


@dataclass
class Thermistors_Reading:
    mean_temperature: float
    range_temperature: float
    probe_id: int


@dataclass
class Thermistors_Packet:
    datetime_packet: datetime.datetime
    datetime_posix: int
    thermistors_readings: list
    mean_pitch: float
    min_pitch: float
    max_pitch: float
    mean_roll: float
    min_roll: float
    max_roll: float


@dataclass
class Thermistors_Metadata:
    nbr_thermistors_measurements: int


# --------------------------------------------------------------------------------
# data quality and processing utils


def find_low_frequency_cutoff(list_frequencies, list_elevation_energies):
    """Find the low frequency cutoff to avoid double integration noise contamination
    for IMU measurements of waves, similar to what is discussed in Fig. 7 of
    https://www.mdpi.com/2076-3263/12/3/110 .

    Inputs:
        list_frequencies: the list of frequencies at which spectrum is provided
        list_elevation_energies: the wave spectrum elevation energies at the corresponding
            frequencies
    Output:
        index_low_frequency_cutoff: the index to use to cut off low frequency double
            integration noise contamination; i.e., use only indexes > to the output
            to consider the valid part of the spectrum.
    """

    assert isinstance(list_frequencies, list)
    assert isinstance(list_elevation_energies, list)
    assert len(list_frequencies) == len(list_elevation_energies)

    # findpeaks:
    # we look for minima (peaks looks for maxima, so -)
    # we want peaks that are local peaks in their neighborhood and not just "super local minima": require distance = 3
    # we want only peaks that are clear enough: normalize the spectrum so that the maximum is always 1.0, and want a prominence of at least 0.05
    normalized_spectrum = -np.array(list_elevation_energies) / np.max(
        list_elevation_energies)
    peaks_output = signal.find_peaks(normalized_spectrum,
                                     distance=3,
                                     prominence=0.05)

    peaks = list(peaks_output[0])
    if len(peaks) == 0:
        peaks = [0]

    # isolate the peak we want
    # we are only interested in the first minimum, and it has to be low enough that it does correspond to a low freq. threshold
    first_peak = peaks[0]
    if list_frequencies[
            first_peak] > 0.10:  # the 0.1 value is somewhat arbitrary, but in practice we always have some energy at least there or before
        first_peak = 0  # we keep all indexes if there was no clear minimum on the left (ie no low energy noise)

    # we do not want to flag out valid parts of the spectrum when the spectrum is "really clean"
    # in cases where the spectrum is "really clean", can happen that the first minimum is a local minimum after the first valid peak
    # detect these cases and set the full spectrum as valid then
    if (list_elevation_energies[first_peak] >
        ((list_elevation_energies[0] + list_elevation_energies[1]) / 2.0)):
        first_peak = 0

    index_low_frequency_cutoff = first_peak

    return index_low_frequency_cutoff


# --------------------------------------------------------------------------------
# packets and messages decoding


def hex_to_bin_message(hex_string_message, print_info=False):
    bin_msg = binascii.unhexlify(hex_string_message)
    return bin_msg


def message_kind(bin_msg):
    first_char = byte_to_char(bin_msg[0])
    valid_first_chars = ["G", "Y", "T"]
    assert first_char in valid_first_chars, "unknown first_char message kind: got {}, valids are {}".format(
        first_char, valid_first_chars)
    return first_char


def decode_gnss_packet(bin_packet,
                       print_decoded=False,
                       print_debug_information=False):
    assert len(
        bin_packet
    ) == _BD_GNSS_PACKET_LENGTH, "GNSS packets with start and end byte have 14 bytes, got {} bytes".format(
        len(bin_packet))

    char_first_byte = byte_to_char(bin_packet[0])

    assert char_first_byte == 'F', "GNSS packets must start with a 'F', got {}".format(
        char_first_byte)

    posix_timestamp_fix = four_bytes_to_long(bin_packet[1:5])
    # print(posix_timestamp_fix)
    datetime_fix = datetime.datetime.utcfromtimestamp(posix_timestamp_fix)

    latitude_long = four_bytes_to_long(bin_packet[5:9])
    latitude = latitude_long / 1.0e7

    longitude_long = four_bytes_to_long(bin_packet[9:13])
    longitude = longitude_long / 1.0e7

    if print_decoded:
        print("-------------------- decoded GNSS packet ---------------------")
        print("fix at posix {}, i.e. {}".format(posix_timestamp_fix,
                                                datetime_fix))
        print("latitude {}, i.e. {}".format(latitude_long, latitude))
        print("longitude {}, i.e. {}".format(longitude_long, longitude))
        print("--------------------------------------------------------------")

    char_next_byte = byte_to_char(bin_packet[13])

    assert char_next_byte == 'E' or char_next_byte == 'F', "either end ('E') or fix ('F') expected at the end, got {}".format(
        char_next_byte)

    decoded_packet = GNSS_Packet(datetime_fix=datetime_fix,
                                 datetime_posix=posix_timestamp_fix,
                                 latitude=latitude,
                                 longitude=longitude,
                                 is_valid=True)

    return decoded_packet


def decode_gnss_message(bin_msg,
                        print_decoded=True,
                        print_debug_information=False):
    if print_decoded:
        print(
            "----------------------- START DECODE GNSS MESSAGE -----------------------"
        )

    assert message_kind(bin_msg) == 'G'
    expected_message_length = int(1 +
                                  (len(bin_msg) - 2 - _BD_GNSS_PACKET_LENGTH) /
                                  (_BD_GNSS_PACKET_LENGTH - 1))

    if print_decoded:
        print("expected number of packets based on message length: {}".format(
            expected_message_length))

    nbr_gnss_fixes = one_byte_to_int(bin_msg[1:2])
    message_metadata = GNSS_Metadata(nbr_gnss_fixes=nbr_gnss_fixes)

    if print_decoded:
        print("number of fixes since boot at message creation: {}".format(
            nbr_gnss_fixes))

    crrt_packet_start = 2
    list_decoded_packets = []

    while True:
        crrt_byte_start = byte_to_char(bin_msg[crrt_packet_start])
        assert crrt_byte_start == "F"

        crrt_decoded_packet = decode_gnss_packet(
            bin_msg[crrt_packet_start:crrt_packet_start +
                    _BD_GNSS_PACKET_LENGTH],
            print_decoded=print_decoded,
            print_debug_information=print_debug_information)
        list_decoded_packets.append(crrt_decoded_packet)

        trailing_char = byte_to_char(bin_msg[crrt_packet_start +
                                             _BD_GNSS_PACKET_LENGTH - 1])
        assert trailing_char in ["E", "F"]
        if trailing_char == "E":
            break
        else:
            crrt_packet_start += _BD_GNSS_PACKET_LENGTH - 1

    assert expected_message_length == len(list_decoded_packets)

    if print_decoded:
        print(
            "----------------------- DONE DECODE GNSS MESSAGE -----------------------"
        )

    return message_metadata, list_decoded_packets


def decode_ywave_packet(bin_packet,
                        print_decoded=False,
                        print_debug_information=False):
    assert len(bin_packet) == _BD_YWAVE_PACKET_LENGTH

    char_first_byte = byte_to_char(bin_packet[0])
    assert char_first_byte == "Y"

    char_last_byte = byte_to_char(bin_packet[-1])
    assert char_last_byte == "E"

    crrt_start_data_field = 1

    posix_timestamp = four_bytes_to_long(
        bin_packet[crrt_start_data_field:crrt_start_data_field + 4])
    crrt_start_data_field += 4
    datetime_packet = datetime.datetime.utcfromtimestamp(posix_timestamp)

    spectrum_number = four_bytes_to_int(
        bin_packet[crrt_start_data_field:crrt_start_data_field + 4])
    crrt_start_data_field += 4

    Hs = four_bytes_to_float(
        bin_packet[crrt_start_data_field:crrt_start_data_field + 4])
    crrt_start_data_field += 4

    Tz = 1.0 / four_bytes_to_float(
        bin_packet[crrt_start_data_field:crrt_start_data_field + 4])
    crrt_start_data_field += 4

    Tc = 1.0 / four_bytes_to_float(
        bin_packet[crrt_start_data_field:crrt_start_data_field + 4])
    crrt_start_data_field += 4

    _array_max_value = four_bytes_to_float(
        bin_packet[crrt_start_data_field:crrt_start_data_field + 4])
    crrt_start_data_field += 4

    nbr_bytes_uint16_array = _BD_YWAVE_NBR_BINS * 2
    _array_uint16 = struct.unpack(
        '<' + _BD_YWAVE_NBR_BINS * "H",
        bin_packet[crrt_start_data_field:crrt_start_data_field +
                   nbr_bytes_uint16_array])
    crrt_start_data_field += nbr_bytes_uint16_array

    assert crrt_start_data_field + 1 + 2 == _BD_YWAVE_PACKET_LENGTH  # the +2 is due to struct alignment issues

    if Hs > 1e-5:
        is_valid = True
    else:
        is_valid = False
        logger.warning(f"got a wave spectrum with corresponding Hs<1e-5; this likely indicates an instrument that landed onshore or an IMU malfunction; setting corresponding spectrum is_valid to False")

    list_frequencies = []
    list_acceleration_energies = []

    for ind, crrt_uint16 in enumerate(_array_uint16):
        list_frequencies.append(
            (_BD_YWAVE_PACKET_MIN_BIN + ind) * _BD_YWAVE_PACKET_FRQ_RES)
        list_acceleration_energies.append(crrt_uint16 * _array_max_value /
                                          _BD_YWAVE_PACKET_SCALER)

    list_omega = [2.0 * math.pi * crrt_freq for crrt_freq in list_frequencies]
    list_omega_4 = [math.pow(crrt_omega, 4) for crrt_omega in list_omega]
    list_elevation_energies = [
        crrt_acceleration_energy / crrt_omega_4
        for (crrt_acceleration_energy,
             crrt_omega_4) in zip(list_acceleration_energies, list_omega_4)
    ]

    def compute_spectral_moment(list_frequencies, list_elevation_energies,
                                order):
        list_to_integrate = [
            math.pow(crrt_freq, order) * crrt_energy
            for (crrt_freq,
                 crrt_energy) in zip(list_frequencies, list_elevation_energies)
        ]

        moment = np.trapz(list_to_integrate, list_frequencies)

        return moment

    m0 = compute_spectral_moment(list_frequencies, list_elevation_energies, 0)
    m2 = compute_spectral_moment(list_frequencies, list_elevation_energies, 2)
    m4 = compute_spectral_moment(list_frequencies, list_elevation_energies, 4)

    spectral_moments = Spectral_Moments(m0, m2, m4)

    # add the post processed part

    low_frequency_index_cutoff = find_low_frequency_cutoff(
        list_frequencies, list_elevation_energies)

    processed_list_frequencies = list_frequencies[low_frequency_index_cutoff:]
    processed_list_elevation_energies = list_elevation_energies[
        low_frequency_index_cutoff:]

    processed_m0 = compute_spectral_moment(processed_list_frequencies,
                                           processed_list_elevation_energies,
                                           0)
    processed_m2 = compute_spectral_moment(processed_list_frequencies,
                                           processed_list_elevation_energies,
                                           2)
    processed_m4 = compute_spectral_moment(processed_list_frequencies,
                                           processed_list_elevation_energies,
                                           4)

    processed_wave_spectral_moments = Spectral_Moments(
        processed_m0,
        processed_m2,
        processed_m4,
    )

    processed_Hs = 4 * math.sqrt(processed_m0)
    processed_Tz = 1.0 / math.sqrt(processed_m2 / processed_m0)
    processed_Tc = 1.0 / math.sqrt(processed_m4 / processed_m2)

    processed_list_frequencies = list_frequencies
    processed_list_elevation_energies = low_frequency_index_cutoff * [
        math.nan
    ] + processed_list_elevation_energies

    decoded_packet = Waves_Packet(
        datetime_packet,
        posix_timestamp,
        spectrum_number,
        Hs,
        Tz,
        Tc,
        _array_max_value,
        _array_uint16,
        list_frequencies,
        list_acceleration_energies,
        _BD_YWAVE_PACKET_FRQ_RES,
        list_elevation_energies,
        spectral_moments,
        is_valid,
        processed_list_frequencies,
        processed_list_elevation_energies,
        processed_wave_spectral_moments,
        processed_Hs,
        processed_Tz,
        processed_Tc,
        low_frequency_index_cutoff,
    )

    return decoded_packet


def decode_ywave_message(bin_msg,
                         print_decoded=True,
                         print_debug_information=False):
    if print_decoded:
        print(
            "----------------------- START DECODE YWAVES MESSAGE -----------------------"
        )

    assert message_kind(bin_msg) == "Y"
    assert byte_to_char(bin_msg[_BD_YWAVE_PACKET_LENGTH - 1]) == "E"

    message_metadata = Waves_Metadata()

    list_decoded_packets = []
    crrt_packet = decode_ywave_packet(
        bin_msg,
        print_decoded=print_decoded,
        print_debug_information=print_debug_information)
    list_decoded_packets.append(crrt_packet)

    if print_decoded:
        print(
            "----------------------- DONE DECODE GNSS MESSAGE -----------------------"
        )

    return message_metadata, list_decoded_packets


def decode_thermistor_reading(crrt_thermistor_bin,
                              print_debug_information=False):

    # quite a bit of tweaking...
    # this is the inverse operation of the binary encoding done in the thermistor manager C++ code

    # TODO: actually, this can be made with clear syntax a la n & 0xffffffff too, fixme

    id_6_bits = one_byte_to_int(crrt_thermistor_bin[0:1]) // 4

    reading_2_higher_bits = one_byte_to_int(crrt_thermistor_bin[0:1]) % 4
    reading_2_higher_bits_lower = reading_2_higher_bits % 2
    reading_2_higher_bits_higher = (reading_2_higher_bits -
                                    reading_2_higher_bits_lower) // 2
    reading_8_middle_bits = one_byte_to_int(crrt_thermistor_bin[1:2])
    reading_2_lower_bits = one_byte_to_int(crrt_thermistor_bin[2:3]) // 64

    reading_reconstructed_bin = reading_2_lower_bits + (
        2**2) * reading_8_middle_bits + (2**10) * reading_2_higher_bits_lower
    if reading_2_higher_bits_higher:
        reading_reconstructed_bin = reading_reconstructed_bin - 2**11 - 1

    range_6_bits_bin = one_byte_to_int(crrt_thermistor_bin[2:3]) % 64

    reading_reconstructed = reading_reconstructed_bin * _BD_THERM_12BITS_TO_FLOAT_TEMPERATURE_FACTOR
    range_temperature = range_6_bits_bin * _BD_THERM_12BITS_TO_FLOAT_TEMPERATURE_FACTOR

    crrt_thermistor_reading = Thermistors_Reading(
        mean_temperature=reading_reconstructed,
        range_temperature=range_temperature,
        probe_id=id_6_bits)

    return crrt_thermistor_reading


def decode_thermistors_packet(bin_packet,
                              print_decoded=False,
                              print_debug_information=False):
    if print_debug_information:
        print("----- START DECODE THERM PACKET -----")

    assert len(bin_packet) == _BD_THERM_PACKET_LENGTH

    char_first_byte = byte_to_char(bin_packet[0])
    assert char_first_byte == "P"

    crrt_start_field = 1

    posix_timestamp = four_bytes_to_long(
        bin_packet[crrt_start_field:crrt_start_field + 4])
    datetime_packet = datetime.datetime.utcfromtimestamp(posix_timestamp)
    crrt_start_field += 4

    list_thermistors_readings = []

    for crrt_thermistor in range(_BD_THERM_MSG_NBR_THERMISTORS):
        crrt_thermistor_bin = bin_packet[crrt_start_field:crrt_start_field + 3]
        crrt_start_field += 3
        crrt_thermistor_reading = decode_thermistor_reading(
            crrt_thermistor_bin,
            print_debug_information=print_debug_information)
        assert isinstance(crrt_thermistor_reading, Thermistors_Reading)
        list_thermistors_readings.append(crrt_thermistor_reading)

    mean_pitch_bin = one_byte_to_signed_int(
        bin_packet[crrt_start_field:crrt_start_field + 1])
    crrt_start_field += 1

    mean_roll_bin = one_byte_to_signed_int(
        bin_packet[crrt_start_field:crrt_start_field + 1])
    crrt_start_field += 1

    min_pitch_bin = one_byte_to_signed_int(
        bin_packet[crrt_start_field:crrt_start_field + 1])
    crrt_start_field += 1

    max_pitch_bin = one_byte_to_signed_int(
        bin_packet[crrt_start_field:crrt_start_field + 1])
    crrt_start_field += 1

    min_roll_bin = one_byte_to_signed_int(
        bin_packet[crrt_start_field:crrt_start_field + 1])
    crrt_start_field += 1

    max_roll_bin = one_byte_to_signed_int(
        bin_packet[crrt_start_field:crrt_start_field + 1])
    crrt_start_field += 1

    assert crrt_start_field == _BD_THERM_PACKET_LENGTH

    crrt_thermistor_packet = Thermistors_Packet(
        datetime_packet=datetime_packet,
        datetime_posix=posix_timestamp,
        thermistors_readings=list_thermistors_readings,
        mean_pitch=mean_pitch_bin / _BD_THERM_PITCH_FLOAT_TO_INT8_FACTOR,
        min_pitch=min_pitch_bin / _BD_THERM_PITCH_FLOAT_TO_INT8_FACTOR,
        max_pitch=max_pitch_bin / _BD_THERM_PITCH_FLOAT_TO_INT8_FACTOR,
        mean_roll=mean_roll_bin / _BD_THERM_ROLL_FLOAT_TO_INT8_FACTOR,
        min_roll=min_roll_bin / _BD_THERM_ROLL_FLOAT_TO_INT8_FACTOR,
        max_roll=max_roll_bin / _BD_THERM_ROLL_FLOAT_TO_INT8_FACTOR)

    if print_debug_information:
        print("----- DONE DECODE THERM PACKET -----")

    return crrt_thermistor_packet


def decode_thermistors_message(bin_msg,
                               print_decoded=False,
                               print_debug_information=False):
    if print_decoded:
        print(
            "----------------------- START DECODE THERMISTORS MESSAGE -----------------------"
        )

    assert message_kind(bin_msg) == "T"
    assert byte_to_char(bin_msg[-1]) == "E"

    if (print_debug_information):
        print("received message of length: {}".format(len(bin_msg)))

    expected_message_length = int(
        (len(bin_msg) - _BD_THERM_MSG_FIXED_LENGTH) / _BD_THERM_PACKET_LENGTH)
    assert expected_message_length * _BD_THERM_PACKET_LENGTH + _BD_THERM_MSG_FIXED_LENGTH == len(
        bin_msg)

    nbr_thermistors_measurements = one_byte_to_int(bin_msg[1:2])
    message_metadata = Thermistors_Metadata(
        nbr_thermistors_measurements=nbr_thermistors_measurements)

    if print_decoded:
        print(message_metadata)

    crrt_packet_start = 2
    list_decoded_packets = []

    while True:
        if print_decoded:
            print("----- START PACKET -----")

        crrt_byte_start = byte_to_char(bin_msg[crrt_packet_start])
        assert crrt_byte_start == "P"

        # decode
        crrt_decoded_packet = decode_thermistors_packet(
            bin_msg[crrt_packet_start:crrt_packet_start +
                    _BD_THERM_PACKET_LENGTH],
            print_decoded=print_decoded,
            print_debug_information=print_debug_information)
        list_decoded_packets.append(crrt_decoded_packet)

        trailing_char = byte_to_char(bin_msg[crrt_packet_start +
                                             _BD_THERM_PACKET_LENGTH])
        assert trailing_char in ["P", "E"]
        if trailing_char == "E":
            break
        else:
            crrt_packet_start += _BD_THERM_PACKET_LENGTH

        if print_decoded:
            print("----- END PACKET -----")

    assert expected_message_length == len(list_decoded_packets)

    if print_decoded:
        print(
            "----------------------- DONE DECODE THERMISTORS MESSAGE -----------------------"
        )

    return message_metadata, list_decoded_packets


def decode_message(hex_string_message,
                   print_decoded=True,
                   print_debug_information=False,
                   dict_wave_packet_params=None):
    # NOTE: this is pushing the decoder further than it was initially thought...
    # some users change the default size of the transmitted spectra; as the logics of the packets formats were initially done
    # as global variables, this implies doing some hacking for allowing to change this on the fly
    # the on the fly globals editing below is dangerous and bad; that would need refactor in the future
    # this decoder has outgrown itself :)
    if dict_wave_packet_params is not None:
        global _BD_YWAVE_PACKET_MIN_BIN
        global _BD_YWAVE_PACKET_MAX_BIN
        global _BD_YWAVE_NBR_BINS
        global _BD_YWAVE_PACKET_LENGTH
        global LENGTH_FROM_SERIAL_OUTPUT

        old_BD_YWAVE_PACKET_MIN_BIN = _BD_YWAVE_PACKET_MIN_BIN
        old_BD_YWAVE_PACKET_MAX_BIN = _BD_YWAVE_PACKET_MAX_BIN
        old_BD_YWAVE_NBR_BINS = _BD_YWAVE_NBR_BINS
        old_BD_YWAVE_PACKET_LENGTH = _BD_YWAVE_PACKET_LENGTH
        oldLENGTH_FROM_SERIAL_OUTPUT = LENGTH_FROM_SERIAL_OUTPUT

        _BD_YWAVE_PACKET_MIN_BIN = dict_wave_packet_params[
            "_BD_YWAVE_PACKET_MIN_BIN"]
        _BD_YWAVE_PACKET_MAX_BIN = dict_wave_packet_params[
            "_BD_YWAVE_PACKET_MAX_BIN"]
        _BD_YWAVE_NBR_BINS = _BD_YWAVE_PACKET_MAX_BIN - _BD_YWAVE_PACKET_MIN_BIN
        _BD_YWAVE_PACKET_LENGTH = \
            1 + 4 + 4 + 4*4 + _BD_YWAVE_NBR_BINS * 2 + 2 + 1
        LENGTH_FROM_SERIAL_OUTPUT = dict_wave_packet_params[
            "LENGTH_FROM_SERIAL_OUTPUT"]

        assert _BD_YWAVE_PACKET_LENGTH == LENGTH_FROM_SERIAL_OUTPUT, "the arduino printout indicates that wave packets have length {}".format(
            LENGTH_FROM_SERIAL_OUTPUT)

    bin_msg = hex_to_bin_message(hex_string_message)

    kind = message_kind(bin_msg)

    if kind == "G":
        message_metadata, list_decoded_packets = decode_gnss_message(
            bin_msg,
            print_decoded=print_decoded,
            print_debug_information=print_debug_information)
    elif kind == "Y":
        message_metadata, list_decoded_packets = decode_ywave_message(
            bin_msg,
            print_decoded=print_decoded,
            print_debug_information=print_debug_information)
    elif kind == "T":
        message_metadata, list_decoded_packets = decode_thermistors_message(
            bin_msg,
            print_decoded=print_decoded,
            print_debug_information=print_debug_information)
    else:
        raise RuntimeError("Unknown message kind: {}".format(kind))

    if dict_wave_packet_params is not None:
        _BD_YWAVE_PACKET_MIN_BIN = old_BD_YWAVE_PACKET_MIN_BIN
        _BD_YWAVE_PACKET_MAX_BIN = old_BD_YWAVE_PACKET_MAX_BIN
        _BD_YWAVE_NBR_BINS = old_BD_YWAVE_NBR_BINS
        _BD_YWAVE_PACKET_LENGTH = old_BD_YWAVE_PACKET_LENGTH
        LENGTH_FROM_SERIAL_OUTPUT = oldLENGTH_FROM_SERIAL_OUTPUT

    return (kind, message_metadata, list_decoded_packets)

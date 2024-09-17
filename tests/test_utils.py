from trajan.utils import interpolate_variable_to_newtimes


def test_interpolate_variable_to_newtimes_order_0():
    times = np.array(
        [
            np.datetime64("2024-09-17T12:00:00"),
            np.datetime64("2024-09-17T20:00:00"),
            np.datetime64("2024-09-17T21:00:00"),
            np.datetime64("2024-09-18T12:00:00"),
            np.datetime64("2024-09-18T13:00:00"),
            np.datetime64("2024-09-18T14:00:00"),
            np.datetime64("2024-09-18T15:00:00"),
        ]
    )

    variable_on_times = np.array(
        [
            0,
            10,
            11,
            100,
            200,
            np.nan,
            400,
        ]
    )

    newtimes = np.array(
        [
            np.datetime64("2024-09-17T06:00:00"),
            np.datetime64("2024-09-17T11:00:00"),
            np.datetime64("2024-09-17T13:00:00"),
            np.datetime64("2024-09-17T16:00:00"),
            np.datetime64("2024-09-17T19:00:00"),
            np.datetime64("2024-09-17T20:00:00"),
            np.datetime64("2024-09-17T20:29:00"),
            np.datetime64("2024-09-18T13:00:00"),
            np.datetime64("2024-09-18T13:30:00"),
            np.datetime64("2024-09-18T13:59:00"),
            np.datetime64("2024-09-18T15:00:00"),
            np.datetime64("2024-09-18T16:00:00"),
            np.datetime64("2024-09-18T20:00:00"),
        ]
    )

    expected = np.array(
        [
            np.nan,
            0,
            0,
            np.nan,
            10,
            10,
            10,
            200,
            200,
            200,
            400,
            400,
            np.nan,
        ]
    )

    result = interpolate_variable_to_newtimes(times, variable_on_times, newtimes, max_order=0)

    # print(f"{result}")
    # print(f"{expected}")

    assert ((result == expected) | (np.isnan(result) & np.isnan(expected))).all()


def test_interpolate_variable_to_newtimes_order_1():
    times = np.array(
        [
            np.datetime64("2024-09-17T12:00:00"),
            np.datetime64("2024-09-17T20:00:00"),
            np.datetime64("2024-09-17T21:00:00"),
            np.datetime64("2024-09-18T12:00:00"),
            np.datetime64("2024-09-18T13:00:00"),
            np.datetime64("2024-09-18T14:00:00"),
            np.datetime64("2024-09-18T15:00:00"),
        ]
    )

    variable_on_times = np.array(
        [
            0,
            10,
            11,
            100,
            200,
            np.nan,
            400,
        ]
    )

    newtimes = np.array(
        [
            np.datetime64("2024-09-17T06:00:00"),
            np.datetime64("2024-09-17T11:00:00"),
            np.datetime64("2024-09-17T13:00:00"),
            np.datetime64("2024-09-17T16:00:00"),
            np.datetime64("2024-09-17T19:00:00"),
            np.datetime64("2024-09-17T20:00:00"),
            np.datetime64("2024-09-17T20:29:00"),
            np.datetime64("2024-09-18T13:00:00"),
            np.datetime64("2024-09-18T13:30:00"),
            np.datetime64("2024-09-18T14:00:00"),
            np.datetime64("2024-09-18T15:00:00"),
            np.datetime64("2024-09-18T16:00:00"),
            np.datetime64("2024-09-18T20:00:00"),
        ]
    )

    expected = np.array(
        [
            np.nan,
            0,
            0,
            np.nan,
            10,
            10,
            10.5,
            200,
            250,
            300,
            400,
            400,
            np.nan,
        ]
    )

    result = interpolate_variable_to_newtimes(times, variable_on_times, newtimes, max_order=1)

    # print(f"{result}")
    # print(f"{expected}")

    assert ((np.abs(result - expected) < 0.1) | (np.isnan(result) & np.isnan(expected))).all()

"""Test for pdcleaner date_range detection method (Series)"""

import pytest

import pandas as pd
from pandas.testing import assert_series_equal


def test_detect_call_as_parameter(series_with_datetime):
    detect_results = \
        series_with_datetime.cleaner.detect('date_range', lower='2020-06-15', upper='2022-08-05')
    assert detect_results.lower == '2020-06-15'
    assert detect_results.upper == '2022-08-05'


def test_detect_call_as_method(series_with_datetime):
    detect_results = \
        series_with_datetime.cleaner.detect.date_range(lower='2020-06-15', upper='2022-08-05')
    assert detect_results.lower == '2020-06-15'
    assert detect_results.upper == '2022-08-05'


def test_detect_call_with_no_upper(series_with_datetime):
    detect_results = series_with_datetime.cleaner.detect('date_range', lower='2020-06-15')
    expected = pd.Series([False, False, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_call_with_no_lower(series_with_datetime):
    detect_results = series_with_datetime.cleaner.detect('date_range', upper='2022-08-05')
    expected = pd.Series([True, False, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_inclusive_arg_detect_on_series(series_with_datetime):
    # pd.Series(['2022-10-01', '2021-06-11', '2019-04-03', ' 2020-09-25'])
    result = series_with_datetime.cleaner.detect.date_range(lower='2019-04-03',
                                                            upper='2022-10-01').is_error()

    expected = pd.Series([False, False, False, False])
    assert_series_equal(result, expected)

    result = series_with_datetime.cleaner.detect.date_range(lower='2019-04-03',
                                                            upper='2022-10-01',
                                                            inclusive="right").is_error()

    expected = pd.Series([False, False, True, False])
    assert_series_equal(result, expected)

    result = series_with_datetime.cleaner.detect.date_range(lower='2019-04-03',
                                                            upper='2022-10-01',
                                                            inclusive="left").is_error()

    expected = pd.Series([True, False, False, False])
    assert_series_equal(result, expected)

    result = series_with_datetime.cleaner.detect.date_range(lower='2019-04-03',
                                                            upper='2022-10-01',
                                                            inclusive="neither").is_error()
    expected = pd.Series([True, False, True, False])
    assert_series_equal(result, expected)


def test_no_specified_bounded_warns(series_with_datetime):
    msg = "Neither lower nor upper specified"
    with pytest.raises(ValueError, match=msg):
        series_with_datetime.cleaner.detect.date_range()


def test_lower_gte_upper(series_with_datetime):
    msg = "Lower bound is >= upper bound"
    with pytest.raises(ValueError, match=msg):
        series_with_datetime.cleaner.detect.date_range(lower='2022-08-05', upper='2020-06-15')


def test_lower_eq_upper(series_with_datetime):
    msg = "Lower bound is >= upper bound"
    with pytest.raises(ValueError, match=msg):
        series_with_datetime.cleaner.detect.date_range(lower='2022-08-05', upper='2022-08-05')


def test_invalid_lower_type_raise(series_with_datetime):
    msg = "Lower bound must be date format"
    with pytest.raises(TypeError, match=msg):
        series_with_datetime.cleaner.detect.date_range(lower='2022-18-05')


def test_invalid_upper_type_raise(series_with_datetime):
    msg = "Upper bound must be date format"
    with pytest.raises(TypeError, match=msg):
        series_with_datetime.cleaner.detect.date_range(upper='2022-58-05')


def test_date_range_from_existing_detector(series_with_datetime):
    detect_results = series_with_datetime.cleaner.detect('date_range', lower='2020-06-15',
                                                         upper='2022-08-05')

    detector2 = series_with_datetime.cleaner.detect(detect_results)

    assert detect_results.upper == detector2.upper
    assert detect_results.lower == detector2.lower

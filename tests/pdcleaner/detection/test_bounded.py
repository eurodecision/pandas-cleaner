"""Tests for `pdcleaner` bounded detection method (Series)."""

import pytest

import pandas as pd
from pandas.testing import assert_series_equal


def test_detect_call_as_parameter(series_with_outlier):
    detect_results = \
        series_with_outlier.cleaner.detect('bounded', lower=2, upper=4)
    assert detect_results.lower == 2
    assert detect_results.upper == 4


def test_detect_call_as_method(series_with_outlier):
    detect_results = \
        series_with_outlier.cleaner.detect.bounded(lower=2, upper=4)
    assert detect_results.lower == 2
    assert detect_results.upper == 4


def test_detect_with_lower_and_upper(series_with_outlier):
    detect_results = \
        series_with_outlier.cleaner.detect('bounded', lower=2, upper=4)
    expected = pd.Series([True, False, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_call_with_no_upper(series_with_outlier):
    detect_results = series_with_outlier.cleaner.detect('bounded', lower=2)
    expected = pd.Series([True, False, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_call_with_no_lower(series_with_outlier):
    detect_results = series_with_outlier.cleaner.detect('bounded', upper=4)
    expected = pd.Series([False, False, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_call_on_series_with_nan(series_with_nan):
    detect_results = series_with_nan.cleaner\
        .detect.bounded(lower=2, upper=4)
    expected = pd.Series([False, False, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_inclusive_arg_detect_on_series(series_with_nan):
    # [np.nan, 2, 100, 3])
    result = \
        series_with_nan.cleaner.detect.bounded(lower=2, upper=3).is_error()
    expected = pd.Series([False, False, True, False])
    assert_series_equal(result, expected)

    result = (series_with_nan
              .cleaner
              .detect
              .bounded(lower=2,
                       upper=3,
                       inclusive="right")
              .is_error())

    expected = pd.Series([False, True, True, False])
    assert_series_equal(result, expected)

    result = (series_with_nan
              .cleaner
              .detect
              .bounded(lower=2,
                       upper=3,
                       inclusive="left")
              .is_error())

    expected = pd.Series([False, False, True, True])
    assert_series_equal(result, expected)

    result = (series_with_nan
              .cleaner
              .detect
              .bounded(lower=2,
                       upper=3,
                       inclusive="neither")
              .is_error())
    expected = pd.Series([False, True, True, True])
    assert_series_equal(result, expected)


# TestInvalidbounded:


def test_no_specified_bounded_warns(series_with_outlier):
    msg = 'Neither lower nor upper specified'
    with pytest.warns(UserWarning, match=msg):
        series_with_outlier.cleaner.detect.bounded()


def test_lower_gte_upper(series_with_outlier):
    msg = "Lower bound is >= upper bound"
    with pytest.raises(ValueError, match=msg):
        series_with_outlier.cleaner.detect.bounded(lower=2, upper=0)


def test_lower_eq_upper(series_with_outlier):
    msg = "Lower bound is >= upper bound"
    with pytest.raises(ValueError, match=msg):
        series_with_outlier.cleaner.detect.bounded(lower=2, upper=2)


def test_invalid_lower_type_raise(series_with_outlier):
    msg = "bound must be a number"
    with pytest.raises(TypeError, match=msg):
        series_with_outlier.cleaner.detect.bounded(lower="str")


def test_invalid_upper_type_raise(series_with_outlier):
    msg = "bound must be a number"
    with pytest.raises(TypeError, match=msg):
        series_with_outlier.cleaner.detect.bounded(upper='str')

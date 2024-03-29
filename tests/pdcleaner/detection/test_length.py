"""Tests for `pdcleaner` length checking method (Series)."""

import pytest

import pandas as pd
from pandas.testing import assert_series_equal
import pdcleaner


def test_detect_call_as_parameter(series_with_different_length):
    detect_results = \
        series_with_different_length.cleaner.detect('length', lower=2, upper=5)
    assert detect_results.lower == 2
    assert detect_results.upper == 5


def test_detect_call_as_method(series_with_different_length):
    detect_results = \
        series_with_different_length.cleaner.detect.length(lower=2, upper=5)
    assert detect_results.lower == 2
    assert detect_results.upper == 5

def test_length_from_existing_detector(series_with_different_length):
    detect_results = \
        series_with_different_length.cleaner.detect.length(lower=2, upper=5)
    series2 = pd.Series(['123'])
    detector2 = series2.cleaner.detect(detect_results)
    assert detector2.mode=='bound'
    assert detector2.lower == 2
    assert detector2.upper == 5

def test_detect_with_lower_and_upper(series_with_different_length):
    detect_results = series_with_different_length.cleaner.detect('length',
                                                                 lower=2, upper=5)
    expected = pd.Series([False, True, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_only_upper(series_with_different_length):
    detect_results = series_with_different_length.cleaner.detect('length', upper=5)
    expected = pd.Series([False, True, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_only_lower(series_with_different_length):
    detect_results = series_with_different_length.cleaner.detect('length', lower=3)
    expected = pd.Series([False, False, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_fixed_value(series_with_different_length):
    detect_results = series_with_different_length.cleaner.detect('length', value=5)
    expected = pd.Series([False, True, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_call_on_series_with_nan(cat_series_with_nan):
    detect_results = cat_series_with_nan.cleaner.detect.length(value=3)
    expected = pd.Series([False, False, False, False, False, True])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_on_series_with_integers(series_with_integers):
    detect_results = series_with_integers.cleaner.detect.length(upper=6)
    expected = pd.Series([False, True, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_on_series_with_floats(series_with_floats):
    detect_results = series_with_floats.cleaner.detect.length(upper=6)
    expected = pd.Series([False, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_inclusive_arg_detect_on_series(series_with_different_length):
    # pd.Series(['75013', '952401', '93250', '94230'])
    result = series_with_different_length.cleaner.detect.length(lower=2, upper=5).is_error()
    expected = pd.Series([False, True, False, False])
    assert_series_equal(result, expected)

    result = (series_with_different_length
              .cleaner
              .detect
              .length(lower=5,
                      upper=6,
                      inclusive="right")
              .is_error())

    expected = pd.Series([True, False, True, True])
    assert_series_equal(result, expected)

    result = (series_with_different_length
              .cleaner
              .detect
              .length(lower=5,
                      upper=6,
                      inclusive="left")
              .is_error())

    expected = pd.Series([False, True, False, False])
    assert_series_equal(result, expected)

    result = (series_with_different_length
              .cleaner
              .detect
              .length(lower=5,
                      upper=6,
                      inclusive="neither")
              .is_error())
    expected = pd.Series([True, True, True, True])
    assert_series_equal(result, expected)


def test_incompatible_args(series_with_different_length):
    msg = "Incompatible arguments: value and upper or lower"
    with pytest.raises(ValueError, match=msg):
        series_with_different_length.cleaner.detect.length(value=3, upper=5)


def test_incompatible_args_2(series_with_different_length):
    msg = "Incompatible arguments: value and upper or lower"
    with pytest.raises(ValueError, match=msg):
        series_with_different_length.cleaner.detect.length(value=3, lower=1)


def test_lower_gte_upper(series_with_different_length):
    msg = "Lower bound is >= upper bound"
    with pytest.raises(ValueError, match=msg):
        series_with_different_length.cleaner.detect.length(lower=2, upper=0)


def test_lower_eq_upper(series_with_different_length):
    msg = "Lower bound is >= upper bound"
    with pytest.raises(ValueError, match=msg):
        series_with_different_length.cleaner.detect.length(lower=2, upper=2)


# def test_unused_upper_value_given(series_with_different_length):
#     msg = "Upper or Lower bound is not necessary in fixed_value mode"
#     with pytest.warns(UserWarning, match=msg):
#         series_with_different_length.cleaner.detect.length(mode='fixed_value', value=5, upper=5)


# def test_unused_lower_value_given(series_with_different_length):
#     msg = "Upper or Lower bound is not necessary in fixed_value mode"
#     with pytest.warns(UserWarning, match=msg):
#         series_with_different_length.cleaner.detect.length(mode='fixed_value', value=5, lower=1)


# def test_unused_fixed_value_given(series_with_different_length):
#     msg = "value is not necessary in bound mode"
#     with pytest.warns(UserWarning, match=msg):
#         series_with_different_length.cleaner.detect.length(mode='bound', value=5, lower=1)


def test_invalid_lower_type_raise(series_with_outlier):
    msg = "Lower bound must be a number"
    with pytest.raises(TypeError, match=msg):
        series_with_outlier.cleaner.detect.length(lower="str")


def test_invalid_upper_type_raise(series_with_outlier):
    msg = "Upper bound must be a number"
    with pytest.raises(TypeError, match=msg):
        series_with_outlier.cleaner.detect.length(upper="str")


def test_invalid_value_type_raise(series_with_outlier):
    msg = "Argument value must be a number"
    with pytest.raises(TypeError, match=msg):
        series_with_outlier.cleaner.detect.length(value="str")


def test_length_no_arg_raise(series_with_outlier):
    msg="At least one argument must be provided"
    with pytest.raises(ValueError, match=msg):
        series_with_outlier.cleaner.detect.length()

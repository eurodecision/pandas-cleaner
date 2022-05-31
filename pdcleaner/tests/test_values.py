"""Tests for `pdcleaner` detection method (Series)
using an already existing outliers object."""

import pytest
import pandas as pd
from pandas.testing import assert_series_equal


# TestCategoricalValuesInList:


def test_empty_list_raise(cat_series):
    with pytest.raises(ValueError, match="is empty"):
        cat_series.cleaner.detect('enum')


def test_detect_call_on_series(cat_series):
    results = cat_series.cleaner.detect.enum(values=['cat', 'dog'])
    expected = pd.Series([False, False, False, False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_detect_call_on_num_series(series_test_set):
    r"""
    Test with numerical values
    Input: pd.Series([5, 3, 100])
    Output: [True, True, True]
    """
    results = series_test_set.cleaner.detect.enum(values=[5, 3])
    expected = pd.Series([False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_detect_call_on_series_with_nan(cat_series_with_nan):
    results = cat_series_with_nan.cleaner.detect.enum(values=['cat', 'dog'])
    expected = pd.Series([False, False, False, False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_detect_from_existing_object(cat_series, cat_series_test):
    results = cat_series.cleaner.detect.enum(values=['cat', 'dog'])
    results_test = cat_series_test.cleaner.detect(results)
    expected = pd.Series([False, False, True])
    assert_series_equal(results_test.is_error(), expected)


# TestCategoricalValue:


def test_value_empty_list_raise(cat_series):
    r"""Fails when no value is defined"""
    with pytest.raises(ValueError, match="is not defined"):
        cat_series.cleaner.detect('value')


def test_value_detect_call_on_series(cat_series):
    r"""
    Test with categorical values
    Input: pd.Series(['cat', 'cat', 'dog', 'dog', 'dog', 'bird'])
    Output: [False, False, True, True, True, True]
    """
    results = cat_series.cleaner.detect.value(value='cat')
    expected = pd.Series([False, False, True, True, True, True])
    assert_series_equal(results.is_error(), expected)


def test_value_detect_call_on_num_series(series_test_set):
    r"""
    Test with numerical values
    Input: pd.Series([5, 3, 100])
    Output: [True, False, True]
    """
    results = series_test_set.cleaner.detect.value(value=3)
    expected = pd.Series([True, False, True])
    assert_series_equal(results.is_error(), expected)


def test_value_detect_call_on_num_series_float(series_test_set):
    r"""
    Test with numerical values
    Input: pd.Series([5, 3, 100])
    Output: [True, True, True]
    """
    results = series_test_set.cleaner.detect.value(value=3.0)
    expected = pd.Series([True, True, True])
    assert_series_equal(results.is_error(), expected)


def test_value_detect_call_on_num_series_float_check(series_test_set):
    r"""
    Test with numerical values, check if float value is present
    Input: pd.Series([5, 3, 100])
    Output: [True, True, True]
    """
    results = series_test_set.cleaner.detect.value(value=3.0, check_type=True)
    expected = pd.Series([True, True, True])
    assert_series_equal(results.is_error(), expected)


def test_value_detect_call_on_num_series_float_no_check(series_test_set):
    r"""
    Test with numerical values, check if float value is present
    Input: pd.Series([5, 3, 100])
    Output: [True, False, True]
    """
    results = series_test_set.cleaner.detect.value(value=3.0, check_type=False)
    expected = pd.Series([True, False, True])
    assert_series_equal(results.is_error(), expected)


def test_value_detect_call_on_num_series_str_no_check(series_test_set):
    r"""
    Test with numerical values, check if str value present
    Input: pd.Series([5, 3, 100])
    Output: [True, True, True]
    """
    results = series_test_set.cleaner.detect.value(value='3', check_type=False)
    expected = pd.Series([True, True, True])
    assert_series_equal(results.is_error(), expected)


def test_value_detect_call_on_series_with_nan(cat_series_with_nan):
    r"""
    Test with categorical values and nan values
    Input: pd.Series(['cat', 'cat', 'dog', np.nan, 'dog', 'bird'])
    Output: [False, False, True, False, True, True]
    """
    results = cat_series_with_nan.cleaner.detect.value(value='cat')
    expected = pd.Series([False, False, True, False, True, True])
    assert_series_equal(results.is_error(), expected)


def test_value_detect_from_existing_object(cat_series, cat_series_test):
    r"""
    Test with categorical values, with a previously defined detector
    The second detector uses the authorized value ('cat') from
    the first detector
    Input: pd.Series(['cat', 'dog', 'bird'])
    Output: [False, True, True]
    """
    results = cat_series.cleaner.detect.value(value='cat')
    results_test = cat_series_test.cleaner.detect(results)
    expected = pd.Series([False, True, True])
    assert_series_equal(results_test.is_error(), expected)


# TestCategoricalValueCounts:


def test_negative_n_raise(cat_series):
    with pytest.raises(TypeError, match="must be a >0 integer"):
        cat_series.cleaner.detect('counts', n=-1)


def test_n_not_int_raise(cat_series):
    with pytest.raises(TypeError, match="must be a >0 integer"):
        cat_series.cleaner.detect('counts', n=2.3)


def test_detect_call_on_series_count(cat_series):
    results = cat_series.cleaner.detect.counts(n=1)
    expected = pd.Series([False, False, False, False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_detect_call_on_series_count_num(series_test_counts_set):
    results = series_test_counts_set.cleaner.detect.counts(n=1)
    expected = pd.Series([False, False, False, True, False])
    assert_series_equal(results.is_error(), expected)


def test_detect_call_on_series_with_nan_count(cat_series_with_nan):
    results = cat_series_with_nan.cleaner.detect.counts(n=1)
    expected = pd.Series([False, False, False, False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_results_attributes(cat_series):
    results = cat_series.cleaner.detect.counts(n=1)
    assert results.n == 1


def test_detect_from_existing_object_count(cat_series, cat_series_test):
    results = cat_series.cleaner.detect.counts(n=1)
    results_test = cat_series_test.cleaner.detect(results)
    expected = pd.Series([False, False, True])
    assert_series_equal(results_test.is_error(), expected)


# TestCategoricalValueFreqs:


def test_not_valid_freq_raise(cat_series):
    with pytest.raises(TypeError, match="must be in the range"):
        cat_series.cleaner.detect.freq(freq=-1.3)


def test_not_freq_not_float_raise(cat_series):
    with pytest.raises(TypeError, match="freq must be a float"):
        cat_series.cleaner.detect.freq(freq='1')


def test_freq_not_float_raise(cat_series):
    with pytest.raises(TypeError):
        cat_series.cleaner.detect('freq', freq='str')


def test_detect_call_on_series_freq(cat_series):
    results = cat_series.cleaner.detect.freq(freq=0.25)
    expected = pd.Series([False, False, False, False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_detect_call_on_series_freq_num(series_test_counts_set):
    results = series_test_counts_set.cleaner.detect.freq(freq=0.25)
    expected = pd.Series([False, False, False, True, False])
    assert_series_equal(results.is_error(), expected)


def test_detect_call_on_series_with_nan_freq(cat_series_with_nan):
    results = cat_series_with_nan.cleaner.detect.freq(freq=0.25)
    expected = pd.Series([False, False, False, False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_results_attributes_freq(cat_series):
    results = cat_series.cleaner.detect.freq(freq=0.1)
    assert results.freq == .1


def test_detect_from_existing_object_freq(cat_series, cat_series_test):
    results = cat_series.cleaner.detect.freq(freq=0.25)
    results_test = cat_series_test.cleaner.detect(results)
    expected = pd.Series([False, False, True])
    assert_series_equal(results_test.is_error(), expected)

# Test argument forbidden


def test_value_forbidden_false(series_test_set):
    r"""
    Test with numerical values
    Input: pd.Series([5, 3, 100])
    """
    results = \
        series_test_set.cleaner.detect.value(value=3, forbidden=False)
    expected = pd.Series([True, False, True])
    assert_series_equal(results.is_error(), expected)


def test_value_forbidden_true(series_test_set):
    r"""
    Test with numerical values and value forbidden
    """
    results = \
        series_test_set.cleaner.detect.value(value=3, forbidden=True)
    expected = pd.Series([False, True, False])
    assert_series_equal(results.is_error(), expected)


def test_values_forbidden_false(series_test_set):
    r"""
    Test with numerical values
    Input: pd.Series([5, 3, 100])
    """
    results = \
        series_test_set.cleaner.detect.enum(values=[5, 3], forbidden=False)
    expected = pd.Series([False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_values_forbidden_true(series_test_set):
    r"""
    Test with numerical values and value forbidden
    """
    results = \
        series_test_set.cleaner.detect.enum(values=[5, 3], forbidden=True)
    expected = pd.Series([True, True, False])
    assert_series_equal(results.is_error(), expected)

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal


def test_series_type():
    series = pd.Series([1, 2, 100, 3], dtype='float64')
    series[1] = 'One'
    results = series.cleaner.detect.types(ptype=float)
    expected = pd.Series([False, True, False, False, ])
    assert_series_equal(results.is_error(), expected)


def test_series_type_with_nan():
    """Missing values are not treated as errors."""
    series = pd.Series([1, 2, 3, 4], dtype='int64')
    series[1] = 'One'
    series[2] = np.nan
    results = series.cleaner.detect.types(ptype=int)
    expected = pd.Series([False, True, False, False, ])
    assert_series_equal(results.is_error(), expected)


def test_series_type_different_than_first():
    """If no type is specified, element are compared to the first one"""
    series = pd.Series(['A', 2, np.nan, 'D'])
    results = series.cleaner.detect('types')
    expected = pd.Series([False, True, False, False, ])
    assert_series_equal(results.is_error(), expected)


def test_series_type_from_existing_detector():
    """The first detector detects the right type as str"""
    series = pd.Series(['A', 2, np.nan, 'D'])
    series_test = pd.Series([1, 'Two'])
    results = series.cleaner.detect('types')
    results_test = series_test.cleaner.detect(results)
    expected = pd.Series([True, False])
    assert results_test.ptype == str == results.ptype
    assert_series_equal(results_test.is_error(), expected)


def test_series_type_unvalid_ptype():
    series = pd.Series([1, 2, 100, 3], dtype='float64')
    match = 'must be a python built-in type'
    with pytest.raises(TypeError, match=match):
        series.cleaner.detect.types(ptype='toto')


def test_series_type_given_as_string():
    series = pd.Series([1, 'A', 100, 3])
    results = series.cleaner.detect.types(ptype='int')
    expected = pd.Series([False, True, False, False, ])
    assert_series_equal(results.is_error(), expected)

"""Tests for `pdcleaner` detecting missing values (DataFrame or Series)."""

import pytest

import pandas as pd
from pandas.testing import assert_series_equal


def test_detect_call_as_parameter(dataframe_with_nan):
    detect_results = dataframe_with_nan.cleaner.detect('missing')
    assert detect_results.how == 'any'


def test_detect_call_as_method(dataframe_with_nan):
    detect_results = dataframe_with_nan.cleaner.detect.missing(how='all')
    assert detect_results.how == 'all'


def test_detect_with_any_parameter(dataframe_with_nan):
    detect_results = dataframe_with_nan.cleaner.detect.missing(how='any')
    expected = pd.Series([True, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_all_parameter(dataframe_with_nan):
    detect_results = dataframe_with_nan.cleaner.detect.missing(how='all')
    expected = pd.Series([False, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_series(series_with_nan):
    detect_results = series_with_nan.cleaner.detect.missing()
    expected = pd.Series([True, False, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_series_object_type(series_with_nan):
    detect_results = series_with_nan.cleaner.detect.missing()
    assert detect_results.obj_type == 'series'


def test_dataframe_object_type(dataframe_with_nan):
    detect_results = dataframe_with_nan.cleaner.detect.missing()
    assert detect_results.obj_type == 'dataframe'


def test_with_unknown_parameter(dataframe_with_nan):
    msg = "how parameter must be any or all"
    with pytest.raises(ValueError, match=msg):
        dataframe_with_nan.cleaner.detect.missing(how='neither')


def test_missing_with_existing_detector(dataframe_with_nan):
    detector = dataframe_with_nan.cleaner.detect.missing(how='all')
    detector2 = pd.DataFrame({'x': [1, 2]}).cleaner.detect(detector)
    assert detector.how == detector2.how

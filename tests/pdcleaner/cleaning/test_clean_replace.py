import pytest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal


def test_replace_num_series(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean('replace', results, value=0)
    expected = pd.Series([1, 2, 0, 4])
    assert_series_equal(cleaned, expected)


def test_replace_num_series_direct_call(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean.replace(results, value=0)
    expected = pd.Series([1, 2, 0, 4])
    assert_series_equal(cleaned, expected)


def test_replace_num_col(df_quanti_quali):
    results = df_quanti_quali['num'].cleaner.detect.bounded(upper=10)
    df_quanti_quali['num'].cleaner.clean('replace', results, value=0, inplace=True)
    expected = pd.Series([1, 2, 0, 4, 5], name='num')
    assert_series_equal(df_quanti_quali['num'], expected)



def test_replace_with_value(series_with_nan):
    # pd.Series([np.nan, 2, 100, 3])
    results = series_with_nan.cleaner.detect.bounded(upper=10)
    cleaned = series_with_nan.cleaner.clean('replace', results, value=0)
    expected = pd.Series([np.nan, 2, 0, 3])
    assert_series_equal(cleaned, expected)


def test_replace_inplace(series_with_nan):
    series_obj = series_with_nan
    # pd.Series([np.nan, 2, 100, 3])
    results = series_obj.cleaner.detect.bounded(upper=10)
    series_obj.cleaner.clean('replace', results, value=0, inplace=True)
    expected = pd.Series([np.nan, 2, 0, 3])
    assert_series_equal(series_obj, expected)


def test_replace_with_lambda(series_with_nan):
    # pd.Series([np.nan, 2, 100, 3])
    results = series_with_nan.cleaner.detect.bounded(upper=10)
    cleaned = series_with_nan.cleaner.clean('replace', results, value=lambda s: 5)
    expected = pd.Series([np.nan, 2, 5, 3])
    assert_series_equal(cleaned, expected)


def test_replace_with_dict(series_with_nan):
    # pd.Series([np.nan, 2, 100, 3])
    results = series_with_nan.cleaner.detect.bounded(upper=10)
    cleaned = series_with_nan.cleaner.clean('replace', results, value={100: 1, 99: 9, 2: -2})
    expected = pd.Series([np.nan, 2, 1, 3])
    assert_series_equal(cleaned, expected)


def test_replace_with_incomplete_dict(series_with_nan):
    # pd.Series([np.nan, 2, 100, 3])
    results = series_with_nan.cleaner.detect.bounded(upper=10)
    cleaned = series_with_nan.cleaner.clean('replace', results, value={99: 9, 2: -2})
    expected = pd.Series([np.nan, 2, np.nan, 3])
    assert_series_equal(cleaned, expected)

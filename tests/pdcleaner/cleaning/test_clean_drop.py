import pytest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal


def test_drop_num_series_direct_call(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean.drop(results)
    expected = pd.Series([1, 2, 4], index=[0, 1, 3])
    assert_series_equal(cleaned, expected)


def test_drop_num_series(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean('drop', results)
    expected = pd.Series([1, 2, 4], index=[0, 1, 3])
    assert_series_equal(cleaned, expected)


def test_drop_num_series_direct_call(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean.drop(results)
    expected = pd.Series([1, 2, 4], index=[0, 1, 3])
    assert_series_equal(cleaned, expected)


def test_drop_num_series(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean('drop', results)
    expected = pd.Series([1, 2, 4], index=[0, 1, 3])
    assert_series_equal(cleaned, expected)


def test_drop_num_col(df_quanti_quali):
    results = df_quanti_quali['num'].cleaner.detect.bounded(upper=10)
    msg = "Dropping inplace will not modify the DataFrame"
    with pytest.warns(UserWarning, match=msg):
        df_quanti_quali['num'].cleaner.clean('drop', results, inplace=True)


def test_clean_df_drop(df_quanti_quali):
    detector = df_quanti_quali['cat'].cleaner.detect.enum(values=['cat', 'dog'])
    df_clean = df_quanti_quali.cleaner.clean('drop', detector)
    expected = pd.DataFrame({'num': pd.Series([1, 2, 100, 4]),
                             'cat': pd.Series(['cat',
                                               'cat',
                                               'dog',
                                               'dog',
                                               ])})
    assert_frame_equal(df_clean, expected)

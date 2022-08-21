"""Test methods detected() and valid()"""

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal


def test_valid(series_with_outlier):
    """valid detector property"""
    detector = \
        series_with_outlier.cleaner.detect('bounded', lower=2, upper=4)
    expected = pd.Series([2, 4], index=[1, 3])
    assert_series_equal(detector.valid, expected)


def test_detected_series(series_with_outlier):
    """[1, 2, 100, 4]"""
    detector = \
        series_with_outlier.cleaner.detect('bounded', lower=2, upper=4)
    # expected = pd.Series([True, False, True, False])
    detected  = series_with_outlier.cleaner.detected(detector)
    assert_series_equal(detected, detector.detected)


def test_valid_series(series_with_outlier):
    """[1, 2, 100, 4]"""
    detector = \
        series_with_outlier.cleaner.detect('bounded', lower=2, upper=4)
    # expected = pd.Series([True, False, True, False])
    valid  = series_with_outlier.cleaner.valid(detector)
    assert_series_equal(valid, detector.valid)


def test_detected_df(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect.associations(freq=0.05)
    expected_detected = pd.DataFrame({'col1': 'B', 'col2': 'a'}, index=[19])
    assert_frame_equal(df_two_cat_cols.cleaner.detected(detector), expected_detected)


def test_valid_df(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect.associations(freq=0.05)
    assert_frame_equal(df_two_cat_cols.cleaner.valid(detector), detector.valid)
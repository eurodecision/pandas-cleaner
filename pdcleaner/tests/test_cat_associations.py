"""
Module to detection method for two-columns categorical DataFrames
that counts the associations
"""

import pandas as pd

import pytest

from pandas.testing import assert_frame_equal, assert_series_equal


# TestAssociationsDetector:


def test_applied_to_series():
    series = pd.Series(['a', 'b'])
    match = "detector applies to DataFrames."
    with pytest.raises(TypeError, match=match):
        series.cleaner.detect.associations()


def test_applied_to_wrong_type_df_two_cat_cols():
    df_two_cat_cols = pd.DataFrame({'x': [1, 2], 'y': ['A', 'B']})
    match = "applies to DataFrames with string/text/categorical columns"
    with pytest.raises(TypeError, match=match):
        df_two_cat_cols.cleaner.detect.associations()


def test_applied_to_too_many_obj_columns():
    df_two_cat_cols = pd.DataFrame({'x': ['A', 'A'],
                                    'y': ['a', 'b'],
                                    'z': ['c', 'c']})
    match = "applies to DataFrames with two object/categorical columns"
    with pytest.raises(ValueError, match=match):
        df_two_cat_cols.cleaner.detect("associations")


def test_count_is_int(df_two_cat_cols):
    match = "count must be an integer"
    with pytest.raises(TypeError, match=match):
        df_two_cat_cols.cleaner.detect.associations(count='one')


def test_freq_is_number(df_two_cat_cols):
    match = 'freq must be a number'
    with pytest.raises(TypeError, match=match):
        df_two_cat_cols.cleaner.detect.associations(freq='zero')


def test_freq_value_error(df_two_cat_cols):
    match = "freq must be between 0 and 1 exclusive"
    with pytest.raises(ValueError, match=match):
        df_two_cat_cols.cleaner.detect.associations(freq=2.)


def test_missing_params(df_two_cat_cols):
    match = "Either freq or count must be provided"
    with pytest.raises(ValueError, match=match):
        df_two_cat_cols.cleaner.detect.associations()


def test_too_many_params(df_two_cat_cols):
    match = "Either freq or count must be provided"
    with pytest.raises(ValueError, match=match):
        df_two_cat_cols.cleaner.detect.associations(count=2, freq=0.1)


def test_parameters_properties(df_two_cat_cols):
    assert df_two_cat_cols.cleaner.detect("associations", count=1).count == 1
    assert df_two_cat_cols.cleaner.detect("associations", freq=.1).freq == .1


def test_normalize(df_two_cat_cols):
    assert df_two_cat_cols.cleaner.detect("associations", count=1).normalize is False
    assert df_two_cat_cols.cleaner.detect("associations", freq=.1).normalize is True


def test_limit(df_two_cat_cols):
    assert df_two_cat_cols.cleaner.detect("associations", count=1).limit == 1
    assert df_two_cat_cols.cleaner.detect("associations", freq=.1).limit == .1


def test_valid_associations(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect.associations(freq=0.05)
    assert detector.valid_associations == [('A', 'a'), ('A', 'c'), ('B', 'b')]
    detector = df_two_cat_cols.cleaner.detect("associations", count=3)
    assert detector.valid_associations == [('A', 'a'), ('B', 'b')]


def test_errors_detected_with_freq(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect.associations(freq=0.05)
    expected_detected = pd.DataFrame({'col1': 'B', 'col2': 'a'}, index=[19])
    expected_errors = pd.Series([False] * 19 + [True])
    assert_frame_equal(detector.detected, expected_detected)
    assert_series_equal(detector.is_error(), expected_errors)


def test_errors_detected_with_count(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect.associations(count=3)
    expected_detected = pd.DataFrame({'col1': ['A', 'A', 'B'],
                                      'col2': ['c', 'c', 'a']},
                                     index=[8, 9, 19]
                                     )
    assert_frame_equal(detector.detected, expected_detected)


def test_from_existing(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect.associations(count=3)
    df_two_cat_cols_test = pd.DataFrame({'col1': ['A', 'A', 'B'],
                                         'col2': ['c', 'a', 'b']})
    errors = df_two_cat_cols_test.cleaner.detect(detector).is_error()
    expected = pd.Series([True, False, False])
    assert_series_equal(errors, expected)

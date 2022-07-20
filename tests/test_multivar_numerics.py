"""
Module to test multivariate numeric detectors
"""

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

import pytest


def subset(df: pd.DataFrame, category: str) -> pd.DataFrame:
    return df[df.dataset == category].reset_index()[['x', 'y']]


def test_applied_to_series():
    series = pd.Series([1, np.nan, 'a'])
    match = "detector applies to DataFrames."
    with pytest.raises(TypeError, match=match):
        series.cleaner.detect.ndoutliers()


def test_applied_to_unvalid_df(anscombe):
    match = "applies to DataFrames with numeric columns only"
    with pytest.raises(TypeError, match=match):
        anscombe.cleaner.detect.ndoutliers()


def test_parameters_default(anscombe):
    df = subset(anscombe, 'I')
    detector = df.cleaner.detect.ndoutliers()
    assert detector.min_samples == 2


def test_parameters_modify(anscombe):
    df = subset(anscombe, 'I')
    detector = df.cleaner.detect.ndoutliers(eps=0.1, min_samples=3)
    assert detector.eps == 0.1
    assert detector.min_samples == 3


def test_type_error_parameter_eps(anscombe):
    df = subset(anscombe, 'I')
    match = 'eps must be a number'
    with pytest.raises(TypeError, match=match):
        df.cleaner.detect.ndoutliers(eps='string')


def test_type_error_parameter_min_samples(anscombe):
    df = subset(anscombe, 'I')
    match = 'min_samples must be an integer'
    with pytest.raises(TypeError, match=match):
        df.cleaner.detect.ndoutliers(min_samples=2.5)


def test_anscombe_i(anscombe):
    df = subset(anscombe, 'I')
    detector = df.cleaner.detect.ndoutliers()
    print(detector.detected)
    assert detector.has_errors() is False


def test_anscombe_ii(anscombe):
    df = subset(anscombe, 'II')
    detector = df.cleaner.detect.ndoutliers()
    assert detector.has_errors() is False


def test_anscombe_iii(anscombe):
    df = subset(anscombe, 'III')
    detector = df.cleaner.detect.ndoutliers()
    expected = pd.Series([False] * 2 + [True] + [False] * 8)
    assert_series_equal(detector.is_error(), expected)


def test_anscombe_iv(anscombe):
    df = subset(anscombe, 'IV')
    detector = df.cleaner.detect.ndoutliers()
    expected = pd.Series([False] * 7 + [True] + [False] * 3)
    assert_series_equal(detector.is_error(), expected)


def test_from_existing_detector(anscombe):
    df = subset(anscombe, 'II')
    df_test = pd.DataFrame({'x': [12], 'y': [4]})
    detector = df.cleaner.detect('ndoutliers')
    match = "This detection method can not be used"\
            " with an existing detector as an input."
    with pytest.raises(ValueError, match=match):
        df_test.cleaner.detect(detector)


def test_with_nan(anscombe):
    df = subset(anscombe, 'I')
    df.loc[df.x == 10, 'y'] = np.nan
    detector = df.cleaner.detect.ndoutliers()
    assert detector.has_errors() is False


def test_simple_3dset():
    df = pd.DataFrame({'x': [1, 1.1, 4],
                       'y': [1.1, 1, 4],
                       'z': [1, 1.1, 4]})
    detector = df.cleaner.detect.ndoutliers()
    expected = pd.Series([False, False, True])
    assert_series_equal(detector.is_error(), expected)


def test_dbscan_error():
    dataframe = pd.DataFrame({'x': [1], 'y': [2]})
    match = 'DBScan error: see sklearn documentation for help'
    with pytest.raises(ValueError, match=match):
        dataframe.cleaner.detect('ndoutliers').index

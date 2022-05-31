"""Tests for `pdcleaner` detection method (Series)
using an already existing outliers object."""

import pandas as pd
from pandas.testing import assert_series_equal


def test_alphabetic_index_is_error(series_alpha_index):
    detector = \
        series_alpha_index.cleaner.detect.bounded(lower=2, upper=4)
    actual = detector.is_error()
    expected = pd.Series([True, False, True, False],
                         index=["a", "b", "c", "d"])
    assert_series_equal(actual, expected)


def test_alphabetic_index_not_error(series_alpha_index):
    detector = \
        series_alpha_index.cleaner.detect.bounded(lower=2, upper=4)
    actual = detector.not_error()
    expected = pd.Series([False, True, False, True],
                         index=["a", "b", "c", "d"])
    assert_series_equal(actual, expected)


def test_alphabetic_index_detected(series_alpha_index):
    detector = series_alpha_index.cleaner.detect.bounded(lower=2, upper=4)
    actual = detector.detected
    expected = pd.Series([1, 100], index=["a", "c"])
    assert_series_equal(actual, expected)


def test_unordered_numeric_index_is_error(series_unsorted_idx):
    detector = series_unsorted_idx.cleaner.detect.bounded(lower=2, upper=4)
    actual = detector.is_error()
    expected = pd.Series([True, False, True, False], index=[12, 3, 5, 1])
    assert_series_equal(actual, expected)


def test_unordered_numeric_index_not_error(series_unsorted_idx):
    detector = series_unsorted_idx.cleaner.detect.bounded(lower=2, upper=4)
    actual = detector.not_error()
    expected = pd.Series([False, True, False, True], index=[12, 3, 5, 1])
    assert_series_equal(actual, expected)


def test_unordered_numeric_index_detected(series_unsorted_idx):
    detector = series_unsorted_idx.cleaner.detect.bounded(lower=2, upper=4)
    actual = detector.detected
    expected = pd.Series([1, 100], index=[12, 5])
    assert_series_equal(actual, expected)

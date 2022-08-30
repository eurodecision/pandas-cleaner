"""Tests for `pdcleaner` detection method (altern_series)
based on key collisions."""

import pytest
import pandas as pd
from pandas.testing import assert_series_equal


def test_raise_if_numerical_series(series_with_outlier):
    with pytest.raises(TypeError, match="applies to categorical/string Series"):
        series_with_outlier.cleaner.detect('alternatives')


def test_wrong_method(altern_series):
    with pytest.raises(ValueError, match='ot a valid method'):
        altern_series.cleaner.detect('alternatives', keys='blabla')


def test_method_not_a_string(altern_series):
    with pytest.raises(TypeError, match='must be a string'):
        altern_series.cleaner.detect('alternatives', keys=42)


def test_alternatives(altern_series):
    results = altern_series.cleaner.detect.alternatives()
    expected = pd.Series([False, True, True, False, False, ])
    assert_series_equal(results.is_error(), expected)


def test_alternatives_with_nan(altern_series_with_nan):
    results = altern_series_with_nan.cleaner.detect.alternatives()
    expected = pd.Series([False, False, True, True, False, ])
    assert_series_equal(results.is_error(), expected)


def test_alternatives_from_existing_detector(altern_series, altern_test_series):
    detector = altern_series.cleaner.detect.alternatives()
    results_test = altern_test_series.cleaner.detect(detector)
    expected = pd.Series([True, False])
    assert_series_equal(results_test.is_error(), expected)


def test_alternatives_dict(altern_series):
    results = altern_series.cleaner.detect.alternatives().dict_keys
    expected = {'linus torvalds': 'Linus Torvalds',
                'bill gates': 'Bill Gates'}
    assert results == expected

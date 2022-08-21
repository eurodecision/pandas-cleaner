"""Tests for `pdcleaner` detection method (keycol_series)
based on key collisions."""

import pytest
import pandas as pd
from pandas.testing import assert_series_equal


def test_raise_if_numerical_series(series_with_outlier):
    with pytest.raises(TypeError, match="applies to categorical/string Series"):
        series_with_outlier.cleaner.detect('keycollision')


def test_wrong_method(keycol_series):
    with pytest.raises(ValueError, match='ot a valid method'):
        keycol_series.cleaner.detect('keycollision', keys='blabla')


def test_method_not_a_string(keycol_series):
    with pytest.raises(TypeError, match='must be a string'):
        keycol_series.cleaner.detect('keycollision', keys=42)


def test_keycollision(keycol_series):
    results = keycol_series.cleaner.detect.keycollision()
    expected = pd.Series([False, True, True, False, False, ])
    assert_series_equal(results.is_error(), expected)


def test_keycollision_with_nan(keycol_series_with_nan):
    results = keycol_series_with_nan.cleaner.detect.keycollision()
    expected = pd.Series([False, False, True, True, False, ])
    assert_series_equal(results.is_error(), expected)


def test_keycollision_from_existing_detector(keycol_series, keycol_test_series):
    detector = keycol_series.cleaner.detect.keycollision()
    results_test = keycol_test_series.cleaner.detect(detector)
    expected = pd.Series([True, False])
    assert_series_equal(results_test.is_error(), expected)


def test_keycollision_dict(keycol_series):
    results = keycol_series.cleaner.detect.keycollision().dict_keys
    expected = {'linus torvalds': 'Linus Torvalds',
                'bill gates': 'Bill Gates'}
    assert results == expected

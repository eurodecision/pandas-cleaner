"""Tests for `pdcleaner` detection method (Series)
that detects if categorical/qualitative values match a given pattern."""

import re
import pytest
import pandas as pd
from pandas.testing import assert_series_equal


def test_numerical_series_raise(series_with_outlier):
    with pytest.raises(TypeError, match="applies to categorical/string Series"):
        series_with_outlier.cleaner.detect('pattern')


def test_empty_pattern(cat_series_with_nan_and_numbers):
    with pytest.raises(ValueError, match="is empty"):
        cat_series_with_nan_and_numbers.cleaner.detect('pattern')


def test_wrong_mode(cat_series_with_nan_and_numbers):
    with pytest.raises(ValueError, match="mode shoud be one"):
        cat_series_with_nan_and_numbers.cleaner.detect('pattern', pattern=r"d", mode='blabla')


def test_fullmatch(cat_series_with_nan_and_numbers):
    results = \
        cat_series_with_nan_and_numbers.cleaner.detect.pattern(pattern=r"[a-z]*",
                                                               mode='fullmatch'
                                                               )
    expected = pd.Series([True, False, False, False, True, False, False])
    assert_series_equal(results.is_error(), expected)


def test_contains(cat_series_with_nan_and_numbers):
    results = cat_series_with_nan_and_numbers.cleaner.detect.pattern(pattern=r"d", mode='contains')
    expected = pd.Series([True, True, False, False, True, False, True])
    assert_series_equal(results.is_error(), expected)


def test_match_and_case(cat_series_with_nan_and_numbers):
    results = \
        cat_series_with_nan_and_numbers.cleaner.detect.pattern(pattern=r"cat|dog",
                                                               mode='match',
                                                               case=False
                                                               )
    expected = pd.Series([False, False, False, True, True, False, True])
    assert_series_equal(results.is_error(), expected)


def test_detect_from_existing_object(cat_series_with_nan_and_numbers, cat_series_with_capital):
    results = \
        cat_series_with_nan_and_numbers.cleaner.detect.pattern(pattern=r"[a-z]*", mode='fullmatch')
    results_test = cat_series_with_capital.cleaner.detect(results)
    expected = pd.Series([False, True])
    assert_series_equal(results_test.is_error(), expected)


def test_compiled_regex(cat_series_with_nan_and_numbers):
    regex = re.compile(r"[a-z]*")
    results = \
        cat_series_with_nan_and_numbers.cleaner.detect.pattern(pattern=regex,
                                                               mode='fullmatch'
                                                               )
    expected = pd.Series([True, False, False, False, True, False, False])
    assert_series_equal(results.is_error(), expected)


def test_compiled_regex_with_case(cat_series_with_nan_and_numbers):
    regex = re.compile(r"[a-z]*")
    msg = "case and flag are ignored with a compiled regex"
    with pytest.warns(UserWarning, match=msg):
        results = \
            cat_series_with_nan_and_numbers.cleaner.detect.pattern(pattern=regex,
                                                                mode='fullmatch',
                                                                case=False
                                                                )
    expected = pd.Series([True, False, False, False, True, False, False])
    assert_series_equal(results.is_error(), expected)


def test_compiled_regex_with_flags(cat_series_with_nan_and_numbers):
    regex = re.compile(r"[a-z]*")
    msg = "case and flag are ignored with a compiled regex"
    with pytest.warns(UserWarning, match=msg):
        results = \
            cat_series_with_nan_and_numbers.cleaner.detect.pattern(pattern=regex,
                                                                mode='fullmatch',
                                                                flags=1
                                                                )
    expected = pd.Series([True, False, False, False, True, False, False])
    assert_series_equal(results.is_error(), expected)


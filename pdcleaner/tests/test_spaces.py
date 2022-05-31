"""Tests for `pdcleaner` detecting extra spaces"""

import pytest

import pandas as pd
from pandas.testing import assert_series_equal


def test_detect_call_as_parameter(series_with_extra_spaces):
    detect_results = series_with_extra_spaces.cleaner.detect('spaces')
    assert detect_results.side == 'both'


def test_detect_call_as_method(series_with_extra_spaces):
    detect_results = series_with_extra_spaces.cleaner.detect.spaces(side='left')
    assert detect_results.side == 'left'


def test_detect_with_both_parameter(series_with_extra_spaces):
    detect_results = series_with_extra_spaces.cleaner.detect.spaces(side='both')
    expected = pd.Series([False, True, True, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_left_parameter(series_with_extra_spaces):
    detect_results = series_with_extra_spaces.cleaner.detect.spaces(side='left')
    expected = pd.Series([False, False, True, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_right_parameter(series_with_extra_spaces):
    detect_results = series_with_extra_spaces.cleaner.detect.spaces(side='right')
    expected = pd.Series([False, True, False, True, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_with_unknown_parameter(series_with_extra_spaces):
    msg = "Parameter side must be left or right or both"
    with pytest.raises(ValueError, match=msg):
        series_with_extra_spaces.cleaner.detect.spaces(side='east')


def test_side_from_existing_detector(series_with_extra_spaces):
    detector = series_with_extra_spaces.cleaner.detect.spaces(side='right')
    series = pd.Series([' Paris'])
    detector2 = series.cleaner.detect(detector)
    assert detector2.side == detector.side

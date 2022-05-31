"""Tests for `pdcleaner` detecting values castable into specific type (DataFrame or Series)."""

from unicodedata import decimal
import pytest

import pandas as pd
from pandas.testing import assert_series_equal


def test_detect_call_as_parameter(series_with_different_types):
    detect_results = series_with_different_types.cleaner.detect('castable', target='int')
    assert detect_results.target == 'int'


def test_detect_call_as_method(series_with_different_types):
    detect_results = series_with_different_types.cleaner.detect.castable(target='float',
                                                                         decimal=',')
    assert detect_results.target == 'float'
    assert detect_results.decimal == ','


def test_with_float_parameter(series_with_different_types):
    detect_results = series_with_different_types.cleaner.detect.castable(target='float')
    expected = pd.Series([False, False, True, True])
    assert_series_equal(detect_results.is_error(), expected)


def test_with_int_parameter(series_with_different_types):
    detect_results = series_with_different_types.cleaner.detect.castable(target='int')
    expected = pd.Series([True, False, True, True])
    assert_series_equal(detect_results.is_error(), expected)


def test_with_date_parameter(series_with_different_types):
    detect_results = series_with_different_types.cleaner.detect.castable(target='date')
    expected = pd.Series([True, True, False, True])
    assert_series_equal(detect_results.is_error(), expected)


def test_with_boolean_parameter(series_with_boolean):
    detect_results = series_with_boolean.cleaner.detect('castable',
                                                        target='boolean',
                                                        bool_values={"Yes": True, "No": False})
    expected = pd.Series([False, False, False, False, True, True])
    assert_series_equal(detect_results.is_error(), expected)


def test_with_boolean_parameter_without_params(series_with_boolean):
    detect_results = series_with_boolean.cleaner.detect('castable',
                                                        target='boolean')
    expected = pd.Series([True, True, True, True, True, True])
    assert_series_equal(detect_results.is_error(), expected)


def test_with_unknown_parameter(series_with_different_types):
    legal_values = ["int", "float", "date"]
    msg = f"target must be in {', '.join(legal_values)}"
    with pytest.raises(ValueError, match=msg):
        series_with_different_types.cleaner.detect.castable(target="str")


def test_wihout_parameter(series_with_different_types):
    msg = "Target parameter must be defined"
    with pytest.raises(ValueError, match=msg):
        series_with_different_types.cleaner.detect.castable()


def test_with_separator_parameter(series_with_separate_numbers):
    detect_results = series_with_separate_numbers.cleaner.detect.castable(target='float',
                                                                          thousands=' ',
                                                                          decimal=',')
    expected = pd.Series([False, False, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_warning_separators(series_with_different_types):
    msg = "Thousands/decimal separator parameter is not necessary to check " \
          "if value is castable to date"
    with pytest.raises(ValueError, match=msg):
        series_with_different_types.cleaner.detect.castable(target='date', decimal=',')


def test_with_int_type_series(series_with_integers):
    msg = "This detector is only for object series."
    with pytest.raises(TypeError, match=msg):
        series_with_integers.cleaner.detect.castable(target='int')


def test_detect_with_existing_detector(series_with_different_types):
    detect_results = \
        series_with_different_types.cleaner.detect('castable',
                                                   target='int',
                                                   thousands='_',
                                                   decimal='.')
    series2 = pd.Series(['2'])
    detector2 = series2.cleaner.detect(detect_results)
    assert detector2.target == 'int'
    assert detector2.thousands == '_'
    assert detector2.decimal == '.'


def test_castable_with_integers():
    s = pd.Series(['1', '1.0', '1.1'])
    detector = s.cleaner.detect.castable(target='int')
    expected = pd.Series([False, False, True])
    assert_series_equal(detector.is_error(), expected)


def test_castable_with_only_integers():
    s = pd.Series(['1', '2', '3'])
    detector = s.cleaner.detect.castable(target='int')
    expected = pd.Series([False, False, False])
    assert_series_equal(detector.is_error(), expected)
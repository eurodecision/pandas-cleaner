"""Tests for `pdcleaner` zscore detection method (Series)."""

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal


def test_zscore_default_threshold_property(series_mini_normal):
    detector = series_mini_normal.cleaner.detect.zscore()
    assert detector.threshold == 1.96


def test_zscore_modified_threshold_property(series_mini_normal):
    detector = series_mini_normal.cleaner.detect.zscore(threshold=2.5)
    assert detector.threshold == 2.5


def test_zscore_negative_threshold_raise(series_mini_normal):
    match = "Threshold must be >= 0"
    with pytest.raises(ValueError, match=match):
        series_mini_normal.cleaner.detect.zscore(threshold=-1)
    with pytest.raises(ValueError, match=match):
        series_mini_normal.cleaner.detect('zscore', threshold=-1)


def test_zscore_invalid_dtype_threshold_raise(series_mini_normal):
    match = "Threshold must be a number"
    with pytest.raises(TypeError, match=match):
        series_mini_normal.cleaner.detect.zscore(threshold="str")
    with pytest.raises(TypeError, match=match):
        series_mini_normal.cleaner.detect('zscore', threshold="str")


def test_zscore_on_series_with_nan(series_with_nan):
    """Ensure nan are not considered as outliers."""
    detection_detector = series_with_nan.cleaner.detect.zscore()
    assert not detection_detector.is_error()[0]


def tests_zscore_iserror(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('zscore')
    expected = pd.Series([False] * 8 + [True] * 2)
    assert_series_equal(detector.is_error(), expected)


def tests_zscore_noterror(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('zscore')
    expected = pd.Series([True] * 8 + [False] * 2)
    assert_series_equal(detector.not_error(), expected)


def test_zscore_has_errors(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('zscore')
    assert detector.has_errors()


def test_zscore_n_errors(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('zscore')
    assert detector.n_errors == 2


def test_zscore_properties(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('zscore')
    mean = series_mini_normal.mean()
    std = series_mini_normal.std()
    assert detector.mean == mean
    assert detector.std == series_mini_normal.std()
    assert detector.upper == mean + 1.96 * std
    assert detector.lower == mean - 1.96 * std


def test_zscore_properties_from_existing(series_mini_normal, series_test_normal):
    detector = series_mini_normal.cleaner.detect('zscore')
    detector2 = series_test_normal.cleaner.detect(detector)
    mean = series_mini_normal.mean()
    std = series_mini_normal.std()
    assert detector2.mean == mean
    assert detector2.std == series_mini_normal.std()
    assert detector2.upper == mean + 1.96 * std
    assert detector2.lower == mean - 1.96 * std


def test_zscore_from_existing(series_mini_normal, series_test_normal):
    detector = series_mini_normal.cleaner.detect('zscore')
    detector_test = series_test_normal.cleaner.detect(detector)
    expected = pd.Series([False, True])
    assert_series_equal(detector_test.is_error(), expected)


def test_sided_zscore_on_series(series_mini_normal):
    """Ensure the sided argument works correctly."""
    mean = series_mini_normal.mean()
    std = series_mini_normal.std()
    expected_lower = mean - 1.96 * std  # -5.695
    expected_upper = mean + 1.96 * std  # 5.695

    # by default sided = "both"
    detector = series_mini_normal.cleaner.detect.zscore()
    assert detector.sided == "both"
    assert detector.lower == pytest.approx(expected_lower)
    assert detector.upper == pytest.approx(expected_upper)
    # -6 and 6 are errors
    expected = pd.Series([False] * 8 + [True] * 2)
    assert_series_equal(expected, detector.is_error())

    # sided=="right" => only consider the upper bound
    detector = series_mini_normal.cleaner.detect.zscore(sided="right")
    assert detector.lower == np.NINF
    assert detector.upper == pytest.approx(expected_upper)
    # -6 is no longer an error but 6 is still
    expected = pd.Series([False] * 8 + [False] + [True])
    assert_series_equal(expected, detector.is_error())

    # sided=="left" => only consider the lower bound
    detector = series_mini_normal.cleaner.detect.zscore(sided="left")
    assert detector.lower == pytest.approx(expected_lower)
    assert detector.upper == np.inf
    # 6 is no longer an error but -6 is still
    expected = pd.Series([False] * 8 + [True] + [False])
    assert_series_equal(expected, detector.is_error())


def test_modzscore_default_threshold_property(series_mini_normal):
    assert series_mini_normal.cleaner.detect.modzscore().threshold == 3.5
    assert series_mini_normal.cleaner.detect('modzscore').threshold == 3.5


def test_modzscore_modified_threshold_property(series_mini_normal):
    detector = series_mini_normal.cleaner.detect.modzscore(threshold=2.5)
    assert detector.threshold == 2.5
    assert detector.threshold == 2.5


def test_modzscore_negative_threshold_raise(series_mini_normal):
    match = "Threshold must be >= 0"
    with pytest.raises(ValueError, match=match):
        series_mini_normal.cleaner.detect.modzscore(threshold=-1)
    with pytest.raises(ValueError, match=match):
        series_mini_normal.cleaner.detect('modzscore', threshold=-1)


def test_modzscore_invalid_dtype_threshold_raise(series_mini_normal):
    match = "Threshold must be a number"
    with pytest.raises(TypeError, match=match):
        series_mini_normal.cleaner.detect.modzscore(threshold="str")
    with pytest.raises(TypeError, match=match):
        series_mini_normal.cleaner.detect('modzscore', threshold="str")


def test_modzscore_on_series_with_nan(series_with_nan):
    """Ensure nan are not considered as outliers."""
    detection_detector = series_with_nan.cleaner.detect.modzscore()
    assert not detection_detector.is_error()[0]


def tests_modzscore_iserror(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('modzscore')
    expected = pd.Series([False] * 8 + [True] * 2)
    assert_series_equal(detector.is_error(), expected)


def tests_modzscore_noterror(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('modzscore')
    expected = pd.Series([True] * 8 + [False] * 2)
    assert_series_equal(detector.not_error(), expected)


def test_modzscore_has_errors(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('modzscore')
    assert detector.has_errors()


def test_modzscore_n_errors(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('modzscore')
    assert detector.n_errors == 2


def test_modzscore_properties(series_mini_normal):
    detector = series_mini_normal.cleaner.detect('modzscore')
    med = series_mini_normal.median()
    assert detector.median == med
    mad = (series_mini_normal - med).abs().median()
    assert detector.mad == mad
    assert detector.threshold == 3.5

    assert np.abs(detector.lower - (med - 3.5 * mad / .6475)) < 1e-8
    assert np.abs(detector.upper - (med + 3.5 * mad / .6475)) < 1e-8


def test_modzscore_properties_from_existing_detector(series_mini_normal, series_test_normal):
    detector = series_mini_normal.cleaner.detect('modzscore')
    med = series_mini_normal.median()
    assert series_test_normal.cleaner.detect(detector).median == med
    mad = (series_mini_normal - med).abs().median()
    assert series_test_normal.cleaner.detect(detector).mad == mad
    detector2 = series_test_normal.cleaner.detect(detector)
    assert np.abs(detector2.upper - (med + 3.5 * mad / .6475)) < 1e-8
    assert np.abs(detector2.lower - (med - 3.5 * mad / .6475)) < 1e-8


def test_modzscore_from_existing(series_mini_normal, series_test_normal):
    detector = series_mini_normal.cleaner.detect('modzscore')
    detector_test = series_test_normal.cleaner.detect(detector)
    expected = pd.Series([False, True])
    assert_series_equal(detector_test.is_error(), expected)


def test_sided_modzscore_on_series(series_mini_normal):
    """Ensure the sided argument works correctly."""
    med = series_mini_normal.median()
    mad = (series_mini_normal - med).abs().median()
    expected_lower = med - 3.5 * mad / .6475  # -5.405
    expected_upper = med + 3.5 * mad / .6475  # 5.405

    # by default sided = "both"
    detector = series_mini_normal.cleaner.detect.modzscore()
    assert detector.sided == "both"
    assert detector.lower == pytest.approx(expected_lower)
    assert detector.upper == pytest.approx(expected_upper)
    # -6 and 6 are errors
    expected = pd.Series([False] * 8 + [True] * 2)
    assert_series_equal(expected, detector.is_error())

    # sided=="right" => only consider the upper bound
    detector = series_mini_normal.cleaner.detect.modzscore(sided="right")
    assert detector.lower == np.NINF
    assert detector.upper == pytest.approx(expected_upper)
    # -6 is no longer an error but 6 is still
    expected = pd.Series([False] * 8 + [False] + [True])
    assert_series_equal(expected, detector.is_error())

    # sided=="left" => only consider the lower bound
    detector = series_mini_normal.cleaner.detect.modzscore(sided="left")
    assert detector.lower == pytest.approx(expected_lower)
    assert detector.upper == np.inf
    # 6 is no longer an error but -6 is still
    expected = pd.Series([False] * 8 + [True] + [False])
    assert_series_equal(expected, detector.is_error())

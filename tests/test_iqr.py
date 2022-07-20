"""Tests for `pdcleaner` iqr detection method (series_with_outlier)."""

import pandas as pd
import numpy as np
import pytest

from pandas.testing import assert_series_equal


def test_default_threshold_property(series_with_outlier):
    """ Check default threshold = 1.5"""
    detector = series_with_outlier.cleaner.detect.iqr()
    assert detector.threshold == 1.5
    detector = series_with_outlier.cleaner.detect('iqr')
    assert detector.threshold == 1.5


def test_modified_threshold_property(series_with_outlier):
    """Check modified threshold value"""
    detector = series_with_outlier.cleaner.detect('iqr', threshold=2.5)
    assert detector.threshold == 2.5
    detector = series_with_outlier.cleaner.detect.iqr(threshold=2.5)
    assert detector.threshold == 2.5


def test_negative_threshold_raise(series_with_outlier):
    '''Threshold must be positive'''
    match = "Threshold must be >= 0"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.iqr(threshold=-1)
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect('iqr', threshold=-1)


def test_invalid_dtype_threshold_raise(series_with_outlier):
    """ Threshold must be a number """
    match = "Threshold must be a number"
    with pytest.raises(TypeError, match=match):
        series_with_outlier.cleaner.detect.iqr(threshold="str")
    with pytest.raises(TypeError, match=match):
        series_with_outlier.cleaner.detect('iqr', threshold="str")


def test_iqr_on_series_with_nan(series_with_nan):
    """Ensure nan are not considered as outliers."""
    detector = series_with_nan.cleaner.detect.iqr()
    assert not detector.is_error()[0]


def test_iqr_on_series(series_test_gaussian):
    """Ensure the detector works correctly on a series_with_outlier."""
    detector = series_test_gaussian.cleaner.detect.iqr(threshold=2.)
    assert detector.lower == pytest.approx(-4.25)
    assert detector.upper == pytest.approx(4.5)
    assert detector.iqr == pytest.approx(1.75)
    expected = pd.Series([False] * 3 + [True] + [False] * 4 + [True] * 2)
    assert_series_equal(expected, detector.is_error())


def test_iqr_transfer_bounds_to_copy_detector(series_test_gaussian):
    """Ensure copying a detector transfers the learnt/trained attributes."""
    s_test = pd.Series([-10, 0])
    detector_train = series_test_gaussian.cleaner.detect.iqr(threshold=2.)

    # copy the detector obj => properties are those of the original trained detector
    detector_test = s_test.cleaner.detect(detector_train)
    assert detector_test.q25 == pytest.approx(-0.75)
    assert detector_test.q75 == pytest.approx(1.)
    assert detector_test.lower == pytest.approx(-4.25)
    assert detector_test.upper == pytest.approx(4.5)
    assert detector_test.iqr == pytest.approx(1.75)
    expected = pd.Series([True, False])
    assert_series_equal(expected, detector_test.is_error())

    # creating the detector initializes the properties
    detector_new = s_test.cleaner.detect.iqr(threshold=2.)
    assert detector_new.q25 == pytest.approx(-7.5)
    assert detector_new.q75 == pytest.approx(-2.5)
    assert detector_new.lower == pytest.approx(-17.5)
    assert detector_new.upper == pytest.approx(7.5)
    assert detector_new.iqr == pytest.approx(5)
    expected = pd.Series([False, False])
    assert_series_equal(expected, detector_new.is_error())


def test_sided_iqr_on_series(series_test_gaussian):
    """Ensure the sided argument works correctly."""
    # by default sided = "both"
    detector = series_test_gaussian.cleaner.detect.iqr(threshold=2.)
    assert detector.sided == "both"
    assert detector.lower == pytest.approx(-4.25)
    assert detector.upper == pytest.approx(4.5)
    assert detector.iqr == pytest.approx(1.75)
    # -6, 6 and 100 are errors
    expected = pd.Series([False, False, False, True, False, False, False, False, True, True])
    assert_series_equal(expected, detector.is_error())

    # sided=="right" => only consider the upper bound
    detector = series_test_gaussian.cleaner.detect.iqr(threshold=2., sided="right")
    assert detector.lower == np.NINF
    assert detector.upper == pytest.approx(4.5)
    assert detector.iqr == pytest.approx(1.75)
    # 6 and 100 are errors but not -6
    expected = pd.Series([False, False, False, True] + [False] * 5 + [True])
    assert_series_equal(expected, detector.is_error())

    # sided=="left" => only consider the lower bound
    detector = series_test_gaussian.cleaner.detect.iqr(threshold=2., sided="left")
    assert detector.lower == pytest.approx(-4.25)
    assert detector.upper == np.inf
    assert detector.iqr == pytest.approx(1.75)
    # -6 is an error but 6 and 100 are no longer considered as errors
    expected = pd.Series([False] * 8 + [True, False])
    assert_series_equal(expected, detector.is_error())

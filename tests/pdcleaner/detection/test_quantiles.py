"""UnitTests for :py:class:pdcleaner.detection.gaussian.quantiles"""

import pandas as pd
import pytest

from pandas.testing import assert_series_equal


def test_default_properties(quantiles_series):
    assert quantiles_series.cleaner.detect.quantiles(upperq=0.8).lowerq == 0.
    assert quantiles_series.cleaner.detect.quantiles(lowerq=0.2).upperq == 1.
    assert quantiles_series.cleaner.detect('quantiles', upperq=0.8).lowerq == 0.
    assert quantiles_series.cleaner.detect('quantiles', lowerq=0.2).upperq == 1.


def test_modified_properties(quantiles_series):
    assert quantiles_series.cleaner.detect.quantiles(lowerq=0.25).lowerq == 0.25
    assert quantiles_series.cleaner.detect.quantiles(upperq=0.9).upperq == 0.9
    assert quantiles_series.cleaner.detect('quantiles', lowerq=0.25).lowerq == 0.25
    assert quantiles_series.cleaner.detect('quantiles', upperq=0.9).upperq == 0.9


def test_raise_when_lower_is_bigger_than_upper(quantiles_series):
    match = "Lower quantile is >= upper quantile"
    with pytest.raises(ValueError, match=match):
        quantiles_series.cleaner.detect.quantiles(lowerq=0.7, upperq=0.2)


def test_raise_invalid_dtype_quantiles(quantiles_series):
    match = "lowerq must be a number"
    with pytest.raises(ValueError, match=match):
        quantiles_series.cleaner.detect.quantiles(lowerq="str")
    match = "upperq must be a number"
    with pytest.raises(ValueError, match=match):
        quantiles_series.cleaner.detect('quantiles', upperq="str")

# test quantile not in [0, 1] range


def test_on_series_with_nan(series_with_nan):
    """Ensure nan are not considered as outliers."""
    detector = series_with_nan.cleaner.detect.quantiles(lowerq=0.4, upperq=0.6)
    assert not detector.is_error()[0]


def test_on_series(quantiles_series):
    """Ensure the detector works correctly on a series."""
    detector = quantiles_series.cleaner.detect.quantiles(lowerq=0.2, upperq=0.8)
    assert detector.lower == pytest.approx(2.)
    assert detector.upper == pytest.approx(8.)
    expected = pd.Series([True, False, False, True, True,
                          True, False, False, False, False, False])
    assert_series_equal(expected, detector.is_error())


def test_transfer_bounds_to_copy_detector(quantiles_series):
    """Ensure copying a detector transfers the learnt/trained attributes."""
    s_test = pd.Series([-10, 5, 12])
    detector_train = quantiles_series.cleaner.detect.quantiles(lowerq=0.2, upperq=0.8)

    detector_test = s_test.cleaner.detect(detector_train)
    assert detector_test.lowerq == pytest.approx(0.2)
    assert detector_test.upperq == pytest.approx(0.8)
    assert detector_test.lower == pytest.approx(2.)
    assert detector_test.upper == pytest.approx(8.)
    expected = pd.Series([True, False, True])
    assert_series_equal(expected, detector_test.is_error())


def test_quantile_no_q(quantiles_series):
    """If neither upperq or lowerq is specified"""
    msg = "Neither lower or upper quantile specified"
    with pytest.warns(UserWarning, match=msg):
        quantiles_series.cleaner.detect("quantiles")

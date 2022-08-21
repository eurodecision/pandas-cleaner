"""Tests for `pdcleaner` detection api."""

import pytest
import pdcleaner

def test_version():
    assert isinstance(pdcleaner.__version__, str)


# TestAccessor:


def test_missing_detection_method_raise(series_with_outlier):
    with pytest.raises(ValueError, match="must be provided"):
        series_with_outlier.cleaner.detect()


def test_bad_detection_method_raise(series_with_outlier):
    with pytest.raises(ValueError, match="not a valid detection method"):
        series_with_outlier.cleaner.detect('bad_method')

#Test NumericalSeries

def test_wrong_obj(df_check_col_types):
    with pytest.raises(TypeError, match="This detector applies to Series"):
        df_check_col_types.cleaner.detect("types")

#TestSeriesDetector

def test_not_a_num_series(cat_series):
    with pytest.raises(TypeError, match="This detector applies to numerical Series"):
        cat_series.cleaner.detect('bounded')

def test_not_a_valid_arg(series_with_outlier):
    with pytest.raises(TypeError, match="is not a valid argument for detect"):
        series_with_outlier.cleaner.detect(1)

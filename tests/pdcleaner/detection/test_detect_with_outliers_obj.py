"""Tests for `pdcleaner` detection method (Series)
using an already existing outliers object."""

import pandas as pd
from pandas.testing import assert_series_equal


def get_params(results_obj):
    return {"name": results_obj.name,
            "lower": results_obj.lower,
            "upper": results_obj.upper}


def test_bounds_with_outliers_obj_params(series_with_outlier, series_test_set):
    detect_results_train = series_with_outlier.cleaner.\
        detect.bounded(lower=2, upper=4)
    detect_results_test = series_test_set.cleaner.detect(detect_results_train)
    params_train = get_params(detect_results_train)
    params_test = get_params(detect_results_test)
    assert params_train == params_test


def test_bounds_with_outliers_obj_returns(series_with_outlier, series_test_set):
    outliers_train = series_with_outlier.cleaner.detect.bounded(lower=2, upper=4)
    mask_test = series_test_set.cleaner.detect(outliers_train).is_error()
    assert_series_equal(mask_test, pd.Series([True, False, True]))


def test_iqr_with_outliers_obj_params(series_with_outlier, series_test_set):
    detect_results_train = series_with_outlier.cleaner.detect.iqr(threshold=3)
    detect_results_test = series_test_set.cleaner.detect(detect_results_train)
    params_train = get_params(detect_results_train)
    params_test = get_params(detect_results_test)
    assert params_train == params_test


def test_iqr_with_outliers_obj_returns(series_with_outlier, series_test_set):
    outliers_train = series_with_outlier.cleaner.detect.iqr(threshold=1.5)
    mask_test = series_test_set.cleaner.detect(outliers_train).is_error()
    assert_series_equal(mask_test, pd.Series([False, False, True]))

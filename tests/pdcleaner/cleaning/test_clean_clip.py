
import pandas as pd
from pandas.testing import assert_series_equal


def test_clip_num_series(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean('clip', results)
    expected = pd.Series([1, 2, 10, 4])
    assert_series_equal(cleaned, expected)


def test_clip_num_series_direct_call(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean.clip(results)
    expected = pd.Series([1, 2, 10, 4])
    assert_series_equal(cleaned, expected)


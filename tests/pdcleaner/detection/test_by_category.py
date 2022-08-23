import pytest

import pandas as pd
from pandas.testing import assert_series_equal


def test_iqr_by_cat(df_num_cat):
    results = df_num_cat.cleaner.detect('iqr').is_error()
    expected = pd.Series([False] * 10 + [True] + [False]*10)
    assert_series_equal(results, expected)


def test_with_wrong_detector(df_num_cat):
    detector = df_num_cat.cleaner.detect('iqr')
    with pytest.raises(ValueError, match="This detection method can not be used with"):
        df_num_cat.cleaner.detect(detector)


def test_get_method(df_num_cat):
    detector = df_num_cat.cleaner.detect('iqr')
    assert detector.method == 'iqr'


def test_get_kwargs(df_num_cat):
    kwargs = {'threshold': 3}
    detector = df_num_cat.cleaner.detect('iqr', **kwargs)
    assert detector.method_kwargs == kwargs


def test_with_existing_detector(df_two_cat_cols):
    msg = "Dataframe must contain one numerical column and one categorical column"
    with pytest.raises(TypeError, match=msg):
        df_two_cat_cols.cleaner.detect('iqr')

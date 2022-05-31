import pytest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal


# API


def test_not_a_valid_argument(series_with_outlier):
    detector = series_with_outlier.cleaner.detect('iqr')
    with pytest.raises(TypeError, match="is not a valid argument for clean()"):
        series_with_outlier.cleaner.clean(1, detector)


def test_not_a_valid_method(series_with_outlier):
    detector = series_with_outlier.cleaner.detect('iqr')
    with pytest.raises(ValueError, match="is not a valid cleaning method"):
        series_with_outlier.cleaner.clean('bad_method', detector)


def test_not_a_bounded_detector(cat_series):
    detector = cat_series.cleaner.detect('types', str)
    with pytest.raises(ValueError, match="does not have lower and upper bounds and can not"):
        cat_series.cleaner.clean('clip', detector)


def test_no_keys(cat_series):
    detector = cat_series.cleaner.detect('types', str)
    with pytest.raises(ValueError, match="method does not have a keys dictionary"):
        cat_series.cleaner.clean('bykeys', detector)


# TestCleanSeries:


def test_drop_num_series_direct_call(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean.drop(results)
    expected = pd.Series([1, 2, 4], index=[0, 1, 3])
    assert_series_equal(cleaned, expected)


def test_drop_num_series(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean('drop', results)
    expected = pd.Series([1, 2, 4], index=[0, 1, 3])
    assert_series_equal(cleaned, expected)


def test_replace_num_series(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean('replace', results, value=0)
    expected = pd.Series([1, 2, 0, 4])
    assert_series_equal(cleaned, expected)


def test_replace_num_series_direct_call(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean.replace(results, value=0)
    expected = pd.Series([1, 2, 0, 4])
    assert_series_equal(cleaned, expected)


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


def test_to_na_num_series(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean('to_na', results)
    expected = pd.Series([1, 2, np.nan, 4])
    assert_series_equal(cleaned, expected)


def test_to_na_num_series_direct_call(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    cleaned = series_with_outlier.cleaner.clean.to_na(results)
    expected = pd.Series([1, 2, np.nan, 4])
    assert_series_equal(cleaned, expected)


def test_clean_series_inplace(series_with_outlier):
    results = series_with_outlier.cleaner.detect.bounded(upper=10)
    series_with_outlier.cleaner.clean('clip', results, inplace=True)
    expected = pd.Series([1, 2, 10, 4])
    assert_series_equal(series_with_outlier, expected)


# TestCleanDataFrameColumn:


def test_drop_num_col(df_quanti_quali):
    results = df_quanti_quali['num'].cleaner.detect.bounded(upper=10)
    msg = "Dropping inplace will not modify the DataFrame"
    with pytest.warns(UserWarning, match=msg):
        df_quanti_quali['num'].cleaner.clean('drop', results, inplace=True)


def test_replace_num_col(df_quanti_quali):
    results = df_quanti_quali['num'].cleaner.detect.bounded(upper=10)
    df_quanti_quali['num'].cleaner.clean('replace', results, value=0, inplace=True)
    expected = pd.Series([1, 2, 0, 4, 5], name='num')
    assert_series_equal(df_quanti_quali['num'], expected)


def test_clip_num_col(df_quanti_quali):
    results = df_quanti_quali['num'].cleaner.detect.bounded(upper=10)
    df_quanti_quali['num'].cleaner.clean('clip', results, inplace=True)
    expected = pd.Series([1, 2, 10, 4, 5], name='num')
    assert_series_equal(df_quanti_quali['num'], expected)


def test_to_na_num_col(df_quanti_quali):
    results = df_quanti_quali['num'].cleaner.detect.bounded(upper=10)
    df_quanti_quali['num'].cleaner.clean('to_na', results, inplace=True)
    expected = pd.Series([1, 2, np.nan, 4, 5], name='num')
    assert_series_equal(df_quanti_quali['num'], expected)


# TestBykeys:


def test_bykeys_keycollision(keycol_series):
    results = keycol_series.cleaner.detect.keycollision()
    corrected = keycol_series.cleaner.clean('bykeys', results)
    expected = pd.Series(['Linus Torvalds',
                          'Linus Torvalds',
                          'Linus Torvalds',
                          'Linus Torvalds',
                          'Bill Gates',
                          ])
    assert_series_equal(corrected, expected)


def test_bykeys_keycollision_inplace(keycol_series):
    results = keycol_series.cleaner.detect.keycollision()
    keycol_series.cleaner.clean('bykeys', results, inplace=True)
    expected = pd.Series(['Linus Torvalds',
                          'Linus Torvalds',
                          'Linus Torvalds',
                          'Linus Torvalds',
                          'Bill Gates',
                          ])
    assert_series_equal(keycol_series, expected)


def test_bykeys_from_existing_detector(keycol_series, keycol_test_series):
    results = keycol_series.cleaner.detect.keycollision()
    corrected = keycol_test_series.cleaner.clean('bykeys', results)
    expected = pd.Series(['Linus Torvalds', 'Bill Gates', ])
    assert_series_equal(corrected, expected)


def test_bykeys_from_existing_inplace(keycol_series, keycol_test_series):
    results = keycol_series.cleaner.detect.keycollision()
    keycol_test_series.cleaner.clean('bykeys', results, inplace=True)
    expected = pd.Series(['Linus Torvalds', 'Bill Gates', ])
    assert_series_equal(keycol_test_series, expected)


# TestCleanReplace:


def test_replace_with_value(series_with_nan):
    # pd.Series([np.nan, 2, 100, 3])
    results = series_with_nan.cleaner.detect.bounded(upper=10)
    cleaned = series_with_nan.cleaner.clean('replace', results, value=0)
    expected = pd.Series([np.nan, 2, 0, 3])
    assert_series_equal(cleaned, expected)


def test_replace_inplace(series_with_nan):
    series_obj = series_with_nan
    # pd.Series([np.nan, 2, 100, 3])
    results = series_obj.cleaner.detect.bounded(upper=10)
    series_obj.cleaner.clean('replace', results, value=0, inplace=True)
    expected = pd.Series([np.nan, 2, 0, 3])
    assert_series_equal(series_obj, expected)


def test_replace_with_lambda(series_with_nan):
    # pd.Series([np.nan, 2, 100, 3])
    results = series_with_nan.cleaner.detect.bounded(upper=10)
    cleaned = series_with_nan.cleaner.clean('replace', results, value=lambda s: 5)
    expected = pd.Series([np.nan, 2, 5, 3])
    assert_series_equal(cleaned, expected)


def test_replace_with_dict(series_with_nan):
    # pd.Series([np.nan, 2, 100, 3])
    results = series_with_nan.cleaner.detect.bounded(upper=10)
    cleaned = series_with_nan.cleaner.clean('replace', results, value={100: 1, 99: 9, 2: -2})
    expected = pd.Series([np.nan, 2, 1, 3])
    assert_series_equal(cleaned, expected)


def test_replace_with_incomplete_dict(series_with_nan):
    # pd.Series([np.nan, 2, 100, 3])
    results = series_with_nan.cleaner.detect.bounded(upper=10)
    cleaned = series_with_nan.cleaner.clean('replace', results, value={99: 9, 2: -2})
    expected = pd.Series([np.nan, 2, np.nan, 3])
    assert_series_equal(cleaned, expected)


def test_cast_to_float(series_with_separate_numbers):
    # pd.Series(['100 000', '154,5', '9 000', '250,12'], dtype='object')
    results = series_with_separate_numbers.cleaner.detect.castable(target='float',
                                                                   thousands=' ',
                                                                   decimal=',')
    cleaned = series_with_separate_numbers.cleaner.clean('cast', results)
    expected = pd.Series([100000, 154.5, 9000, 250.12])
    assert_series_equal(cleaned, expected)


def test_cast_to_int(series_with_separate_numbers):
    # pd.Series(['100 000', '154,5', '9 000', '250,12'], dtype='object')
    results = series_with_separate_numbers.cleaner.detect.castable(target='int',
                                                                   thousands=' ',
                                                                   decimal=',')
    cleaned = series_with_separate_numbers.cleaner.clean('cast', results)
    expected = pd.Series([100000, np.nan, 9000, np.nan], dtype='Int32')
    assert_series_equal(cleaned, expected)


def test_cast_to_date(series_with_different_types):
    # pd.Series(['1.05', '154', '15/05/2022', 'Alice'], dtype='object')
    results = series_with_different_types.cleaner.detect.castable(target='date')
    cleaned = series_with_different_types.cleaner.clean('cast', results, format="%d/%m/%Y")
    expected = pd.Series([np.nan, np.nan, '2022-05-15',  np.nan], dtype='datetime64[ns]')
    assert_series_equal(cleaned, expected)


def test_cast_to_boolean(series_with_boolean):
    detect_results = series_with_boolean.cleaner.detect('castable',
                                                        target='boolean',
                                                        bool_values={"Yes": True, "No": False})
    cleaned = series_with_boolean.cleaner.clean('cast', detect_results)
    expected = pd.Series([True, False, False, True, np.nan, np.nan])
    assert_series_equal(cleaned, expected)


def test_cast_with_different_detector(series_with_nan):
    msg = "This cleaning method works only with the castable detector"
    results = series_with_nan.cleaner.detect.bounded(upper=10)
    with pytest.raises(ValueError, match=msg):
        series_with_nan.cleaner.clean('cast', results)


def test_clean_df_drop(df_quanti_quali):
    detector = df_quanti_quali['cat'].cleaner.detect.enum(values=['cat', 'dog'])
    df_clean = df_quanti_quali.cleaner.clean('drop', detector)
    expected = pd.DataFrame({'num': pd.Series([1, 2, 100, 4]),
                             'cat': pd.Series(['cat',
                                               'cat',
                                               'dog',
                                               'dog',
                                               ])})
    assert_frame_equal(df_clean, expected)


def test_clean_strip_left(series_with_extra_spaces):
    detector = series_with_extra_spaces.cleaner.detect.spaces(side='left')
    cleaned = series_with_extra_spaces.cleaner.clean.strip(detector)
    expected = pd.Series(['Paris', 'Paris ', 'Lille', 'Lille ', 'Troyes'])
    assert_series_equal(cleaned, expected)


def test_clean_strip_right(series_with_extra_spaces):
    detector = series_with_extra_spaces.cleaner.detect.spaces(side='right')
    cleaned = series_with_extra_spaces.cleaner.clean.strip(detector)
    expected = pd.Series(['Paris', 'Paris', ' Lille', ' Lille', 'Troyes'])
    assert_series_equal(cleaned, expected)


def test_clean_strip_both(series_with_extra_spaces):
    detector = series_with_extra_spaces.cleaner.detect.spaces(side='both')
    cleaned = series_with_extra_spaces.cleaner.clean.strip(detector)
    expected = pd.Series(['Paris', 'Paris', 'Lille', 'Lille', 'Troyes'])
    assert_series_equal(cleaned, expected)


def test_clean_strip_wrong_detector(series_with_extra_spaces):
    detector = series_with_extra_spaces.cleaner.detect.value(value='Paris')
    msg = 'This cleaning method works only with the spaces detector'
    with pytest.raises(ValueError, match=msg):
        series_with_extra_spaces.cleaner.clean('strip', detector)

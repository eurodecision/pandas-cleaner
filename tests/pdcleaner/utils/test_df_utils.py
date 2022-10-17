"""Unit tests for the utilities functions defined in utils/df_utils
Check if a DataFrame has the required number of numerical
and categorical columns for a given multi-var detector
"""

from pdcleaner.utils.df_utils import (is_numerics_df,
                                      is_cats_df,
                                      is_nums_and_cat_df,
                                      is_twocats_df,
                                      is_twonums_df,
                                      is_num_and_cat_df,
                                      )


def test_df_utils_is_numerics(df_check_col_types):
    df_test = df_check_col_types[['num1', 'num2']]
    assert is_numerics_df(df_test)


def test_df_utils_is_numerics_fail(df_check_col_types):
    df_test = df_check_col_types[['num1', 'obj1']]
    assert ~is_numerics_df(df_test)


def test_df_utils_is_cats(df_check_col_types):
    df_test = df_check_col_types[['obj1', 'obj2']]
    assert is_cats_df(df_test)


def test_df_utils_is_cats_fail(df_check_col_types):
    df_test = df_check_col_types[['num1', 'obj1']]
    assert ~is_cats_df(df_test)


def test_df_utils_is_twocats(df_check_col_types):
    df_test = df_check_col_types[['obj1', 'obj2']]
    assert is_twocats_df(df_test)


def test_df_utils_is_twocats_fail(df_check_col_types):
    df_test = df_check_col_types[['num1', 'obj1', 'obj2']]
    assert ~is_twocats_df(df_test)


def test_df_utils_is_twocats_fail_more_than_two(df_check_col_types):
    df_test = df_check_col_types[['num1', 'obj1', 'obj2']]
    assert ~is_twocats_df(df_test)


def test_df_utils_is_twonums(df_check_col_types):
    df_test = df_check_col_types[['num1', 'num2']]
    assert is_twonums_df(df_test)


def test_df_utils_is_twonums_fail(df_check_col_types):
    df_test = df_check_col_types[['num1', 'num2', 'obj1']]
    assert ~is_twonums_df(df_test)


def test_df_utils_is_twonums_fail_more_than_two(df_check_col_types):
    df_test = df_check_col_types[['num1', 'num1', 'num2']]
    assert ~is_twonums_df(df_test)


def test_df_utils_is_num_and_cat(df_check_col_types):
    df_test = df_check_col_types[['num1', 'obj1']]
    assert is_num_and_cat_df(df_test)


def test_df_utils_is_num_and_cat_fail(df_check_col_types):
    df_test = df_check_col_types[["num1", 'num2']]
    assert ~is_num_and_cat_df(df_test)


def test_df_utils_is_nums_and_cat(df_check_col_types):
    df_test = df_check_col_types[['num1', 'num2', 'obj1']]
    assert is_nums_and_cat_df(df_test)


def test_df_utils_is_nums_and_cat_fail(df_check_col_types):
    df_test = df_check_col_types[['num1', 'num2', 'obj1', 'obj2']]
    assert ~is_nums_and_cat_df(df_test)

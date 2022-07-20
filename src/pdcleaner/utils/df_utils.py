"""
   Utility functions to check if a DataFrame has the required number
   of numerical and categorical columns for a given multi-var detector
"""

import pandas as pd

from pandas.api.types import (is_numeric_dtype,
                              is_object_dtype,
                              is_categorical_dtype,
                              is_datetime64_any_dtype)


def nb_cols_types(df: pd.DataFrame) -> pd.Series:
    """returns the number of columns with types :
    - numeric
    - object or categorical
    - date
    and the total number of columns as a pandas series.
    """
    dict_by_type = {}
    dict_by_type['num'] = df.apply(is_numeric_dtype).sum()
    dict_by_type['cat'] = (df.apply(is_object_dtype).sum()
                           + df.apply(is_categorical_dtype).sum())
    dict_by_type['date'] = df.apply(is_datetime64_any_dtype).sum()
    dict_by_type['total'] = len(df.columns)

    return pd.Series(dict_by_type)


def is_numerics_df(df: pd.DataFrame) -> bool:
    """returns True if the DataFrame contains only numeric columns"""
    nb_types = nb_cols_types(df)
    return nb_types['num'] == nb_types['total']


def is_cats_df(df: pd.DataFrame) -> bool:
    """returns True if the DataFrame contains only categorical or object columns
    """
    nb_types = nb_cols_types(df)
    return nb_types['cat'] == nb_types['total']


def is_twocats_df(df: pd.DataFrame) -> bool:
    """returns True if the DataFrame contains only two categorical or object columns"""
    nb_types = nb_cols_types(df)
    return nb_types['cat'] == nb_types['total'] == 2


def is_twonums_df(df: pd.DataFrame) -> bool:
    """returns True if the DataFrame contains only two categorical or object columns"""
    nb_types = nb_cols_types(df)
    return nb_types['num'] == nb_types['total'] == 2


def is_num_and_cat_df(df: pd.DataFrame) -> bool:
    """returns True if the DataFrame contains only one categorical and one numerical columns"""
    nb_types = nb_cols_types(df)
    nb_num = nb_types['num']
    nb_cat = nb_types['cat']
    nb_total = nb_types['total']
    return (nb_num == 1) and (nb_cat == 1) and (nb_num + nb_cat == nb_total)


def is_nums_and_cat_df(df: pd.DataFrame) -> bool:
    """returns True if the DataFrame contains only 1 categorical and 1 or more numerical columns"""
    nb_types = nb_cols_types(df)
    nb_num = nb_types['num']
    nb_cat = nb_types['cat']
    nb_total = nb_types['total']
    return (nb_cat == 1) and (nb_num + nb_cat == nb_total)

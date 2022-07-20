"""Fixtures"""

import pytest
import numpy as np
import pandas as pd


# series
@pytest.fixture
def series_with_outlier():
    """Create a dummy series with one outlier."""
    return pd.Series([1, 2, 100, 4])


@pytest.fixture
def series_alpha_index():
    """Create a dummy series with one outlier."""
    return pd.Series([1, 2, 100, 4], index=["a", "b", "c", "d"])


@pytest.fixture
def series_unsorted_idx():
    """Create a dummy series with one outlier."""
    return pd.Series([1, 2, 100, 4], index=[12, 3, 5, 1])


@pytest.fixture
def series_with_nan():
    """Create a dummy series with one outlier and a nan."""
    return pd.Series([np.nan, 2, 100, 3])


@pytest.fixture
def dataframe_with_nan():
    """Create a dummy dataframe with nan values"""
    return pd.DataFrame({'col1': ['Alice', np.nan, 'Charles'],
                         'col2': [np.nan, np.nan, 15]})


@pytest.fixture
def series_test_set():
    """Create a small series for detection with series outliers."""
    return pd.Series([5, 3, 100])


@pytest.fixture
def series_test_counts_set():
    """Create a small series of numerical values for
    detection with counts or freq."""
    return pd.Series([5, 3, 3.0, 100, 5])


@pytest.fixture
def series_mini_normal():
    """Create a pseudo-gaussian series with few points"""
    return pd.Series([0, 0, 0, 0, -1, 1, -1, 1, -6, 6])


@pytest.fixture
def series_test_normal():
    """Create a test series where the 1st element is in the gaussian
    and the second an obvious outlier"""
    return pd.Series([0, 100])


@pytest.fixture
def series_test_gaussian():
    """Create a test series with an outlier and gaussian data"""
    return pd.Series([0, 0, 0, 100, -1, 1, -1, 1, -6, 6])


@pytest.fixture
def quantiles_series():
    """Create a test series with an outlier and gaussian data"""
    return pd.Series([-150, 7, 8, 9, 10000, 1, 2, 3, 4, 5, 6])


# Long and short series to test normality

rng = np.random.default_rng(42)


@pytest.fixture
def series_normal():
    """Normal/gaussian series"""
    return pd.Series(rng.normal(0, 1, 1000))


@pytest.fixture
def series_lognormal():
    """Log-normal series"""
    return pd.Series(rng.lognormal(0, 1, 1000))


@pytest.fixture
def series_short():
    return pd.Series([1, 2])


# Categories
@pytest.fixture
def cat_series():
    """Create a categorical series with dogs and cats,
    including one occurence of bird"""
    return pd.Series(['cat', 'cat', 'dog', 'dog', 'dog', 'bird'])


@pytest.fixture
def cat_series_with_nan():
    """Create a categorical series with dogs and cats,
    including one occurence of bird and one missing value"""
    return pd.Series(['cat', 'cat', 'dog', np.nan, 'dog', 'bird'])


@pytest.fixture
def cat_series_test():
    """Create a categorical series with dogs, cats, bird for testing"""
    return pd.Series(['cat', 'dog', 'bird'])


@pytest.fixture
def cat_series_with_nan_and_numbers():
    """Create a categorical series with number, missing value,
    empty string and capital letter for testing"""
    return pd.Series(['Cat', 'cat', 'dog', 'bird', '14', np.nan, ""])


@pytest.fixture
def cat_series_with_capital():
    """Create a categorical series with nthe word
    with and without capital letter for testing"""
    return pd.Series(['dog', 'Dog'])


@pytest.fixture
def keycol_series():
    """Create a categorical series with alternative
    formulations for the same person"""
    return pd.Series(['Linus Torvalds',
                      'linus.torvalds',
                      'Torvalds, Linus',
                      'Linus Torvalds',
                      'Bill Gates',
                      ])


@pytest.fixture
def keycol_series_with_nan():
    """Create a categorical series with alternative
    formulations for the same person and a NaN"""
    return pd.Series(['Linus Torvalds',
                      'Linus Torvalds',
                      'linus.torvalds',
                      'Torvalds, Linus',
                      np.nan
                      ])


@pytest.fixture
def keycol_test_series():
    """Create a categorical series to test on a previously defined detector"""
    return pd.Series(['torvalds LinuS',
                      'Bill Gates'
                      ])


@pytest.fixture
def keycol_plot_series():
    """Create a categorical series to test plotting method"""
    return pd.Series(['Linus Torvalds',
                      'Torvalds, Linus',
                      'Linus Torvalds',
                      'Bill Gates',
                      'Bill Gates',
                      'Steve Jobs',
                      ])


# dataframes
@pytest.fixture
def df_check_col_types():
    """DF with different columns dtypes"""
    df = pd.DataFrame(
        {
            'num1': [1, 2, 3],
            'num2': [np.pi, 2.3, 1],
            'obj1': ['a', 'b', np.nan],
            'obj2': ['A+', 'B', 'C-'],
            'date1': [np.nan, np.nan, pd.to_datetime('2021/07/05')],
        }
    )

    df["obj1"] = df["obj1"].astype("category")
    return df


@pytest.fixture
def df_quanti_quali():
    """Create a dummy series with one outlier."""
    return pd.DataFrame({'num': pd.Series([1, 2, 100, 4, 5]),
                         'cat': pd.Series(['cat',
                                           'cat',
                                           'dog',
                                           'dog',
                                           'bird'
                                           ])})


@pytest.fixture
def df_two_cat_cols():
    """DF with two object columns"""
    return pd.DataFrame({'col1': ['A'] * 10 + ['B'] * 10,
                         'col2': ['a'] * 8 + ['c'] * 2 + ['b'] * 9 + ['a'],
                         })


@pytest.fixture
def df_num_cat():
    """DF with pseudo gaussian num series for two categories and 1 outlier"""
    return pd.DataFrame({
        'col1': [0, 0, 0, 0, -1, 1, -1, 1, -2, 2, 5,
                 6, 6, 6, 6, 5, 7, 5, 7, 4, 8],
        'col2': ['I'] * 11 + ["II"] * 10
    })


@pytest.fixture
def anscombe():
    """Anscombe's quartet comprises four data sets that have nearly identical
    simple descriptive statistics, yet have very different distributions
    and appear very different when graphed.
    https://en.wikipedia.org/wiki/Anscombe%27s_quartet
    """
    x_1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y_1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y_2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    y_3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x_4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y_4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

    return pd.DataFrame({
        'dataset': ['I'] * 11 + ['II'] * 11 + ['III'] * 11 + ['IV'] * 11,
        'x': x_1 * 3 + x_4,
        'y': y_1 + y_2 + y_3 + y_4
    })


@pytest.fixture
def series_with_different_length():
    """Create a dummy serie with different length value"""
    return pd.Series(['75013', '952401', '93250', '94230'])


@pytest.fixture
def series_with_integers():
    """Create a dummy series with different length of integers"""
    return pd.Series([1, 1234567, 1460, 15])


@pytest.fixture
def series_with_floats():
    """Create a dummy series with different length of floats"""
    return pd.Series([1.007, 1.234567, 1.460], dtype='float64')


@pytest.fixture
def series_with_different_types():
    """Create a dummy series with different types"""
    return pd.Series(['1.05', '154', '15/05/2022', 'Alice'], dtype='object')


@pytest.fixture
def series_with_separate_numbers():
    """Create a dummy series containing number with separator"""
    return pd.Series(['100 000', '154,5', '9 000', '250,12'], dtype='object')


@pytest.fixture
def series_with_emails():
    """Create a dummy series with strings representing emails,
    or not"""
    return pd.Series([
        'toto@caramail.com', 'toto@caramail.com_', 'to?to@caramail.com',
        'to.to@caramail.com', 'to,to@caramail.com', 'to,to@cara-mail.com',
        'jAime_589_Les_Frites@yahoo.com', 'toto@caramail.co-m', 'to|to@caramail.com',
        'jaime_589_les-frites@yahoo.com', 'roger', 'gerard@', 'blabla@gmail'])


@pytest.fixture
def series_with_empty_emails():
    """Create a dummy series with strings representing emails,
    empty string, and nan value"""
    return pd.Series([np.nan, 'toto@caramail.com', '', np.nan])


@pytest.fixture
def series_with_urls():
    """Create a dummy series with strings representing emails,
    empty string, and nan value"""
    return pd.Series(['google.com', 'https://www.google.com/', 'https://127.0.0.1:80', 'dummy'])


@pytest.fixture
def series_with_empty_urls():
    """Create a dummy series with strings representing emails,
    empty string, and nan value"""
    return pd.Series([np.nan, ''])


@pytest.fixture
def series_with_extra_spaces():
    """Create a dummy series containing extra spaces in value"""
    return pd.Series(['Paris', 'Paris ', ' Lille', ' Lille ', 'Troyes'])


@pytest.fixture
def dataframe_with_duplicates():
    """Create a dummy dataframe containing duplicate elements """
    return pd.DataFrame({'col1': ['Alice', 'Bob', 'Alice', 'Bob', 'Alice'],
                         'col2': [15, 13, 15, 10, 13]})


@pytest.fixture
def series_with_duplicates():
    """Create a dummy series containing duplicate elements"""
    return pd.Series(['Alice', 'Bob', 'Alice', 'Bob', 'Alice'])


@pytest.fixture
def series_with_boolean():
    """Create a dummy series with boolean"""
    return pd.Series(['Yes', 'No', 'No', 'Yes', 'Ok', 'Nok'], dtype='object')

@pytest.fixture
def series_with_datetime():
    """Create a dummy series containing date elements"""
    return pd.to_datetime(pd.Series(['2022-10-01', '2021-06-11', '2019-04-03', ' 2020-09-25']))


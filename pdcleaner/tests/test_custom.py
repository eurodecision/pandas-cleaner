"""Unit tests for custom detector"""

from http.client import EXPECTATION_FAILED
from unittest import expectedFailure
import pytest
import pandas as pd

from pandas.testing import assert_series_equal


@pytest.fixture
def series_with_a_neg():
    return pd.Series([-1, 2, 3])


@pytest.fixture
def df_squares_one_error():
    return pd.DataFrame({'x' : [1,2,3], 'x2' : [1,3,9] })


def test_series_lambda(series_with_a_neg):
    """on a series with a lambda"""
    detector = \
        series_with_a_neg.cleaner.detect('custom', 
                                         error_func=lambda x: x<0
                                         )
    expected = pd.Series([True, False, False])
    assert_series_equal(expected, detector.is_error())


def func(x: int) -> bool:
    """returns True when there is an error"""
    if x**2 > 5:
        return True
    return False


def test_series_func(series_with_a_neg):
    """on a series with a function"""
    detector = \
        series_with_a_neg.cleaner.detect('custom', 
                                         error_func=func
                                         )
    expected = pd.Series([False, False, True])
    assert_series_equal(expected, detector.is_error())


bad_square = lambda x,y: x**2!=y


def test_dataframe_lambda(df_squares_one_error):
    """On a df with a lambda"""
    detector = \
        df_squares_one_error.cleaner.detect("custom",
                                            error_func=bad_square
                                            )
    expected = pd.Series([False, True, False])
    assert_series_equal(expected, detector.is_error())


def test_error_func_not_callable(series_with_a_neg):
    """error_func must be a callable"""
    with pytest.raises(TypeError, match='error_func sould be a callable'):
        series_with_a_neg.cleaner.detect.custom(error_func=1)


def test_error_arg_number(df_squares_one_error):
    """Check number of args == number of columns"""
    with pytest.raises(ValueError, match='error_func does not have the required number of arguments'):
        df_squares_one_error.cleaner.detect('custom', error_func=func)


def test_returns_bool(series_with_a_neg):
    """Check error_func returns a boolean"""
    with pytest.raises(TypeError, match="error_func must return a boolean"):
        detector = series_with_a_neg.cleaner.detect("custom", error_func=lambda x: 'A')
        detector.index


def test_error_func_defined(series_with_a_neg):
    """ Check if error_func is given"""
    with pytest.raises(ValueError, match='error_func must be defined'):
        series_with_a_neg.cleaner.detect('custom')


def test_from_existing_detector(series_with_a_neg):
    """retrieve error_func from a previously defined detector"""
    f = lambda x: x<0
    detector = \
        series_with_a_neg.cleaner.detect.custom(error_func=f)
    series2 = pd.Series([-1])
    detector2 = series2.cleaner.detect(detector)
    expected = pd.Series([True])
    assert_series_equal(expected, detector2.is_error())
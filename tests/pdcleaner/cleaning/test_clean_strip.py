
import pytest

import pandas as pd
from pandas.testing import assert_series_equal


def test_clean_strip_leading(series_with_extra_spaces):
    detector = series_with_extra_spaces.cleaner.detect.spaces(side='leading')
    cleaned = series_with_extra_spaces.cleaner.clean.strip(detector)
    expected = pd.Series(['Paris', 'Paris ', 'Lille', 'Lille ', 'Troyes'])
    assert_series_equal(cleaned, expected)


def test_clean_strip_trailing(series_with_extra_spaces):
    detector = series_with_extra_spaces.cleaner.detect.spaces(side='trailing')
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

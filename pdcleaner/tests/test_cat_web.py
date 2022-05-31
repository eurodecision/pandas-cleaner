"""Tests for `pdcleaner` detection method (Series)
that detects if categorical/qualitative values match a given pattern."""

import pytest
import pandas as pd
from pandas.testing import assert_series_equal


def test_numerical_series_raise(series_with_outlier):
    r"""Fails with numerical series"""
    with pytest.raises(TypeError, match="applies to categorical/string Series"):
        series_with_outlier.cleaner.detect('email')


def test_emails(series_with_emails):
    r"""
    Test with various emails
    Input: d.Series([
                    'toto@caramail.com', 'toto@caramail.com_', 'to?to@caramail.com',
                    'to.to@caramail.com', 'to,to@caramail.com', 'to,to@cara-mail.com',
                    'jAime_589_Les_Frites@yahoo.com', 'toto@caramail.co-m', 'to|to@caramail.com',
                    'jaime_589_les-frites@yahoo.com', 'roger', 'gerard@', 'blabla@gmail'])
    Output: [
                    False, True, True,
                    False, True, True,
                    False, True, True,
                    False, True, True, True]
    """
    results = series_with_emails.cleaner.detect.email()
    expected = pd.Series([
        False, True, True,
        False, True, True,
        False, True, True,
        False, True, True, True])
    assert_series_equal(results.is_error(), expected)


def test_empty_emails(series_with_empty_emails):
    r"""
    Test with empty emails
    Input: pd.Series(np.nan, 'toto@caramail.com', '', np.nan)
    Output: [False, False, True, False]
    """
    results = series_with_empty_emails.cleaner.detect.email()
    results.report()
    expected = pd.Series([False, False, True, False])
    assert_series_equal(results.is_error(), expected)


def test_url_numerical_series_raise(series_with_outlier):
    r"""Fails with numerical series"""
    with pytest.raises(TypeError, match="applies to categorical/string Series"):
        series_with_outlier.cleaner.detect('url')


def test_urls_protocol_mandatory(series_with_urls):
    r"""
    Test with urls with mandatory protocol
    Input: pd.Series(['google.com','https://www.google.com/', 'https://127.0.0.1:80', 'dummy'])
    Output: [True, False, False, True]
    """
    results = series_with_urls.cleaner.detect.url()
    expected = pd.Series([True, False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_urls_protocol_optional(series_with_urls):
    r"""
    Test with urls with optional protocol
    Input: pd.Series(['google.com','https://www.google.com/', 'https://127.0.0.1:80', 'dummy'])
    Output: [False, False, False, True]
    """
    results = series_with_urls.cleaner.detect.url(check_protocol=False)
    expected = pd.Series([False, False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_urls_from_detector(series_with_urls):
    r"""
    Test with urls with a previously defined detector
    Input: pd.Series(['google.com','https://www.google.com/', 'https://127.0.0.1:80', 'dummy'])
    Output: [False, False, False, True]
    """
    results_before = series_with_urls.cleaner.detect.url(check_protocol=False)
    results = series_with_urls.cleaner.detect(results_before)
    expected = pd.Series([False, False, False, True])
    assert_series_equal(results.is_error(), expected)


def test_urls_empty(series_with_empty_urls):
    r"""
    Test with empty urls
    Input: pd.Series([np.nan(), ''])
    Output: [False, True]
    """
    results = series_with_empty_urls.cleaner.detect.url()
    results.report()
    expected = pd.Series([False, True])
    assert_series_equal(results.is_error(), expected)


def test_ping_numerical_series_raise(series_with_outlier):
    r"""Fails with numerical series"""
    with pytest.raises(TypeError, match="applies to categorical/string Series"):
        series_with_outlier.cleaner.detect('ping')


def test_ping(series_with_urls):
    r"""
    Test with urls
    Input: pd.Series(['google.com','https://www.google.com/', 'https://127.0.0.1:80', 'dummy'])
    Output: [True, False, True, True]
    """
    results = series_with_urls.cleaner.detect.ping()
    expected = pd.Series([True, False, True, True])
    assert_series_equal(results.is_error(), expected)


def test_ping_empty(series_with_empty_urls):
    r"""
    Test with empty urls
    Input: pd.Series([np.nan(), ''])
    Output: [False, True]
    """
    results = series_with_empty_urls.cleaner.detect.ping()
    results.report()
    expected = pd.Series([False, True])
    assert_series_equal(results.is_error(), expected)

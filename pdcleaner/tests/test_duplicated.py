"""Tests for `pdcleaner` duplicated detection method. """

import pytest

import pandas as pd
from pandas.testing import assert_series_equal


def test_detect_call_as_parameter(dataframe_with_duplicates):
    detect_results = dataframe_with_duplicates.cleaner.detect('duplicated', subset='col1',
                                                              keep='first')
    assert detect_results.subset == 'col1'
    assert detect_results.keep == 'first'


def test_detect_call_as_method(dataframe_with_duplicates):
    detect_results = dataframe_with_duplicates.cleaner.detect.duplicated(subset='col1',
                                                                         keep='first')
    assert detect_results.subset == 'col1'
    assert detect_results.keep == 'first'


def test_detect_without_parameters(dataframe_with_duplicates):
    detect_results = dataframe_with_duplicates.cleaner.detect.duplicated()
    expected = pd.Series([False, False, True, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_first(dataframe_with_duplicates):
    detect_results = dataframe_with_duplicates.cleaner.detect.duplicated(subset='col1',
                                                                         keep='first')
    expected = pd.Series([False, False, True, True, True])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_last(dataframe_with_duplicates):
    detect_results = dataframe_with_duplicates.cleaner.detect.duplicated(subset='col1',
                                                                         keep='last')
    expected = pd.Series([True, True, True, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_false(dataframe_with_duplicates):
    detect_results = dataframe_with_duplicates.cleaner.detect.duplicated(subset='col1',
                                                                         keep=False)
    expected = pd.Series([True, True, True, True, True])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_subsets(dataframe_with_duplicates):
    detect_results = dataframe_with_duplicates.cleaner.detect.duplicated(subset=['col1', 'col2'])
    expected = pd.Series([False, False, True, False, False])
    assert_series_equal(detect_results.is_error(), expected)


def test_detect_with_series(series_with_duplicates):
    detect_results = series_with_duplicates.cleaner.detect.duplicated()
    expected = pd.Series([False, False, True, True, True])
    assert_series_equal(detect_results.is_error(), expected)


def test_duplicated_from_existing_detector(dataframe_with_duplicates):
    detect_results = dataframe_with_duplicates.cleaner.detect.duplicated(subset='col1',
                                                                         keep='first')
    detector2 = dataframe_with_duplicates.cleaner.detect(detect_results)
    assert detect_results.subset == detector2.subset
    assert detect_results.keep == detector2.keep

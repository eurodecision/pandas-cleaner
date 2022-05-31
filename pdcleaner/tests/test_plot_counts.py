"""Module to test plotting method associated with a value counts detector
"""

import pytest
import matplotlib.pyplot as plt


def test_plot_counts_nfirst_not_int(cat_series_with_nan):
    match = "should be an integer"
    detector = cat_series_with_nan.cleaner.detect.counts(n=1)
    with pytest.raises(TypeError, match=match):
        detector.plot(nfirst='one')


def test_plot_counts_nlast_not_int(cat_series_with_nan):
    match = "should be an integer"
    detector = cat_series_with_nan.cleaner.detect.counts(n=1)
    with pytest.raises(TypeError, match=match):
        detector.plot(nlast='one')


def test_plot_counts_nfirst_negative(cat_series_with_nan):
    match = "should be >=0"
    detector = cat_series_with_nan.cleaner.detect.counts(n=1)
    with pytest.raises(ValueError, match=match):
        detector.plot(nfirst=-1)


def test_plot_counts_nlast_negative(cat_series_with_nan):
    match = "should be >=0"
    detector = cat_series_with_nan.cleaner.detect.counts(n=1)
    with pytest.raises(ValueError, match=match):
        detector.plot(nlast=-3)


def assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text):
    for cnt, text_properties in enumerate(ax.get_yticklabels()):
        assert text_properties.get_text() == expected_ytickslabels[cnt]

    for cnt, text_properties in enumerate(ax.texts):
        assert text_properties.get_text() == expected_text[cnt]


def test_plot_counts_no_arg(cat_series_with_nan):
    detector = \
        cat_series_with_nan.cleaner.detect('counts', n=1)
    ax = detector.plot()
    expected_ytickslabels = ["cat", "dog", "bird"]
    expected_text = []
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_plot_counts_nfirst_and_nlast_zeros(cat_series_with_nan):
    detector = \
        cat_series_with_nan.cleaner.detect('counts', n=1)
    ax = detector.plot(nfirst=0, nlast=0)
    expected_ytickslabels = ["cat", "dog", "bird"]
    expected_text = []
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_plot_counts_nfirst_only(cat_series_with_nan):
    detector = \
        cat_series_with_nan.cleaner.detect('counts', n=1)
    ax = detector.plot(nfirst=2)
    expected_ytickslabels = ["cat", "dog", "1"]
    expected_text = ["    +1    "]
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_plot_counts_nlast_only(cat_series_with_nan):
    detector = \
        cat_series_with_nan.cleaner.detect('counts', n=1)
    ax = detector.plot(nlast=2)
    expected_ytickslabels = ["1", "dog", "bird"]
    expected_text = ["    +1    "]
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_plot_counts_nfirst_and_nlast(cat_series_with_nan):
    detector = \
        cat_series_with_nan.cleaner.detect('counts', n=1)
    ax = detector.plot(nfirst=1, nlast=1)
    expected_ytickslabels = ["cat", "1", "bird"]
    expected_text = ["    +1    "]
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_plot_counts_nfirst_and_nlast_sup_len_df(cat_series_with_nan):
    detector = \
        cat_series_with_nan.cleaner.detect('counts', n=1)
    ax = detector.plot(nfirst=3, nlast=2)
    expected_ytickslabels = ["cat", "dog", "bird"]
    expected_text = []
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()

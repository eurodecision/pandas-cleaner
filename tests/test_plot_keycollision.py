"""Module to test plotting method associated with a key collision detector
"""

import pytest
import matplotlib.pyplot as plt


def test_nfirst_not_int(keycol_plot_series):
    match = "should be an integer"
    detector = keycol_plot_series.cleaner.detect.keycollision()
    with pytest.raises(TypeError, match=match):
        detector.plot(nfirst='one')


def test_nlast_not_int(keycol_plot_series):
    match = "should be an integer"
    detector = keycol_plot_series.cleaner.detect.keycollision()
    with pytest.raises(TypeError, match=match):
        detector.plot(nlast='one')


def test_nfirst_negative(keycol_plot_series):
    match = "should be >=0"
    detector = keycol_plot_series.cleaner.detect.keycollision()
    with pytest.raises(ValueError, match=match):
        detector.plot(nfirst=-1)


def test_nlast_negative(keycol_plot_series):
    match = "should be >=0"
    detector = keycol_plot_series.cleaner.detect.keycollision()
    with pytest.raises(ValueError, match=match):
        detector.plot(nlast=-3)


def assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text):
    for cnt, text_properties in enumerate(ax.get_yticklabels()):
        assert text_properties.get_text() == expected_ytickslabels[cnt]

    for cnt, text_properties in enumerate(ax.texts):
        assert text_properties.get_text() == expected_text[cnt]


def test_no_arg(keycol_plot_series):
    detector = \
        keycol_plot_series.cleaner.detect('keycollision', keys='fingerprint')
    ax = detector.plot()
    expected_ytickslabels = ["Steve Jobs", "Bill Gates", "Linus Torvalds"]
    expected_text = []
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_nfirst_and_nlast_zeros(keycol_plot_series):
    detector = \
        keycol_plot_series.cleaner.detect('keycollision', keys='fingerprint')
    ax = detector.plot(nfirst=0, nlast=0)
    expected_ytickslabels = ["Steve Jobs", "Bill Gates", "Linus Torvalds"]
    expected_text = []
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_nfirst_only(keycol_plot_series):
    detector = \
        keycol_plot_series.cleaner.detect('keycollision', keys='fingerprint')
    ax = detector.plot(nfirst=2)
    expected_ytickslabels = ["1", "Bill Gates", "Linus Torvalds"]
    expected_text = ["    +1    "]
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_nlast_only(keycol_plot_series):
    detector = \
        keycol_plot_series.cleaner.detect('keycollision', keys='fingerprint')
    ax = detector.plot(nlast=2)
    expected_ytickslabels = ["Steve Jobs", "Bill Gates", "1"]
    expected_text = ["    +1    "]
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_nfirst_and_nlast(keycol_plot_series):
    detector = \
        keycol_plot_series.cleaner.detect('keycollision', keys='fingerprint')
    ax = detector.plot(nfirst=1, nlast=1)
    expected_ytickslabels = ["Steve Jobs", "1", "Linus Torvalds"]
    expected_text = ["    +1    "]
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()


def test_nfirst_and_nlast_sup_len_df(keycol_plot_series):
    detector = \
        keycol_plot_series.cleaner.detect('keycollision', keys='fingerprint')
    ax = detector.plot(nfirst=3, nlast=2)
    expected_ytickslabels = ["Steve Jobs", "Bill Gates", "Linus Torvalds"]
    expected_text = []
    assert_ytickslabels_and_text(ax, expected_ytickslabels, expected_text)
    plt.close()

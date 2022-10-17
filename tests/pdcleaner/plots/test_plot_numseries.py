"""
Module to test plotting detected errors

Note that the order of execution of the plotting commands affects
the order of present lines/collections on the axes
and thus affects the testing
"""

import pytest

import numpy as np
from pandas.testing import assert_series_equal
from matplotlib import colors
import matplotlib.pyplot as plt

from tests.tests_utils.utils import plt_collection_to_series
from tests.tests_utils.utils import assert_listoflist_near

# Test numeric plots
import matplotlib; matplotlib.use('agg')

def test_with_bounded(series_test_set):
    axs = series_test_set.cleaner.detect.bounded(lower=0, upper=10).plot()
    assert len(axs) == 4
    assert len(axs[0].collections) == 1
    assert np.shape(axs[0].collections[0].get_offsets().data.T) == (2, 3)
    plt.close()


def test_with_iqr(series_test_set):
    axs = series_test_set.cleaner.detect.iqr(threshold=1.5).plot()
    assert len(axs) == 4
    assert len(axs[0].collections) == 1
    assert np.shape(axs[0].collections[0].get_offsets().data.T) == (2, 3)
    plt.close()


def test_with_values(cat_series):
    match = "'enum' object has no attribute 'plot'"
    vals = ["cat", "dog"]
    with pytest.raises(AttributeError, match=match):
        cat_series.cleaner.detect.enum(values=vals).plot()
    plt.close()


def test_no_upper_limit(series_test_set):
    axs = series_test_set.cleaner.detect.bounded(lower=10).plot()
    assert len(axs) == 4
    assert len(axs[0].collections) == 1
    assert np.shape(axs[0].collections[0].get_offsets().data.T) == (2, 3)
    plt.close()


def test_no_lower_limit(series_test_set):
    axs = series_test_set.cleaner.detect.bounded(upper=10).plot()
    assert len(axs) == 4
    assert len(axs[0].collections) == 1
    assert np.shape(axs[0].collections[0].get_offsets().data.T) == (2, 3)
    plt.close()


def test_all_error(series_test_set):
    axs = \
        series_test_set.cleaner.detect.bounded(lower=-5000, upper=0).plot()
    assert len(axs) == 4
    assert len(axs[0].collections) == 1
    assert np.shape(axs[0].collections[0].get_offsets().data.T) == (2, 3)
    plt.close()


def test_all_correct(series_test_set):
    axs = \
        series_test_set.cleaner.detect.bounded(lower=0, upper=5000).plot()
    assert len(axs) == 4
    assert len(axs[0].collections) == 1
    assert np.shape(axs[0].collections[0].get_offsets().data.T) == (2, 3)
    plt.close()


def test_alphabetic_index(series_alpha_index):
    axs = \
        series_alpha_index.cleaner.detect.bounded(lower=0, upper=5000).plot()
    assert len(axs) == 4
    assert len(axs[0].collections) == 1
    assert np.shape(axs[0].collections[0].get_offsets().data.T) == (2, 4)
    plt.close()


def test_non_linear_index(series_unsorted_idx):
    detector = series_unsorted_idx.cleaner.detect.bounded(lower=0, upper=5000)
    axs = detector.plot()
    assert len(axs) == 4
    assert len(axs[0].collections) == 1
    assert np.shape(axs[0].collections[0].get_offsets().data.T) == (2, 4)
    plt.close()


# Test scatter plot numeric


def test_simple(series_test_set):
    axs = series_test_set.cleaner.detect.bounded(lower=0, upper=10).plot()
    scattered_series = plt_collection_to_series(axs[0].collections[0])
    expected = series_test_set
    assert_series_equal(scattered_series,
                        expected,
                        check_dtype=False,
                        check_index_type=False
                        )
    plt.close()


def test_color(series_test_set):
    # color is green by default for correct values and red for errors
    axs = series_test_set.cleaner.detect.bounded(lower=0, upper=10).plot()
    scatter_points_colors = axs[0].collections[0].get_facecolors().tolist()
    # may require to plt.draw() before
    expected_colors = list(map(colors.to_rgba, ["green", "green", "red"]))
    assert_listoflist_near(scatter_points_colors, expected_colors)

    axs = series_test_set.cleaner.detect.bounded(lower=0, upper=10).plot(color="blue")
    scatter_points_colors = axs[0].collections[0].get_facecolors().tolist()
    expected_colors = list(map(colors.to_rgba, ["blue", "blue", "red"]))
    assert_listoflist_near(scatter_points_colors, expected_colors)

    axs = series_test_set.cleaner.detect.bounded(lower=0, upper=10).plot(errors_color="blue")
    scatter_points_colors = axs[0].collections[0].get_facecolors().tolist()
    expected_colors = list(map(colors.to_rgba, ["green", "green", "blue"]))
    assert_listoflist_near(scatter_points_colors, expected_colors)

    plt.close("all")


def test_compact(series_test_set):
    axs = series_test_set.cleaner.detect.bounded(lower=0, upper=10).plot(compact=True)
    scattered_series = plt_collection_to_series(axs[0].collections[0])

    expected = series_test_set
    # note : the value 100, will not be visible in the scatterplot
    # , yet it exists in the axes data
    # we can filter using axes limits (on scattered_series)
    #  and bounds (on expected) here
    assert_series_equal(scattered_series,
                        expected,
                        check_dtype=False,
                        check_index_type=False
                        )
    plt.close()


def test_with_nan(series_with_nan):
    axs = series_with_nan.cleaner.detect.bounded(lower=0, upper=10).plot()
    scattered_series = plt_collection_to_series(axs[0].collections[0])

    expected = series_with_nan.dropna()
    assert_series_equal(scattered_series,
                        expected,
                        check_dtype=False,
                        check_index_type=False
                        )

    plt.close()


# Test the horizontal lines on the first three plots (scatter, histogram and kde)


def test_no_limits(series_test_set):
    """On the first three plots (scatter, hist and kde)
    make sure no limiting lines are drawn when limits is set to False"""
    axs = series_test_set.cleaner.detect.bounded(lower=0, upper=10).plot(limits=False)

    assert len(axs) == 4
    for ax_i in axs[:3]:
        assert len(ax_i.lines) == 0

    plt.close()


def test_upper_limits(series_test_set):
    """On the first three plots (scatter, hist and kde),
    make sure upper limit can be drawn correctly """
    axs = series_test_set.cleaner.detect.bounded(upper=6).plot(limits=True)
    assert len(axs) == 4
    for ax_i in axs[:3]:
        assert len(ax_i.lines) == 1

        limit_line = ax_i.lines[0].get_xydata().tolist()
        expected = [[0, 6], [1, 6]]
        assert_listoflist_near(limit_line, expected)

    plt.close()


def test_lower_limits(series_test_set):
    """On the first three plots (scatter, hist and kde),
    make sure lower limit can be drawn correctly """
    axs = series_test_set.cleaner.detect.bounded(upper=0).plot(limits=True)
    assert len(axs) == 4
    for ax_i in axs[:3]:
        assert len(ax_i.lines) == 1

        limit_line = ax_i.lines[0].get_xydata().tolist()
        expected = [[0, 0], [1, 0]]
        assert_listoflist_near(limit_line, expected)

    plt.close()


def test_limits(series_test_gaussian):
    """On the first three plots (scatter, hist and kde),
    make sure that two limiting lines are drawn when option limits is True"""
    lower, upper = -3.375, 3.625
    axs = series_test_gaussian.cleaner.detect.iqr(sided="both").plot(limits=True)
    assert len(axs) == 4
    for ax_i in axs[:3]:
        assert len(ax_i.lines) == 2

        # lower limit
        limit_line = ax_i.lines[0].get_xydata().tolist()
        expected = [[0, lower], [1, lower]]
        assert_listoflist_near(limit_line, expected)

        # upper limit
        limit_line = ax_i.lines[1].get_xydata().tolist()
        expected = [[0, upper], [1, upper]]
        assert_listoflist_near(limit_line, expected)

    plt.close()


def test_right_sided_limits(series_test_gaussian):
    """On the first three plots (scatter, hist and kde),
    make sure that a single line is drawn
    when using a right sided detector with limits set to True
    """
    axs = series_test_gaussian.cleaner.detect.iqr(sided="right").plot(limits=True)
    upper = 3.625
    assert len(axs) == 4
    for ax_i in axs[:3]:
        # only a lower limit
        assert len(ax_i.lines) == 1
        limit_line = ax_i.lines[0].get_xydata().tolist()
        expected = [[0, upper], [1, upper]]
        assert_listoflist_near(limit_line, expected)

    plt.close()


def test_left_sided_limits(series_test_gaussian):
    """On the first three plots (scatter, hist and kde),
    make sure that a single line is drawn
    when using a left sided detector with limits set to True
    """
    axs = series_test_gaussian.cleaner.detect.iqr(sided="left").plot(limits=True)
    lower = -3.375
    assert len(axs) == 4
    for ax_i in axs[:3]:
        # only an upper limit
        assert len(ax_i.lines) == 1
        limit_line = ax_i.lines[0].get_xydata().tolist()
        expected = [[0, lower], [1, lower]]
        assert_listoflist_near(limit_line, expected)

    plt.close()


def test_limits_boxplot(series_test_set):
    """
    check that limits are drawn on the scatter plot when limits is set to True

    Notes
    -----
    box plot already has some drawn lines that are not the limits
    """

    detector = series_test_set.cleaner.detect.bounded(lower=4, upper=6)
    boxplot_axs = detector.plot(limits=True)[3]
    assert len(boxplot_axs.lines) > 2

    # lower limit
    limit_line = boxplot_axs.lines[-2].get_xydata().tolist()
    expected = [[0, 4], [1, 4]]
    assert_listoflist_near(limit_line, expected)

    # upper limit
    limit_line = boxplot_axs.lines[-1].get_xydata().tolist()
    expected = [[0, 6], [1, 6]]
    assert_listoflist_near(limit_line, expected)

    plt.close()


# test texts
# series : pd.Series([5, 3, 100])
# series_with_nan : pd.Series([np.nan, 2, 100, 3]


def test_upper_text(series_test_set):
    detector = series_test_set.cleaner.detect.bounded(upper=6)
    scatter_axs = detector.plot(limits=False, compact=True)[0]
    assert len(scatter_axs.texts) == 2
    expected_texts = [f"max: {float(100):.3}", "1"]
    for cnt, text_properties in enumerate(scatter_axs.texts):
        assert text_properties.get_text() == expected_texts[cnt]

    plt.close()


def test_lower_text(series_test_set):
    detector = series_test_set.cleaner.detect.bounded(lower=10)
    scatter_axs = detector.plot(limits=False, compact=True)[0]
    assert len(scatter_axs.texts) == 2
    expected_texts = [f"min: {float(3):.3}", "2"]
    for cnt, text_properties in enumerate(scatter_axs.texts):
        assert text_properties.get_text() == expected_texts[cnt]

    plt.close()

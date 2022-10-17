"""
Module to test categories associations plotting function
"""

import matplotlib.pyplot as plt


def test_xaxis(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect('associations', count=3)
    ax = detector.plot()
    assert (ax.get_xlim() == (0, df_two_cat_cols.iloc[:, 1].nunique()))
    plt.close()


def test_yaxis(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect('associations', count=3)
    ax = detector.plot()
    assert (ax.get_ylim() == (df_two_cat_cols.iloc[:, 0].nunique(), 0))
    plt.close()


def test_xtickslabels(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect('associations', count=3)
    ax = detector.plot()
    assert set([t.get_text() for t in ax.get_xticklabels()]) \
           == set(df_two_cat_cols.iloc[:, 1].unique())
    plt.close()


def test_ytickslabels(df_two_cat_cols):
    detector = df_two_cat_cols.cleaner.detect('associations', count=3)
    ax = detector.plot()
    assert set([t.get_text() for t in ax.get_yticklabels()]) \
           == set(df_two_cat_cols.iloc[:, 0].unique())
    plt.close()

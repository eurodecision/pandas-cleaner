"""Test transformation for non-gaussian distribution with detection methods iqr,
zscore and modzscore"""

import pytest

import pandas as pd

from scipy.stats import boxcox, yeojohnson
from pandas.testing import assert_series_equal

from pdcleaner.detection.gaussian import _inverse_boxcox, _inverse_yeojohnson


def test_iqr_invalid_transform(series_with_outlier):
    """transform must be None, boxcox or yeojohnson"""
    match = "transform must be"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.iqr(transform='anything')


def test_zscore_invalid_transform(series_with_outlier):
    """transform must be None, boxcox or yeojohnson"""
    match = "transform must be"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.zscore(transform='anything')


def test_modzscore_invalid_transform(series_with_outlier):
    """transform must be None, boxcox or yeojohnson"""
    match = "transform must be"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.modzscore(transform='anything')


def test_iqr_invalid_normaltest(series_with_outlier):
    """normaltest must be ignore, warn or error"""
    match = "normaltest must be"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.iqr(normaltest='anything')


def test_zscore_invalid_normaltest(series_with_outlier):
    """normaltest must be ignore, warn or error"""
    match = "normaltest must be"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.zscore(normaltest='anything')


def test_modzscore_invalid_normaltest(series_with_outlier):
    """normaltest must be ignore, warn or error"""
    match = "normaltest must be"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.modzscore(normaltest='anything')


def test_iqr_wrongtype_pvalue(series_with_outlier):
    """pvalue must be a number"""
    match = "pvalue must be a number"
    with pytest.raises(TypeError, match=match):
        series_with_outlier.cleaner.detect.iqr(pvalue='anything')


def test_zscore_wrongtype_pvalue(series_with_outlier):
    """pvalue must be a number"""
    match = "pvalue must be a number"
    with pytest.raises(TypeError, match=match):
        series_with_outlier.cleaner.detect.zscore(pvalue='anything')


def test_modzscore_wrongtype_pvalue(series_with_outlier):
    """pvalue must be a number"""
    match = "pvalue must be a number"
    with pytest.raises(TypeError, match=match):
        series_with_outlier.cleaner.detect.modzscore(pvalue='anything')


def test_iqr_negative_pvalue(series_with_outlier):
    """pvalue must be positive"""
    match = "pvalue must be positive"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.iqr(pvalue=-1)


def test_zscore_negative_pvalue(series_with_outlier):
    """pvalue must be positive"""
    match = "pvalue must be positive"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.zscore(pvalue=-1)


def test_modzscore_negative_pvalue(series_with_outlier):
    """pvalue must be positive"""
    match = "pvalue must be positive"
    with pytest.raises(ValueError, match=match):
        series_with_outlier.cleaner.detect.modzscore(pvalue=-1)


def test_iqr_transform_params_from_existing(series_with_outlier, series_test_set):
    """params from detector"""
    detector = series_with_outlier.cleaner.detect.iqr(transform='boxcox',
                                                      normaltest='warn',
                                                      pvalue=1e-2)
    detector2 = series_test_set.cleaner.detect(detector)
    assert (detector2.transform, detector2.normaltest, detector2.pvalue) == \
        ('boxcox', 'warn', 1e-2)


def test_zscore_transform_params_from_existing(series_with_outlier, series_test_set):
    """params from detector"""
    detector = series_with_outlier.cleaner.detect.zscore(transform='boxcox',
                                                         normaltest='warn',
                                                         pvalue=1e-2)
    detector2 = series_test_set.cleaner.detect(detector)
    assert (detector2.transform, detector2.normaltest, detector2.pvalue) == \
        ('boxcox', 'warn', 1e-2)


def test_modzscore_transform_params_from_existing(series_with_outlier, series_test_set):
    """params from detector"""
    detector = series_with_outlier.cleaner.detect.modzscore(transform='boxcox',
                                                            normaltest='warn',
                                                            pvalue=1e-2)
    detector2 = series_test_set.cleaner.detect(detector)
    assert (detector2.transform, detector2.normaltest, detector2.pvalue) == \
        ('boxcox', 'warn', 1e-2)


def test_normal_lognormal_warn(series_lognormal):
    """Series distribution is not normal/gaussian"""
    msg = 'Series distribution is not normal/gaussian'
    with pytest.warns(UserWarning, match=msg):
        series_lognormal.cleaner.detect.iqr(normaltest='warn')


def test_normal_lognormal_error(series_lognormal):
    """Series distribution is not normal/gaussian"""
    msg = 'Series distribution is not normal/gaussian'
    with pytest.raises(Exception, match=msg):
        series_lognormal.cleaner.detect.iqr(normaltest='error')


def test_normal_short_warn(series_short):
    """Series distribution is not normal/gaussian"""
    msg = "Not enough rows to test normality. Must be > 8"
    with pytest.warns(UserWarning, match=msg):
        series_short.cleaner.detect.iqr(normaltest='warn')


def test_normal_short_error(series_short):
    """Series distribution is not normal/gaussian"""
    msg = "Not enough rows to test normality. Must be > 8"
    with pytest.raises(Exception, match=msg):
        series_short.cleaner.detect.iqr(normaltest='error')


def test_inverse_boxcox_0():
    """Inverse boxcox with lambda_=0"""
    x = pd.Series([0.1, 1., 2.])
    result = pd.Series(_inverse_boxcox(boxcox(x, lmbda=0), lambda_=0))
    assert_series_equal(x, result)


def test_inverse_boxcox_05():
    """Inverse boxcox with lambda_=0.5"""
    x = pd.Series([0.1, 1., 2.])
    result = pd.Series(_inverse_boxcox(boxcox(x, lmbda=0.5), lambda_=0.5))
    assert_series_equal(x, result)


def test_inverse_yeojohson_0():
    """Inverse Yeo-Johnson with lambda=0"""
    x = pd.Series([-2., 0., 2.])
    result = pd.Series(yeojohnson(x, lmbda=0.)).apply(_inverse_yeojohnson, lambda_=0.)
    assert_series_equal(x, result)


def test_inverse_yeojohson_2():
    """Inverse Yeo-Johnson with lambda=0"""
    x = pd.Series([-2., 0., 2.])
    result = pd.Series(yeojohnson(x, lmbda=2)).apply(_inverse_yeojohnson, lambda_=2)
    assert_series_equal(x, result)


def test_inverse_yeojohson_05():
    """Inverse Yeo-Johnson with lambda=0"""
    x = pd.Series([-2., 0., 2.])
    result = pd.Series(yeojohnson(x, lmbda=.5)).apply(_inverse_yeojohnson, lambda_=.5)
    assert_series_equal(x, result)


def test_transform_lambda(series_with_outlier):
    """lambda boxcox"""
    detector = series_with_outlier.cleaner.detect('iqr', transform="boxcox")
    _, lmdba = boxcox(series_with_outlier)
    assert lmdba == detector.lmbda


def test_transform_lambda_yj(series_with_outlier):
    """lambda yeojohnson"""
    detector = series_with_outlier.cleaner.detect('iqr', transform="yeojohnson")
    _, lmdba = yeojohnson(series_with_outlier)
    assert lmdba == detector.lmbda


def test_iqr_boxcox(series_lognormal):
    detector = series_lognormal.cleaner.detect.iqr(transform='boxcox')
    st = pd.Series(boxcox(series_lognormal + 1 - series_lognormal.min())[0])
    detector2 = st.cleaner.detect('iqr')
    assert_series_equal(detector.is_error(), detector2.is_error())


def test_iqr_yj(series_lognormal):
    detector = series_lognormal.cleaner.detect.iqr(transform='yeojohnson')
    st = pd.Series(yeojohnson(series_lognormal)[0])
    detector2 = st.cleaner.detect('iqr')
    assert_series_equal(detector.is_error(), detector2.is_error())


def test_zscore_boxcox(series_lognormal):
    detector = series_lognormal.cleaner.detect.zscore(transform='boxcox')
    st = pd.Series(boxcox(series_lognormal + 1 - series_lognormal.min())[0])
    detector2 = st.cleaner.detect('zscore')
    assert_series_equal(detector.is_error(), detector2.is_error())


def test_zscore_yj(series_lognormal):
    detector = series_lognormal.cleaner.detect.zscore(transform='yeojohnson')
    st = pd.Series(yeojohnson(series_lognormal)[0])
    detector2 = st.cleaner.detect('zscore')
    assert_series_equal(detector.is_error(), detector2.is_error())


def test_modzscore_boxcox(series_lognormal):
    detector = series_lognormal.cleaner.detect.modzscore(transform='boxcox')
    st = pd.Series(boxcox(series_lognormal + 1 - series_lognormal.min())[0])
    detector2 = st.cleaner.detect('modzscore')
    assert_series_equal(detector.is_error(), detector2.is_error())


def test_modzscore_yj(series_lognormal):
    detector = series_lognormal.cleaner.detect.modzscore(transform='yeojohnson')
    st = pd.Series(yeojohnson(series_lognormal)[0])
    detector2 = st.cleaner.detect('modzscore')
    assert_series_equal(detector.is_error(), detector2.is_error())


def test_iqr_normal_notransform(series_normal, capsys):
    detector = series_normal.cleaner.detect.iqr(transform='boxcox')
    assert detector.isnormal
    assert not hasattr(detector, '_transformed')
    detector.report()
    captured = capsys.readouterr()
    assert 'The transformation has not been applied' in captured.out.replace('\n', ' ')


def test_zscore_normal_notransform(series_normal, capsys):
    detector = series_normal.cleaner.detect.zscore(transform='boxcox')
    assert detector.isnormal
    assert not hasattr(detector, '_transformed')
    detector.report()
    captured = capsys.readouterr()
    assert 'The transformation has not been applied' in captured.out.replace('\n', ' ')


def test_modzscore_normal_notransform(series_normal, capsys):
    detector = series_normal.cleaner.detect.modzscore(transform='boxcox')
    assert detector.isnormal
    assert not hasattr(detector, '_transformed')
    detector.report()
    captured = capsys.readouterr()
    assert 'The transformation has not been applied' in captured.out.replace('\n', ' ')

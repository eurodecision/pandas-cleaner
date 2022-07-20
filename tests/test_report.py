"""Test the report() method for all detectors"""

import re


def test_report_title(capsys, series_with_outlier):
    "iqr title"
    detector = series_with_outlier.cleaner.detect('iqr')
    detector.report()
    captured = capsys.readouterr()
    assert 'Detection report' in captured.out


def test_report_nsamples(capsys, series_with_outlier):
    "iqr nb samples"
    detector = series_with_outlier.cleaner.detect('iqr')
    detector.report()
    captured = capsys.readouterr()
    assert 'Nb samples:                        4' in captured.out


def test_repr(capsys, series_with_outlier):
    """iqr nb errors"""
    detector = series_with_outlier.cleaner.detect('iqr')
    print(detector)
    captured = capsys.readouterr()
    assert 'Detected errors: 1 among 4 samples' in captured.out


def test_report_bounded(capsys, series_with_outlier):
    """bounded"""
    detector = series_with_outlier.cleaner.detect('bounded', lower=2)
    detector.report()
    captured = capsys.readouterr()
    assert 'sided                           both' in captured.out


def test_report_type(capsys, series_with_outlier):
    """types"""
    detector = series_with_outlier.cleaner.detect('types', ptype='int')
    detector.report()
    captured = capsys.readouterr()
    assert 'ptype                            int' in captured.out


def test_report_castable(capsys, series_with_different_types):
    """castable"""
    detector = series_with_different_types.cleaner.detect('castable', target='int')
    detector.report()
    captured = capsys.readouterr()
    assert 'target                           int' in captured.out


def test_report_custom(capsys, series_with_outlier):
    """custom"""
    detector = series_with_outlier.cleaner.detect('custom', error_func=lambda x: False)
    detector.report()
    captured = capsys.readouterr()
    assert 'Method:                       custom' in captured.out


def test_report_quantiles(capsys, series_with_outlier):
    """quantiles"""
    detector = series_with_outlier.cleaner.detect('quantiles', upperq=.9)
    detector.report()
    captured = capsys.readouterr()
    assert 'upperq                           0.9' in captured.out


def test_report_zscore(capsys, series_with_outlier):
    """zscore"""
    detector = series_with_outlier.cleaner.detect('zscore')
    detector.report()
    captured = capsys.readouterr()
    assert 'threshold                       1.96' in captured.out


def test_report_modzscore(capsys, series_with_outlier):
    """modified zscore"""
    detector = series_with_outlier.cleaner.detect('modzscore')
    detector.report()
    captured = capsys.readouterr()
    assert 'threshold                        3.5' in captured.out


def test_report_iqr_transform(capsys, series_with_outlier):
    """IQR with boxcox transform"""
    detector = series_with_outlier.cleaner.detect('iqr', transform='boxcox')
    detector.report()
    captured = capsys.readouterr()
    assert 'iqr parameters after boxcox transformation' in captured.out
    assert 'A boxcox transformation has been applied' in captured.out.replace('\n', ' ')


def test_subtitle_report_iqr_no_transform(capsys, series_with_outlier):
    """IQR with boxcox transform"""
    detector = series_with_outlier.cleaner.detect('iqr')
    detector.report()
    captured = capsys.readouterr()
    assert 'iqr parameters after boxcox transformation' not in captured.out
    assert 'has been applied' not in captured.out.replace('\n', ' ')


def test_report_zscore_transform(capsys, series_with_outlier):
    """Zscore with yeojohnson transform"""
    detector = series_with_outlier.cleaner.detect('zscore', transform='yeojohnson')
    detector.report()
    captured = capsys.readouterr()
    assert 'zscore parameters after yeojohnson transformation' in captured.out
    assert 'A yeojohnson transformation has been applied' in captured.out.replace('\n', ' ')


def test_report_modzscore_transform(capsys, series_with_outlier):
    """modzscore with yeojohnson transform"""
    detector = series_with_outlier.cleaner.detect('modzscore', transform='yeojohnson')
    detector.report()
    captured = capsys.readouterr()
    assert 'modzscore parameters after yeojohnson transformation' in captured.out
    assert 'A yeojohnson transformation has been applied' in captured.out.replace('\n', ' ')


def test_report_value(capsys, cat_series):
    r"""Generates a report for the value detector.
    Checks the value field"""
    detector = cat_series.cleaner.detect('value', value='cat')
    detector.report()
    captured = capsys.readouterr()
    assert 'value                            cat' in captured.out


def test_report_value_check_type(capsys, cat_series):
    r"""Generates a report for the value detector.
    Checks the check_type field"""
    detector = cat_series.cleaner.detect('value', value='cat')
    detector.report()
    captured = capsys.readouterr()
    assert 'check_type                      True' in captured.out


def test_report_cat_counts(capsys, cat_series):
    """counts"""
    detector = cat_series.cleaner.detect('counts')
    detector.report()
    captured = capsys.readouterr()
    assert 'n                                  1' in captured.out


def test_report_cat_freq(capsys, cat_series):
    """freq"""
    detector = cat_series.cleaner.detect('freq')
    detector.report()
    captured = capsys.readouterr()
    assert 'freq                             0.1' in captured.out


def test_report_pattern(capsys, cat_series):
    """pattern"""
    detector = cat_series.cleaner.detect('pattern', pattern=r"[a-z]*")
    detector.report()
    captured = capsys.readouterr()
    assert 'flags                              0' in captured.out


def test_report_pattern_recompile(capsys, cat_series):
    """pattern with a compiled regex"""
    regex = re.compile(r"[a-z]*")
    detector = cat_series.cleaner.detect('pattern', pattern=regex)
    detector.report()
    captured = capsys.readouterr()
    assert 'pattern                     compiled' in captured.out


def test_report_extraspaces(capsys, series_with_extra_spaces):
    """extra spaces detector"""
    detector = series_with_extra_spaces.cleaner.detect('spaces')
    detector.report()
    captured = capsys.readouterr()
    assert 'side                            both' in captured.out


def test_report_web_url_check(capsys, series_with_urls):
    """url detector """
    detector = series_with_urls.cleaner.detect('url')
    detector.report()
    captured = capsys.readouterr()
    assert 'check_protocol                  True' in captured.out


def test_report_web_url_nocheck(capsys, series_with_urls):
    """url detector"""
    detector = series_with_urls.cleaner.detect('url', check_protocol=False)
    detector.report()
    captured = capsys.readouterr()
    assert 'check_protocol                 False' in captured.out


def test_report_fingerprint(capsys, cat_series):
    """key collision"""
    detector = cat_series.cleaner.detect('keycollision')
    detector.report()
    captured = capsys.readouterr()
    assert 'keys                     fingerprint' in captured.out


def test_report_ndoutliers(capsys, anscombe):
    """multivariate outliers"""
    df = anscombe
    df = df[df.dataset == 'I'].reset_index()[['x', 'y']]
    detector = df.cleaner.detect("ndoutliers")
    detector.report()
    captured = capsys.readouterr()
    assert 'min_samples                        2' in captured.out


def test_report_by_category(capsys, df_num_cat):
    """iqr by category"""
    detector = df_num_cat.cleaner.detect("iqr")
    detector.report()
    captured = capsys.readouterr()
    assert 'method                           iqr' in captured.out


def test_report_missing(capsys, dataframe_with_nan):
    """missing"""
    detector = dataframe_with_nan.cleaner.detect.missing(how='all')
    detector.report()
    captured = capsys.readouterr()
    assert 'how                              all' in captured.out


def test_report_length(capsys, series_with_different_length):
    """length"""
    detector = series_with_different_length.cleaner.detect('length', mode='bound', lower=3)
    detector.report()
    captured = capsys.readouterr()
    assert 'lower                              3' in captured.out


def test_report_duplicated(capsys, dataframe_with_duplicates):
    """duplicated"""
    detector = dataframe_with_duplicates.cleaner.detect('duplicated', subset='col1', keep='first')
    detector.report()
    captured = capsys.readouterr()
    assert 'keep                           first' in captured.out


def test_report_date_range(capsys, series_with_datetime):
    """datetime"""
    detector = series_with_datetime.cleaner.detect('date_range', lower='2020-06-15',
                                                   upper='2022-08-05')
    detector.report()
    captured = capsys.readouterr()
    assert 'lower                     2020-06-15' in captured.out
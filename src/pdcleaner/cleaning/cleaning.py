"""Definition of cleaning methods

Those methods apply to detectors.
"""

import warnings

import numpy as np
import pandas as pd


def clip(self, detector, inplace=False):
    """
    Clean numerical series by clipping between lower and upper values

    Parameters
    ----------
    detector: a Detector object,
        The detector obj that will identify entries to clean
        and provide upper and lower limits

    inplace:  bool (Default: False)
        Whether to perform the operation in place on the data.

    Returns
    -------
    The modified data or None if inplace is True

    Raises
    ------
    ValueError when input Detector does not have an upper or a lower attribute.

    Examples
    --------

    >>> series = pd.Series([np.nan,
                               0,
                               -5,
                               4,
                               6,
                               100,
                               ])
    >>> detector = series.cleaner.detect.bounded(lower=0, upper=10)
    >>> series.cleaner.clean.clip(detector)
    0     NaN
    1     0.0
    2     0.0
    3     4.0
    4     6.0
    5    10.0
    dtype: float64

    Modify inplace

    >>> series.cleaner.clean.clip(detector, inplace=True)
    >>> series
    0     NaN
    1     0.0
    2     0.0
    3     4.0
    4     6.0
    5    10.0
    dtype: float64
    """
    series = detector.obj

    if not hasattr(detector, 'lower') or not hasattr(detector, 'upper'):
        raise ValueError('The errors detection method does not have lower and upper bounds'
                         ' and can not be used with clip')

    # else
    lower = detector.lower
    upper = detector.upper

    if inplace:
        series.clip(upper=upper, lower=lower, inplace=True)
        return None
    return series.clip(upper=upper, lower=lower)


def drop(self, detector, inplace=False):
    """
    Clean by dropping errors

    Parameters
    ----------
    detector: a Detector object,
        The detector obj that will identify entries to clean

    inplace:  bool (Default: False)
        Whether to perform the operation in place on the data.

    Warning
    -------

    Dropping inplace for dataframe columns is not supported.

    Returns
    -------
    The modified data or None if inplace is True

    Raises
    -------
    Warning if the method is applied to a dataframe column with inplace

    Examples
    --------

    >>> series = pd.Series([np.nan,
                               0,
                               -5,
                               4,
                               6,
                               100,
                               ])
    >>> detector = series.cleaner.detect.bounded(lower=0, upper=10)
    >>> series.cleaner.clean.drop(detector)
    0    NaN
    1    0.0
    3    4.0
    4    6.0
    dtype: float64

    Modify inplace

    >>> series.cleaner.clean.drop(detector, inplace=True)
    >>> series
    0    NaN
    1    0.0
    3    4.0
    4    6.0
    dtype: float64

    Cleaning a dataframe

    >>> df = pd.DataFrame({"col1": [np.nan, 0, -5, 4, 6, 100],
    >>>                    "col2": ["a", "b", "c", "d", "e", "f"]})
    >>> df_detector = df["col1"].cleaner.detect.bounded(lower=0, upper=10)
    >>> df["col1"].cleaner.clean.drop(df_detector)

    Cleaning a dataframe inplace with the drop method is not supported
    (issues a warning)

    >>> df = pd.DataFrame({"col1": [np.nan, 0, -5, 4, 6, 100],
    >>>                    "col2": ["a", "b", "c", "d", "e", "f"]})
    >>> df_detector = df["col1"].cleaner.detect.bounded(lower=0, upper=10)
    >>> df["col1"].cleaner.clean.drop(df_detector, inplace=True)
    """

    if inplace:

        if isinstance(detector.obj, pd.Series):
            if detector.obj._get_cacher() is not None:
                warnings.warn("Series is a column of a DataFrame. "
                              "Dropping inplace will not modify the DataFrame.")

        self._obj.drop(detector.index, inplace=True)
        return None
    return self._obj.drop(detector.index)


def to_na(self, detector, inplace=False):
    """
    Clean by replacing errors by NaN

    Parameters
    ----------
    detector: a Detector object,
        The detector obj that will identify entries to clean

    inplace:  bool (Default: False)
        Whether to perform the operation in place on the data.


    Returns
    -------
    The modified data or None if inplace is True

    Examples
    --------

    >>> series = pd.Series([np.nan,
                               0,
                               -5,
                               4,
                               6,
                               100,
                               ])
    >>> detector = series.cleaner.detect.bounded(lower=0, upper=10)
    >>> series.cleaner.clean.to_na(detector)
    0    NaN
    1    0.0
    2    NaN
    3    4.0
    4    6.0
    5    NaN
    dtype: float64

    Replace inplace

    >>> series.cleaner.clean.to_na(detector, inplace=True)
    >>> series
    0    NaN
    1    0.0
    2    NaN
    3    4.0
    4    6.0
    5    NaN
    dtype: float64
    """
    series = detector.obj

    if inplace:
        series.where(detector.not_error(), np.nan, inplace=True)
        return None
    return series.where(detector.not_error(), np.nan)


def replace(self, detector, value=None, inplace=False):
    """
    Clean by replacing errors by a value

    Parameters
    ----------
    detector: a Detector object,
        The detector obj that will identify entries to clean
    value: string, numeric, dict or callable
        The replacement strategy.
        If a single value is given, errors will be replaced with that value.

        If a dictionary is given,
        errors will be replaced with respect to the dictionary entries.
        If an erroneous value does not have a corresponding key in the value dict,
        it will be replaced with nan.

        If a callable is given, it is computed on the Series.
        It should return a scalar or a Series.
        It must not change input series.
    inplace:  bool (Default: False)
        Whether to perform the operation in place on the data.


    Returns
    -------
    The modified data or None if inplace is True

    Examples
    --------

    >>> series = pd.Series([np.nan,
                               0,
                               -5,
                               4,
                               6,
                               100,
                               ])
    >>> detector = series.cleaner.detect.bounded(lower=0, upper=10)
    >>> series.cleaner.clean.replace(detector, value=5)
    0    NaN
    1    0.0
    2    5.0
    3    4.0
    4    6.0
    5    5.0
    dtype: float64

    Replace using a dictionary

    >>> series.cleaner.clean.replace(detector, value={100:10})
    0    NaN
    1    0.0
    2    NaN
    3    4.0
    4    6.0
    5   10.0
    dtype: float64

    Replace using a lambda (the lambda applies to the series of erroneous entries)

    >>> series.cleaner.clean.replace(detector,
    >>>                                 value=lambda s: s.clip(lower=0) / 10 )
    0    NaN
    1    0.0
    2    0.0
    3    4.0
    4    6.0
    5   10.0
    dtype: float64
    """
    series = detector.obj

    if isinstance(value, dict):
        value_dict = value

        def dict2func(series_):
            return series_.map(value_dict)
        value = dict2func

    return series.where(detector.not_error(), value, inplace=inplace)


def alternatives(self, detector, inplace=False):
    """
    Clean by merging values that have been identified as alternative representations
    of the same object by a key collision detector.

    Note
    ----
    This method can only applied with a 'alternatives' detector.

    Parameters
    ----------
    detector: a Detector object,
        The detector obj that will identify entries to clean
    inplace:  bool (Default: False)
        Whether to perform the operation in place on the data.

    Returns
    -------
    The modified data or None if inplace is True

    Example
    -------

    Detect alternative formulations for the same name

    >>> series = pd.Series(['Linus Torvalds','linus.torvalds','Torvalds, Linus',
    >>>                     'Linus Torvalds', 'Bill Gates'])
    >>> detector = series.cleaner.detect.alternatives()
    >>> print(detector.is_error())
    0    False
    1     True
    2     True
    3    False
    4    False
    dtype: bool

    Clean by standardizing

    >>> print(series.cleaner.clean('alternatives', detector))
    0    Linus Torvalds
    1    Linus Torvalds
    2    Linus Torvalds
    3    Linus Torvalds
    4        Bill Gates
    dtype: object
    """

    if not hasattr(detector, 'dict_keys'):
        raise ValueError('The errors detection method does not have a keys dictionary')

    series = self._obj

    keys = detector.fingerprints(series)

    if inplace:
        dict_replace = pd.Series(keys.map(detector.dict_keys).values,
                                 index=series
                                 ).to_dict()

        series.replace(dict_replace, inplace=True)
        return None
    return keys.map(detector.dict_keys)


def cast(self, detector, **kwargs):
    """
    Clean by casting value into the specific target type. When the value is not castable, it is
    transformed to NaN. This method works only with the castable detector.

    When casting into date, all parameters in pd.to_datetime method are allowed.
    See https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html

    Parameters
    ----------
    detector: a Detector object,
        The detector obj that will identify entries to clean

    Returns
    -------
    The modified data

    Examples
    --------

    >>> series = pd.Series(['100 000', '154,5', '9 000', '250,12'], dtype='object')
    >>> detector = series.cleaner.detect.castable(target='float', thousands=' ', decimal=',')
    >>> series.cleaner.clean('cast', detector)
    0    100000
    1    154.5
    2    9000
    3    250.12
    dtype: float64

    Casting into int

    >>> detector = series.cleaner.detect.castable(target='int', thousands=' ', decimal=',')
    >>> series.cleaner.clean('cast', detector)
    0    100000
    1      <NA>
    2      9000
    3      <NA>
    dtype: Int32

    Casting into date

    >>> series = pd.Series(['1.05', '154', '15/05/2022', 'Alice'], dtype='object')
    >>> detector = series.cleaner.detect.castable(target='date')
    >>> series.cleaner.clean('cast', detector, format="%d/%m/%Y")
    0    NaT
    1    NaT
    2    2022-05-15
    3    NaT
    dtype: datetime64[ns]
    """

    if detector.name != 'castable':
        raise ValueError('This cleaning method works only with the castable detector')

    series = detector.obj
    series = series.where(detector.not_error(), np.nan)

    if detector.target == "int":
        series = detector.check_separators(series)
        series = pd.to_numeric(series, errors="coerce").astype('Int32')

    if detector.target == "float":
        series = detector.check_separators(series)
        series = pd.to_numeric(series, errors="coerce")

    if detector.target == "date":
        series = pd.to_datetime(series, errors="coerce", **kwargs)

    if detector.target == "boolean":
        series = series.map(detector.bool_values)

    return series


def strip(self, detector):
    """
    Clean by removing extraspaces detected by the 'spaces' detector. This method only works with
    this detector.

    Parameters
    ----------
    detector: a Detector object,
        The detector obj that will identify entries to clean

    Returns
    -------
    The modified data

    Examples
    --------
    >>> series = pd.Series(['Paris','Paris ',' Lille', ' Lille ', 'Troyes'])
    >>> detector = series.cleaner.detect.spaces(side='trailing')
    >>> series.cleaner.clean.strip(detector)
    0    Paris
    1    Paris
    2     Lille
    3     Lille
    4    Troyes
    dtype: object

    >>> detector = series.cleaner.detect.spaces(side='both')
    >>> series.cleaner.clean.strip(detector)
    0    Paris
    1    Paris
    2    Lille
    3    Lille
    4    Troyes
    dtype: object
    """

    if detector.name != 'spaces':
        raise ValueError('This cleaning method works only with the spaces detector')

    series = detector.obj

    if detector.side == 'leading':
        series = series.str.lstrip()
    elif detector.side == 'trailing':
        series = series.str.rstrip()
    elif detector.side == 'both':
        series = series.str.strip()

    return series

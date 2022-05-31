"""Definition of cleaning methods

Those methods apply to detectors.
"""

import warnings

import numpy as np
import pandas as pd


def clip(self, detector_obj, inplace=False):
    """
    Clean numerical series by clipping between lower and upper values

    Parameters
    ----------
    detector_obj: a Detector object,
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

    >>> my_series = pd.Series([np.nan,
                               0,
                               -5,
                               4,
                               6,
                               100,
                               ])
    >>> my_detector = my_series.cleaner.detect.bounded(lower=0, upper=10)
    >>> my_series.cleaner.clean.clip(my_detector)
    0     NaN
    1     0.0
    2     0.0
    3     4.0
    4     6.0
    5    10.0
    dtype: float64

    Modify inplace

    >>> my_series.cleaner.clean.clip(my_detector, inplace=True)
    >>> my_series
    0     NaN
    1     0.0
    2     0.0
    3     4.0
    4     6.0
    5    10.0
    dtype: float64
    """
    series = detector_obj.obj

    if not hasattr(detector_obj, 'lower') or not hasattr(detector_obj, 'upper'):
        raise ValueError('The errors detection method does not have lower and upper bounds'
                         ' and can not be used with clip')

    # else
    lower = detector_obj.lower
    upper = detector_obj.upper

    if inplace:
        series.clip(upper=upper, lower=lower, inplace=True)
        return None
    return series.clip(upper=upper, lower=lower)


def drop(self, detector_obj, inplace=False):
    """
    Clean by dropping errors

    Parameters
    ----------
    detector_obj: a Detector object,
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

    >>> my_series = pd.Series([np.nan,
                               0,
                               -5,
                               4,
                               6,
                               100,
                               ])
    >>> my_detector = my_series.cleaner.detect.bounded(lower=0, upper=10)
    >>> my_series.cleaner.clean.drop(my_detector)
    0    NaN
    1    0.0
    3    4.0
    4    6.0
    dtype: float64

    Modify inplace

    >>> my_series.cleaner.clean.drop(my_detector, inplace=True)
    >>> my_series
    0    NaN
    1    0.0
    3    4.0
    4    6.0
    dtype: float64

    Cleaning a dataframe

    >>> df = pd.DataFrame({"col1": [np.nan, 0, -5, 4, 6, 100],
    >>>                    "col2": ["a", "b", "c", "d", "e", "f"]})
    >>> my_df_detector = df["col1"].cleaner.detect.bounded(lower=0, upper=10)
    >>> df["col1"].cleaner.clean.drop(my_df_detector)

    Cleaning a dataframe inplace with the drop method is not supported
    (issues a warning)

    >>> df = pd.DataFrame({"col1": [np.nan, 0, -5, 4, 6, 100],
    >>>                    "col2": ["a", "b", "c", "d", "e", "f"]})
    >>> my_df_detector = df["col1"].cleaner.detect.bounded(lower=0, upper=10)
    >>> df["col1"].cleaner.clean.drop(my_df_detector, inplace=True)
    """
    series = detector_obj.obj

    if inplace:
        if series._get_cacher() is not None:
            warnings.warn("Series is a column of a DataFrame. "
                          "Dropping inplace will not modify the DataFrame.")

        self._obj.drop(detector_obj.index, inplace=True)
        return None
    return self._obj.drop(detector_obj.index)


def to_na(self, detector_obj, inplace=False):
    """
    Clean by replacing errors by NaN

    Parameters
    ----------
    detector_obj: a Detector object,
        The detector obj that will identify entries to clean

    inplace:  bool (Default: False)
        Whether to perform the operation in place on the data.


    Returns
    -------
    The modified data or None if inplace is True

    Examples
    --------

    >>> my_series = pd.Series([np.nan,
                               0,
                               -5,
                               4,
                               6,
                               100,
                               ])
    >>> my_detector = my_series.cleaner.detect.bounded(lower=0, upper=10)
    >>> my_series.cleaner.clean.to_na(my_detector)
    0    NaN
    1    0.0
    2    NaN
    3    4.0
    4    6.0
    5    NaN
    dtype: float64

    Replace inplace

    >>> my_series.cleaner.clean.to_na(my_detector, inplace=True)
    >>> my_series
    0    NaN
    1    0.0
    2    NaN
    3    4.0
    4    6.0
    5    NaN
    dtype: float64
    """
    series = detector_obj.obj

    if inplace:
        series.where(detector_obj.not_error(), np.nan, inplace=True)
        return None
    return series.where(detector_obj.not_error(), np.nan)


def replace(self, detector_obj, value=None, inplace=False):
    """
    Clean by replacing errors by a value

    Parameters
    ----------
    detector_obj: a Detector object,
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

    >>> my_series = pd.Series([np.nan,
                               0,
                               -5,
                               4,
                               6,
                               100,
                               ])
    >>> my_detector = my_series.cleaner.detect.bounded(lower=0, upper=10)
    >>> my_series.cleaner.clean.replace(my_detector, value=5)
    0    NaN
    1    0.0
    2    5.0
    3    4.0
    4    6.0
    5    5.0
    dtype: float64

    Replace using a dictionary

    >>> my_series.cleaner.clean.replace(my_detector, value={100:10})
    0    NaN
    1    0.0
    2    NaN
    3    4.0
    4    6.0
    5   10.0
    dtype: float64

    Replace using a lambda (the lambda applies to the series of erroneous entries)

    >>> my_series.cleaner.clean.replace(my_detector,
    >>>                                 value=lambda s: s.clip(lower=0) / 10 )
    0    NaN
    1    0.0
    2    0.0
    3    4.0
    4    6.0
    5   10.0
    dtype: float64
    """
    series = detector_obj.obj

    if isinstance(value, dict):
        value_dict = value

        def dict2func(series_):
            return series_.map(value_dict)
        value = dict2func

    return series.where(detector_obj.not_error(), value, inplace=inplace)


def bykeys(self, detector_obj, inplace=False):
    """
    Clean by merging values that have been identified as alternative representations
    of the same object by a key collision detector.

    Note
    ----
    This method can only applied with a 'keycollision' detector.

    Parameters
    ----------
    detector_obj: a Detector object,
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
    >>> detector = series.cleaner.detect.keycollision()
    >>> print(detector.is_error())
    0    False
    1     True
    2     True
    3    False
    4    False
    dtype: bool

    Clean by standardizing

    >>> print(series.cleaner.clean('bykeys', detector))
    0    Linus Torvalds
    1    Linus Torvalds
    2    Linus Torvalds
    3    Linus Torvalds
    4        Bill Gates
    dtype: object
    """

    if not hasattr(detector_obj, 'dict_keys'):
        raise ValueError('The errors detection method does not have a keys dictionary')

    series = self._obj

    keys = detector_obj.fingerprints(series)

    if inplace:
        dict_replace = pd.Series(keys.map(detector_obj.dict_keys).values,
                                 index=series
                                 ).to_dict()

        series.replace(dict_replace, inplace=True)
        return None
    return keys.map(detector_obj.dict_keys)


def cast(self, detector_obj, **kwargs):
    """
    Clean by casting value into the specific target type. When the value is not castable, it is
    transformed to NaN. This method works only with the castable detector.

    When casting into date, all parameters in pd.to_datetime method are allowed.
    See https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html

    Parameters
    ----------
    detector_obj: a Detector object,
        The detector obj that will identify entries to clean

    Returns
    -------
    The modified data

    Examples
    --------

    >>> my_series = pd.Series(['100 000', '154,5', '9 000', '250,12'], dtype='object')
    >>> my_detector = my_series.cleaner.detect.castable(target='float', thousands=' ', decimal=',')
    >>> my_series.cleaner.clean('cast', my_detector)
    0    100000
    1    154.5
    2    9000
    3    250.12
    dtype: float64

    Casting into int

    >>> my_detector = my_series.cleaner.detect.castable(target='int', thousands=' ', decimal=',')
    >>> my_series.cleaner.clean('cast', my_detector)
    0    100000
    1      <NA>
    2      9000
    3      <NA>
    dtype: Int32

    Casting into date

    >>> my_series = pd.Series(['1.05', '154', '15/05/2022', 'Alice'], dtype='object')
    >>> my_detector = my_series.cleaner.detect.castable(target='date')
    >>> my_series.cleaner.clean('cast', my_detector, format="%d/%m/%Y")
    0    NaT
    1    NaT
    2    2022-05-15
    3    NaT
    dtype: datetime64[ns]
    """

    if detector_obj.name != 'castable':
        raise ValueError('This cleaning method works only with the castable detector')

    series = detector_obj.obj
    series = series.where(detector_obj.not_error(), np.nan)

    if detector_obj.target == "int":
        series = detector_obj.check_separators(series)
        series = pd.to_numeric(series, errors="coerce").astype('Int32')

    if detector_obj.target == "float":
        series = detector_obj.check_separators(series)
        series = pd.to_numeric(series, errors="coerce")

    if detector_obj.target == "date":
        series = pd.to_datetime(series, errors="coerce", **kwargs)

    if detector_obj.target == "boolean":
        series = series.map(detector_obj.bool_values)

    return series


def strip(self, detector_obj):
    """
    Clean by removing extraspaces detected by the 'spaces' detector. This method only works with
    this detector.

    Parameters
    ----------
    detector_obj: a Detector object,
        The detector obj that will identify entries to clean

    Returns
    -------
    The modified data

    Examples
    --------
    >>> my_series = pd.Series(['Paris','Paris ',' Lille', ' Lille ', 'Troyes'])
    >>> my_detector = my_series.cleaner.detect.spaces(side='right')
    >>> my_series.cleaner.clean.strip(my_detector)
    0    Paris
    1    Paris
    2     Lille
    3     Lille
    4    Troyes
    dtype: object

    >>> my_detector = my_series.cleaner.detect.spaces(side='both')
    >>> my_series.cleaner.clean.strip(my_detector)
    0    Paris
    1    Paris
    2    Lille
    3    Lille
    4    Troyes
    dtype: object
    """

    if detector_obj.name != 'spaces':
        raise ValueError('This cleaning method works only with the spaces detector')

    series = detector_obj.obj

    if detector_obj.side == 'left':
        series = series.str.lstrip()
    elif detector_obj.side == 'right':
        series = series.str.rstrip()
    elif detector_obj.side == 'both':
        series = series.str.strip()

    return series

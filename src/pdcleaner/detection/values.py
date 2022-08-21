"""
Values detectors
"""
import numbers
import pandas as pd

from pdcleaner.detection._base import _SeriesDetector, _TwoColsCategoricalDataFramesDetector


class enum(_SeriesDetector):
    r"""Detect class values not in a given list.

    Intended to be used by the detect method with the keyword 'enum'.

    >>> series.cleaner.detect.enum(...)
    >>> series.cleaner.detect('enum',...)

    This detection method flags values as potential errors wherever the
    corresponding Series element is not in the given values list.

    Alternatively, if `forbidden=True`, potential errors are detected when
    the corresponding Series element is in the given values list.

    Note
    ----

    NA values are not treated as errors.

    Parameters
    ----------
    values: list of strings
        Authorized values
    forbidden: bool (Default: False)
        If forbidden=True, errors are when elements are in the given list

    Raises
    ------
    ValueError
        when values is empty

    Examples
    --------

    >>> my_series = pd.Series(['cat','cat','dog','bird'])
    >>> my_detector = my_series.cleaner.detect.enum(values=['cat','dog'])
    >>> print(my_detector.is_error())
    0    False
    1    False
    2    False
    3     True
    dtype: bool

    The detector can also be used with numerical values:

    >>> my_series = pd.Series([5, 5.0, 10, 3])
    >>> my_detector = my_series.cleaner.detect.enum(values=[5,3])
    >>> print(my_detector.detected)
    2    10.0
    dtype: float64

    Missing values are not treated as errors.

    >>> my_series = pd.Series(['cat',np.nan,'dog','bird'])
    >>> my_detector = my_series.cleaner.detect.enum(values=['cat','dog'])
    >>> print(my_detector.is_error())
    0    False
    1    False
    2    False
    3     True
    dtype: bool

    Use `forbidden=True` to detect values in the list as errors

    >>> my_series = pd.Series(['cat','cat','dog','bird'])
    >>> my_detector = \
        my_series.cleaner.detect.enum(values=['cat','dog'], forbidden=True)
    >>> print(my_detector.is_error())
    0     True
    1     True
    2     True
    3    False
    dtype: bool

    """
    name = 'enum'

    def __init__(self, obj,
                 detector_obj=None,
                 values=None,
                 forbidden=False):

        super().__init__(obj)

        if values is None:
            values = []

        if not detector_obj:
            self._values = values
            self._forbidden = forbidden
        else:
            self._values = detector_obj.values
            self._forbidden = detector_obj.forbidden

        if len(self._values) == 0:
            raise ValueError("The list of authorized values is empty")

    @property
    def values(self):
        """List of valid values"""
        return self._values

    @property
    def forbidden(self):
        """Is given value a forbidden one (or an expected) ?"""
        return self._forbidden

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        mask = ~self._obj.isin(self.values)

        if self.forbidden:
            mask = ~mask

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index


class value(_SeriesDetector):
    r"""Detect class values different from a value.

    Intended to be used by the detect method with the keyword 'value'.

    >>> series.cleaner.detect.value(...)
    >>> series.cleaner.detect('value',...)

    This detection method flags values as potential errors wherever the
    corresponding Series element is different or not from a given value.

    Note
    ----

    NA values are not treated as errors.

    Parameters
    ----------
    value: value
        Authorized value
    forbidden: bool (Default: False)
        If forbidden=True, errors are elements equal to value

    check_type: Bool (Default: True)
        Checks the type of the value if True
        (3.0 is not the same type as 3)

    Raises
    ------
    ValueError
        when value is None

    Examples
    --------

    >>> my_series = pd.Series(['cat','cat','dog','bird'])
    >>> my_detector = my_series.cleaner.detect.value(value='cat')
    >>> print(my_detector.is_error())
    0    False
    1    False
    2     True
    3     True
    dtype: bool

    By default, the type of value and data is checked and must be identical

    >>> my_series = pd.Series([5, 5.0])
    >>> my_detector = my_series.cleaner.detect.value(value=5)
    >>> print(my_detector.is_error())
    0    False
    1    True
    dtype: bool

    >>> my_series = pd.Series([5, 5.0])
    >>> my_detector = my_series.cleaner.detect.value(value=5, check_type=False)
    >>> print(my_detector.is_error())
    0    False
    1    False
    dtype: bool

    Missing values are not treated as errors.

    >>> my_series = pd.Series(['cat',np.nan,'dog','bird'])
    >>> my_detector = my_series.cleaner.detect.value(value='cat')
    >>> print(my_detector.is_error())
    0    False
    1    False
    2     True
    3     True
    dtype: bool

    Use the `forbidden=True` argument to detect a given value as an error
    
    >>> series= pd.Series([1, 2, 3])
    >>> detector = series.cleaner.detect('value', value=1, forbidden=True)
    >>> print(detector.is_error())
    0     True
    1    False
    2    False
    dtype: bool
    """
    name = 'value'

    def __init__(self, obj,
                 detector_obj=None,
                 value=None,
                 check_type=True,
                 forbidden=False
                 ):

        super().__init__(obj)

        if detector_obj is None:
            self._value = value
            self._check_type = check_type
            self._forbidden = forbidden
        else:
            self._value = detector_obj.value
            self._check_type = detector_obj.check_type
            self._forbidden = detector_obj.forbidden

        if self.value is None:
            raise ValueError("The authorized value is not defined")

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        if self.check_type:
            mask = ~self._obj.apply(lambda x: x is self.value)
        else:
            mask = ~self._obj.apply(lambda x: x == self.value)

        if self.forbidden:
            mask = ~mask

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def value(self):
        """ Authorized value"""
        return self._value

    @property
    def check_type(self):
        """ Authorized value"""
        return self._check_type

    @property
    def forbidden(self):
        """Is given value a forbidden one (or an expected) ?"""
        return self._forbidden

    @property
    def _reported(self):
        r"""Output values for the detection report"""
        return ['value', 'check_type', 'forbidden']


class counts(_SeriesDetector):
    r"""Detect class values that appear at max n times.

    Intended to be used by the detect method with the keyword 'counts'.

    >>> series.cleaner.detect.counts(...)
    >>> series.cleaner.detect('counts',...)

    This detection method flags values as potential errors wherever the
    corresponding Series element appears less or = than n times in the Series.


    Note
    ----

    NA values are not treated as errors.

    Parameters
    ----------
    n: integer > 0 (Default: 1)
        Number of occurences under which the element is flagged as an error

    Raises
    ------
    ValueError
        when n is <= 0
    TypeError
        when n is not an integer

    Examples
    --------

    >>> my_series = pd.Series(['cat','cat','dog','dog','bird'])
    >>> my_detector = my_series.cleaner.detect.counts(n=1)
    >>> print(my_detector.values)
    ['cat','dog']
    >>> print(my_detector.is_error())
    0    False
    1    False
    2    False
    3    False
    4     True
    dtype: bool

    Use resulting object to apply to another series: only the
    previously detected valid values are considered valid.

    >>> my_series_test = pd.Series(['dog','bird','mouse','cat'])
    >>> my_detector_test = my_series.cleaner.detect(my_detector)
    >>> print(my_detector_test.is_error())
    0    False
    1     True
    2     True
    3    False
    dtype: bool

    The detector can also be used with numerical values:

    >>> my_series_test = pd.Series([5, 3, 3.0, 100, 5])
    >>> my_detector_test = my_series_test.cleaner.detect.counts(n=1)
    >>> print(my_detector_test.is_error())
    0    False
    1    False
    2    False
    3     True
    4    False
    dtype: bool

    Missing values are not treated as errors.

    >>> my_series = pd.Series(['cat',np.nan,'dog','bird'])
    >>> my_detector = my_series.cleaner.detect.counts(n=1)
    >>> print(my_detector.is_error())
    0    False
    1    False
    2    False
    3     True
    dtype: bool
    """
    name = 'counts'

    def __init__(self, obj, detector_obj=None, n=1):
        super().__init__(obj)

        if not detector_obj:
            self._n = n
            self._values = self.values
        else:
            self._n = detector_obj.n
            self._values = detector_obj._values

        if not isinstance(n, int):
            raise TypeError('n must be a >0 integer')

        if n <= 0:
            raise TypeError('n must be a >0 integer')

    @property
    def n(self):
        """Minimum number of occurences"""
        return self._n

    @property
    def values(self):
        """List of valid classes"""
        v_c = self._obj.value_counts()
        return v_c[v_c > self.n].index.to_list()

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        mask = ~self._obj.isin(self._values)

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['n']


class freq(_SeriesDetector):
    r"""Detect class values that appear less than a given freq.

    Intended to be used by the detect method with the keyword 'freq'.

    >>> series.cleaner.detect.freq(...)
    >>> series.cleaner.detect('freq',...)

    This detection method flags values as potential errors wherever the
    corresponding Series element appears less than a given frequency (ratio
    between the number of occurences and the total number of non-missing
    elements).

    Note
    ----

    NA values are not treated as errors.

    Parameters
    ----------
    freq: float (Default: .1)
        Frequency under which the element is flagged as an error
        Must be > 0 and < 1

    Raises
    ------
    ValueError
        when freq is <=0 or >=1
    TypeError
        when n is not a float

    Examples
    --------

    >>> my_series = pd.Series(['cat','cat','dog','dog','bird'])
    >>> my_detector = my_series.cleaner.detect.freq(freq=.25)
    >>> print(my_detector.values)
    ['cat','dog']
    >>> print(my_detector.is_error())
    0    False
    1    False
    2    False
    3    False
    4     True
    dtype: bool

    Use resulting object to apply to another series: only the
    previously detected valid values are considered valid.

    >>> my_series_test = pd.Series(['dog','bird','mouse','cat'])
    >>> my_detector_test = my_series.cleaner.detect(my_detector)
    >>> print(my_detector_test.is_error())
    0    False
    1     True
    2     True
    3    False
    dtype: bool

    The detector can also be used with numerical values:

    >>> my_series_test = pd.Series([5, 3, 3.0, 100, 5])
    >>> my_detector_test = my_series_test.cleaner.detect.freq(freq=0.25)
    >>> print(my_detector_test.is_error())
    0    False
    1    False
    2    False
    3     True
    4    False
    dtype: bool

    Missing values are not treated as errors.

    >>> my_series = pd.Series(['cat','cat', np.nan, 'dog', 'dog','bird'])
    >>> my_detector = my_series.cleaner.detect.counts(n=1)
    >>> print(my_detector.is_error())
    0    False
    1    False
    2    False
    3    False
    4    False
    5     True
    dtype: bool
    """
    name = 'freq'

    def __init__(self, obj, detector_obj=None, freq=0.1):
        super().__init__(obj)

        if not detector_obj:
            if not isinstance(freq, float):
                raise TypeError('freq must be a float')
            self._freq = freq
            self._values = self.values
        else:
            self._freq = detector_obj.freq
            self._values = detector_obj._values

        if (self.freq <= 0) or (self.freq >= 1):
            raise TypeError('freq must be in the range ]0;1[')

    @property
    def freq(self):
        """Minimum value frequency"""
        return self._freq

    @property
    def values(self):
        """List of valid classes"""
        v_c = self._obj.value_counts(normalize=True)
        return v_c[v_c > self._freq].index.to_list()

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        mask = ~self._obj.isin(self._values)

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['freq']


class associations(_TwoColsCategoricalDataFramesDetector):
    r"""Detects least frequent associations between two category columns

    Intended to be used by the detect method with the keyword 'associations'

    >>> dataframe.cleaner.detect.associations(...)
    >>> dataframe.cleaner.detect('associations',...)

    Parameters
    ----------
    count: int
        Minimal number of samples in which the categories values must be associated
    freq: float between 0 and 1
        Minimal frequency of samples in which the categories values must be associated
    warning:
        One must provide either count or freq, and not both

    Raises
    ------
    TypeError
        if count is not an integer
        if freq is not a float

    ValueError
        if neither count nor freq is provided
        if count and freq are both provided
        if freq is not >0 and <1

    Examples
    --------
    >>> import pandas as pd
    >>> import pdcleaner

    >>> df = pd.DataFrame({
                'col1': ['A'] * 10 + ['B'] * 10,
                'col2': ['a'] * 8 + ['c'] * 2 + ['b'] * 9 + ['a'],
        })

    >>> detector = df.cleaner.detect.associations(freq=0.05)
    >>> print(detector.detected)
        col1 col2
    19    B    a

    >>> detector = df.cleaner.detect.associations(count=3)
    >>> print(detector.detected)
        col1 col2
    8     A    c
    9     A    c
    19    B    a
    """

    name = "associations"

    def __init__(self, obj, detector_obj=None, count=None, freq=None):
        super().__init__(obj)

        if not detector_obj:
            if ((freq is None) and (count is None)) or ((freq is not None) and (count is not None)):
                raise ValueError("Either freq or count must be provided")

            if (count is not None) and (not isinstance(count, int)):
                raise TypeError("count must be an integer")

            if freq is not None:
                if not isinstance(freq, numbers.Number):
                    raise TypeError('freq must be a number')
                if (freq <= 0) or (freq >= 1):
                    raise ValueError("freq must be between 0 and 1 exclusive")

            self._count = count
            self._freq = freq
            self._valid_associations = \
                self._calculate_valid_associations(self._obj,
                                                   self.normalize,
                                                   self.limit,
                                                   )
        else:
            self._count = detector_obj.count
            self._freq = detector_obj.freq
            self._valid_associations = detector_obj.valid_associations

    @property
    def count(self):
        """Minimal number of samples"""
        return self._count

    @property
    def freq(self):
        """Minimal frequency of samples"""
        return self._freq

    @property
    def normalize(self):
        """True if working with frequencies"""
        if self.freq is not None:
            return True
        if self.count is not None:
            return False

    @property
    def limit(self):
        """Minimal count or frequency"""
        limit = self.freq if self.normalize else self.count
        return limit

    @property
    def valid_associations(self) -> list:
        """List of valid associations"""
        return self._valid_associations

    @staticmethod
    def _calculate_valid_associations(df: pd.DataFrame,
                                      normalize,
                                      limit
                                      ) -> list:
        """Calculate valid associations"""
        col1 = df.columns[0]
        col2 = df.columns[1]

        crosstab = pd.crosstab(index=df[col1],
                               columns=[df[col2]],
                               normalize=normalize,
                               )

        gt_than_limit = (crosstab > limit).stack().reset_index()
        errors = df.merge(gt_than_limit, on=[col1, col2]).iloc[:, -1]

        assoc = (
            df[errors]
            .drop_duplicates()
            .reset_index()
            .iloc[:, -2:]
        )

        valid_associations = list(map(tuple, assoc.values))

        return valid_associations

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        df = self._obj.dropna()

        mask = ~df.apply(tuple, axis=1).isin(self.valid_associations)

        return df[mask].index

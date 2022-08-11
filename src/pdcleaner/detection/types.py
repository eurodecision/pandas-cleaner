"""
Detectors related to element types
"""

from pydoc import locate
import pandas as pd

from pdcleaner.detection._base import _SeriesDetector
from pdcleaner.utils.utils import raise_if_not_in


class types(_SeriesDetector):
    r"""Detect elements with type errors.

    Intended to be used by the detect method with the keyword 'types'

    >>> series.cleaner.detect.types(...)
    >>> series.cleaner.detect('types',...)

    This detection method flags elements as potential errors wherever the
    corresponding python type is different than the one specified.

    If no type is given, elements which don't share the type of the first row
    are flagged as errors.

    Note
    ----

    NA values are not treated as errors.

    Parameters
    ----------
    ptype : python built-in data type or None (Default)
        int, float, str, bool ...

    Raises
    ------
    TypeError
        when the given does not define a valid python built-in data type

    Examples
    --------
    >>> import pandas as pd
    >>> import pdcleaner

    >>> my_series = pd.Series([1, 2, 100, 3], dtype='float64')
    >>> my_series[1] = 'One'
    >>> my_detector = my_series.cleaner.detect.dtype(ptype=float)
    >>> print(my_detector.is_error())
    0    False
    1     True
    2    False
    3    False
    dtype: bool

    Missing values are not treated as errors.

    >>> my_series = pd.Series([1., 2., np.nan, 3.])
    >>> my_series[1] = 'One'
    >>> my_series[2] = np.nan
    >>> my_detector = my_series.cleaner.detect.type(ptype=int)
    >>> print(my_detector.is_error())
    0    False
    1     True
    2    False
    3    False
    dtype: bool

    If no type is specified, find elements whose types differ
    from the first one

    >>> my_series = pd.Series(['A', 2, np.nan, 'D'])
    >>> my_detector = my_series.cleaner.detect('type')
    >>> type(my_series[0])
    str
    >>>print(my_detector.ptype)
    str
    >>> print(my_detector.is_error())
    0    False
    1     True
    2    False
    3    False
    dtype: bool

    The first detector detects the right type as str

    >>> my_series = pd.Series(['A', 2, np.nan, 'D'])
    >>> my_series_test = pd.Series([1, 'Two'])
    >>> my_detector = my_series.cleaner.detect('type')
    >>> my_second_detector = my_series_test.cleaner.detect(my_detector)
    >>> print(my_second_detector.is_error())
    0     True
    1    False
    dtype: bool
    """
    name = 'types'

    def __init__(self, obj, detector_obj=None, ptype=None):
        super().__init__(obj)

        if not detector_obj:
            self._ptype = ptype
        else:
            self._ptype = detector_obj.ptype

        if not isinstance(self.ptype, type):
            raise TypeError("ptype bound must be a python built-in type")

    @property
    def ptype(self):
        """built-in python type"""
        if isinstance(self._ptype, str):
            return locate(self._ptype)

        if self._ptype is None:
            return type(self._obj[self._obj.index.min()])
        return self._ptype

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""
        mask = ~(self._obj.apply(lambda x: type(x)) == self.ptype)

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['ptype']


class castable(_SeriesDetector):
    r"""Detect elements that cannot be casted into target type.

    Intended to be used by the detect method with the keyword 'castable'

    >>> series.cleaner.detect.castable(...)
    >>> series.cleaner.detect('castable',...)

    This detection method flags elements in an object Series as errors
    when they can not be converted (casted) into a given type.

    For example,
    + '1.2' is not castable as an integer
    + 'ABC' is not castable as a number/float
    + '2022-02-28' is castable as a date, but '2022-02-31' is not

    The expected thousands and decimal separators can be customized, so that,
    for example: '100,000.0' can be seen as 100000.

    Note
    ----
    NA values are not treated as errors.

    Parameters
    ----------
    target: {"int", "number", "date", "boolean"}, required parameter
        The target type that the value will be casted into

    decimal: str, Optional
        Specific decimal separator in case of number type

    thousands: str, Optional
        Specific thousand separator in case of number type

    bool_values: dict,Optional
        Specific value of True and False in case of boolean type

    Raises
    ------
    ValueError
        When target is not among the allowed values, or target is not defined.
        When detector is applied to float, int series


    Examples
    --------
    >>> import pandas as pd
    >>> import pdcleaner

    >>> my_series = pd.Series(['1','2.0','1.2','A','100', '22/05/1975'], dtype='object')
    >>> my_detector = my_series.cleaner.detect('castable', target='int')
    >>> print(my_detector.is_error())
    0    False
    1    False
    2     True
    3     True
    4    False
    5     True
    dtype: bool

    >>> my_detector = my_series.cleaner.detect('castable', target='date')
    >>> print(my_detector.is_error())
    0     True
    1     True
    2     True
    3     True
    4     True
    5    False
    dtype: bool

    >>> my_detector = my_series.cleaner.detect('castable', target='number')
    >>> my_detector.is_error()
    0    False
    1    False
    2    False
    3    False
    4    False
    5     True
    dtype: bool

    >>> my_series = pd.Series(['Yes','No','No','Yes','Ok', 'Nok'], dtype='object')
    >>> my_detector = my_series.cleaner.detect('castable', target='boolean',
                                                bool_values={"Yes":True, "No":False})
    >>> my_detector.is_error()
    0    False
    1    False
    2    False
    3    False
    4     True
    5     True
    dtype: bool

    Missing values are not treated as errors.

    >>> my_series = pd.Series(['1','2.0','1.2','A','100', np.nan], dtype='object')
    >>> my_detector = my_series.cleaner.detect('castable', target='number')
    >>> print(my_detector.is_error())
    0    False
    1    False
    2    False
    3     True
    4    False
    5    False
    dtype: bool

    """
    name = 'castable'

    def __init__(self, obj, detector_obj=None, target=None, **kwargs):
        super().__init__(obj)

        if obj.dtype != 'object':
            raise TypeError("This detector is only for object series.")

        if not detector_obj:
            self._target = target
            self._thousands = kwargs.get('thousands')
            self._decimal = kwargs.get('decimal')
            self._bool_values = kwargs.get('bool_values')
        else:
            self._target = detector_obj.target
            self._thousands = detector_obj.thousands
            self._decimal = detector_obj.decimal
            self._bool_values = detector_obj.bool_values

        if self._target is None:
            raise ValueError("Target parameter must be defined")

        legal_values = ["int", "float", "date", "boolean"]
        raise_if_not_in(self._target, legal_values,
                        f"target must be in {', '.join(legal_values)}")

        if self._target == "date":
            if self._thousands or self._decimal:
                raise ValueError("Thousands/decimal separator parameter is not necessary to check "
                                 "if value is castable to date")

        if (self._target == "boolean") & (not self._bool_values):
            self._bool_values = {"True": True, "False": False}

    @property
    def target(self) -> str:
        """Target type that value will be checked"""
        return self._target

    @property
    def thousands(self) -> str:
        """Specific thousand separator"""
        return self._thousands

    @property
    def decimal(self) -> str:
        """Specific decimal separator"""
        return self._decimal

    @property
    def bool_values(self) -> dict:
        """Specific value for True and False"""
        return self._bool_values

    def check_separators(self, series: pd.Series) -> pd.Series:
        """Method to replace specific separator before applying detector"""
        processed_series = series.copy()
        if self.thousands:
            processed_series = processed_series.str.replace(self.thousands, '')

        if self.decimal:
            processed_series = processed_series.str.replace(self.decimal, '.')

        return processed_series

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""
        if self.target == "int":
            processed_series = self.check_separators(self._obj)
            mask = ~pd.to_numeric(processed_series, errors="coerce").astype(float).apply(lambda x: x.is_integer())

        elif self.target == "float":
            processed_series = self.check_separators(self._obj)
            mask = pd.to_numeric(processed_series, errors="coerce").isna()

        elif self.target == "date":
            mask = pd.to_datetime(self._obj, errors="coerce").isna()

        else:
            mask = ~(self._obj.isin(self.bool_values.keys()))

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['target']

"""
Basic detectors
"""
#pylint: disable=too-many-arguments

from cmath import isfinite
import warnings
import numbers

from typing import Callable

import numpy as np
import pandas as pd

from pdcleaner.detection._base import _SeriesDetector, _Detector, _NumericalSeriesDetector
from pdcleaner.utils.utils import raise_if_not_in, nb_of_args


def _raise_if_invalid_sided_or_inclusive_args(inclusive="both", sided="both"):
    legal_values = ["both", "neither", "left", "right"]
    raise_if_not_in(inclusive, legal_values,
                    f"inclusive must be in {legal_values}")
    legal_values = ["both", "left", "right"]
    raise_if_not_in(sided, legal_values,
                    f"sided must be in {legal_values}")


class bounded(_NumericalSeriesDetector):
    r"""Detect values outside of given bounds.

    Intended to be used by the detect method with the keyword 'bounded'

    >>> series.cleaner.detect.bounded(...)
    >>> series.cleaner.detect('bounded',...)


    This detection method flags values as potential errors wherever the
    corresponding Series element is outside the range between lower and upper.

    Note
    ----

    NA values are not treated as errors.

    Parameters
    ----------
    lower : float or -np.inf (Default)
        Lower bound
    upper : float or np.inf (Default)
        Upper bound
    inclusive : {“both”, “neither”, “left”, “right”}, default "both"
        Include boundaries. Whether to set each bound as closed or open.

    Raises
    ------
    Warning
        when neither lower, nor upper is specified
    ValueError
        when lower >= upper

    Examples
    --------

    >>> my_series = pd.Series([1, 2, 100, 3])
    >>> my_detector = my_series.cleaner.detect.bounded(lower=2, upper=4)
    >>> print(my_detector.is_error())
    0     True
    1    False
    2     True
    3    False
    dtype: bool

    With only one bound specified

    >>> my_series = pd.Series([1, 2, 100, 3])
    >>> my_detector = my_series.cleaner.detect.bounded(upper=4)
    >>> print(my_detector.is_error())
    0    False
    1    False
    2     True
    3    False
    dtype: bool

    Missing values are not treated as errors.

    >>> my_series = pd.Series([1, np.nan, 100, 3])
    >>> my_detector = my_series.cleaner.detect.bounded(lower=2, upper=4)
    >>> print(my_detector.is_error())
    0     True
    1    False
    2     True
    3    False
    dtype: bool
    """
    name = 'bounded'

    def _raise_if_non_numeric_bounds(self):
        if not isinstance(self.lower, numbers.Number):
            raise TypeError("Lower bound must be a number")
        if not isinstance(self.upper, numbers.Number):
            raise TypeError("Upper bound must be a number")

    def __init__(self,
                 obj,
                 detector_obj=None,
                 lower=np.NINF,
                 upper=np.inf,
                 inclusive="both"
                 ):
        super().__init__(obj)

        legal_values = ["both", "neither", "left", "right"]
        raise_if_not_in(inclusive, legal_values,
                        f"inclusive must be in {legal_values}")

        if not detector_obj:
            self._lower = lower
            self._upper = upper
            self._inclusive = inclusive
            self._sided = "both"
        else:

            self._lower = detector_obj.lower
            self._upper = detector_obj.upper
            self._inclusive = detector_obj.inclusive
            self._sided = detector_obj.sided

        self._raise_if_non_numeric_bounds()

        if np.isinf(self._lower) & np.isinf(self._upper):
            warnings.warn("Neither lower nor upper specified")

        if self._lower >= self._upper:
            raise ValueError("Lower bound is >= upper bound")

    @property
    def lower(self) -> float:
        """Lower bound"""
        if self.sided == "right":
            return np.NINF
        return self._lower

    @property
    def upper(self) -> float:
        """Upper bound"""
        if self.sided == "left":
            return np.inf
        return self._upper

    @property
    def inclusive(self) -> str:
        """Keyword to indicate if boundaries are included  {“both”, “neither”, “left”, “right”}"""
        return self._inclusive

    @property
    def sided(self) -> str:
        """Keyword to indicate if detection is one side or both {"both", "right", "left"}"""
        return self._sided

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        if self.inclusive == "both":
            mask = ~((self.lower <= self._obj) & (self._obj <= self.upper))
        elif self.inclusive == "neither":
            mask = ~((self.lower < self._obj) & (self._obj < self.upper))
        elif self.inclusive == "left":
            mask = ~((self.lower <= self._obj) & (self._obj < self.upper))
        elif self.inclusive == "right":
            mask = ~((self.lower < self._obj) & (self._obj <= self.upper))

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['lower', 'upper', 'inclusive', 'sided']


class length(_SeriesDetector):
    r"""Detect elements with length outside of given bounds.

    Intended to be used by the detect method with the keyword 'length'

    >>> series.cleaner.detect.length(...)
    >>> series.cleaner.detect('length',...)

    This detection method flags elements as potential errors wherever the corresponding length of
    Series element is outside the range between lower and upper. Alternatively, can be used
    with a fixed lenght value.

    Note
    ----
    NA values are not treated as errors.

    Parameters
    ----------
    lower : float or None (Default)
        Lower bound

    upper : float or None (Default)
        Upper bound

    value : float or None ( Default)
        Specific length of the element

    Raises
    ------
    TypeError
        when at least one of lower, upper or value is not a number
    ValueError
        when lower or upper is specified at the same time as value or 
        when none of the three is given
    ValueError
        when lower >= upper


    Examples
    --------
    >>> import pandas as pd
    >>> import pdcleaner

    >>> my_series = pd.Series(['75013','78000' , '931204', '952684'], dtype='string')
    >>> my_detector = my_series.cleaner.detect.length(value=5)
    >>> print(my_detector.is_error())
    0    False
    1    False
    2     True
    3     True
    dtype: bool

    with two bounds specified

    >>> my_series = pd.Series(['1','001' , '01460', '0011448'], dtype='string')
    >>> my_detector = my_series.cleaner.detect.length(lower=2, upper=6)
    >>> print(my_detector.is_error())
    0    True
    1    False
    2    False
    3    True
    dtype: bool

    Can be used with integers

    >>> my_series = pd.Series([1, 1234567 , 1460, np.nan])
    >>> my_detector = my_series.cleaner.detect.length(upper=6)
    >>> my_detector.is_error()
    0    False
    1     True
    2    False
    3    False
    dtype: bool

    and with floats

    >>> my_series = pd.Series([1.007, 1.234567 , 1.460], dtype='float64')
    >>> my_detector = my_series.cleaner.detect.length(upper=6)
    >>> my_detector.is_error()
    0    False
    1     True
    2    False
    dtype: bool

    Missing values are not treated as errors.

    >>> my_series = pd.Series(['1','001' , '01460', np.nan], dtype='string')
    >>> my_detector = my_series.cleaner.detect.length(upper=6)
    >>> print(my_detector.is_error())
    0     False
    1     False
    2     False
    3     False
    dtype: bool

    """
    name = 'length'

    def _raise_if_non_numeric_bounds(self):
        if not isinstance(self.lower, numbers.Number):
            raise TypeError("Lower bound must be a number")
        if not isinstance(self.upper, numbers.Number):
            raise TypeError("Upper bound must be a number")
        if not isinstance(self.value, numbers.Number):
            raise TypeError("Argument value must be a number")

    def __init__(self,
                 obj,
                 detector_obj=None,
                 lower=np.NINF,
                 upper=np.inf,
                 value=np.inf,
                 inclusive="both"
                 ):
        super().__init__(obj)

        legal_values = ["both", "neither", "left", "right"]
        raise_if_not_in(inclusive, legal_values,
                        f"inclusive must be in {legal_values}")

        if not detector_obj:
            self._lower = lower
            self._upper = upper
            self._value = value
            self._inclusive = inclusive
        else:
            self._lower = detector_obj.lower
            self._upper = detector_obj.upper
            self._value = detector_obj.value
            self._inclusive = detector_obj.inclusive

        self._raise_if_non_numeric_bounds()

        if(not np.isfinite(self._value)
           and not np.isfinite(self._lower)
           and not np.isfinite(self._upper)
           ):
            raise ValueError("At least one argument must be provided")

        if np.isfinite(self._value) and (np.isfinite(self._upper) or np.isfinite(self._lower)):
            raise ValueError("Incompatible arguments: value and upper or lower")

        if np.isfinite(self._value):
            self._mode = 'fixed_value'
        else:
            self._mode = 'bound'

        if self._lower >= self._upper:
            raise ValueError("Lower bound is >= upper bound")

    @property
    def mode(self) -> str:
        """"Checking mode"""
        return self._mode

    @property
    def lower(self) -> float:
        """Lower bound"""
        return self._lower

    @property
    def upper(self) -> float:
        """Upper bound"""
        return self._upper

    @property
    def value(self) -> float:
        """Fix length value"""
        return self._value

    @property
    def inclusive(self) -> str:
        """Keyword to indicate if boundaries are included  {“both”, “neither”, “left”, “right”}"""
        return self._inclusive

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""
        if self.mode == "fixed_value":
            mask = ~(self._obj.apply(lambda x: len(str(x)) == self.value))
        elif self.mode == "bound":
            if self.inclusive == "both":
                mask = ~(self._obj.apply(lambda x: self.lower <= len(str(x)) <= self.upper))
            elif self.inclusive == "neither":
                mask = ~(self._obj.apply(lambda x: self.lower < len(str(x)) < self.upper))
            elif self.inclusive == "left":
                mask = ~(self._obj.apply(lambda x: self.lower <= len(str(x)) < self.upper))
            elif self.inclusive == "right":
                mask = ~(self._obj.apply(lambda x: self.lower < len(str(x)) <= self.upper))

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['mode', 'lower', 'upper', 'value', 'inclusive']


class missing(_Detector):
    r"""Detect elements containing missing values

    Intended to be used by the detect method with the keyword 'missing'

    >>> df.cleaner.detect.missing(...)
    >>> df.cleaner.detect('missing',...)

    Parameters
    ----------
    how: string , default = 'any'
        - 'any' : detected as error if any NA values are present.
        - 'all' : detected as error if all values are NA.

    Raises
    ------
    ValueError
        when unknown value is given to how parameter


    Examples
    --------
    >>> import pandas as pd
    >>> import pdcleaner

    >>> df = pd.DataFrame({'col1' : ['Alice', 'Bob', 'Charles'],
                           'col2' : [15, np.nan, 11] })
    >>> my_detector = df.cleaner.detect.missing(how='any')
    >>> print(my_detector.is_error())
    0    False
    1    True
    2    False
    dtype: bool

    Checking if all values are NA

    >>> df = pd.DataFrame({'col1' : ['Alice', np.nan, 'Charles'],
                           'col2' : [np.nan, np.nan, np.nan] })
    >>> my_detector = df.cleaner.detect.missing(how='all')
    >>> print(my_detector.is_error())
    0    False
    1    True
    2    False
    dtype: bool

    Can be used with series. 'how' parameter is not necessary

    >>> my_series = pd.Series(['Alice', 'Bob', np.nan, 'Charles'])
    >>> my_detector = my_series.cleaner.detect('missing')
    >>> print(my_detector.is_error())
    0    False
    1    False
    2     True
    3    False
    dtype: bool

    """
    name = 'missing'

    def __init__(self,
                 obj,
                 detector_obj=None,
                 how='any',
                 ):
        super().__init__(obj)

        legal_values = ["any", "all"]
        raise_if_not_in(how, legal_values, f"how parameter must be {' or '.join(legal_values)}")

        if not detector_obj:
            self._how = how
        else:
            self._how = detector_obj.how

        self._type = type(obj)

    @property
    def how(self) -> str:
        """Checking mode"""
        return self._how

    @property
    def obj_type(self) -> str:
        """Type of object"""
        return 'series' if self._type == pd.core.series.Series else "dataframe"

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""
        if self.obj_type == "series":
            mask = self._obj.isna()
        elif self.obj_type == 'dataframe':
            if self.how == 'any':
                mask = self._obj.isnull().any(axis=1)
            elif self.how == 'all':
                mask = self._obj.isnull().all(axis=1)

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['how']


class duplicated(_Detector):
    r"""Detect duplicated elements

    Intended to be used by the detect method with the keyword 'duplicated'. Can be used with
    series or dataframe

    >>> df.cleaner.detect.duplicated(...)
    >>> df.cleaner.detect('duplicated',...)

    Parameters
    ----------
    subset : list of string, optional
        Column to be used for identifying duplicates

    keep : string or bool, default = 'first'
        - 'first' : detected as error duplicated elements except for the first occurence.
        - 'last' : detected as error duplicated elements except for the last occurence.
        - False: dectected as error all duplicated elements.

    Raises
    ------
    NameError
        When unknown value is given to keep parameter.
    KeyError
        When inexistant column name is given in subset.

    Examples
    --------
    >>> import pandas as pd
    >>> import pdcleaner

    >>> df = pd.DataFrame({'col1' : ['Alice', 'Bob', 'Alice', 'Bob', 'Alice'],
                           'col2' : [15, 13, 15, 10, 13] })
    >>> my_detector = df.cleaner.detect.duplicated(subset=['col1', 'col2'], keep='first')
    >>> print(my_detector.is_error())
    0    False
    1    False
    2     True
    3    False
    4    False
    dtype: bool

    >>> my_detector = df.cleaner.detect.duplicated(subset=['col1'], keep='last')
    >>> print(my_detector.is_error())
    0     True
    1     True
    2     True
    3    False
    4    False
    dtype: bool

    """
    name = 'duplicated'

    def __init__(self,
                 obj,
                 detector_obj=None,
                 subset=None,
                 keep='first'
                 ):
        super().__init__(obj)

        if not detector_obj:
            self._subset = subset
            self._keep = keep
        else:
            self._subset = detector_obj.subset
            self._keep = detector_obj.keep

    @property
    def subset(self) -> list:
        """List of subset column"""
        return self._subset

    @property
    def keep(self) -> str:
        """Which occurrence to consider as non duplicated"""
        return self._keep

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""
        if isinstance(self._obj, pd.Series):
            mask = self._obj.duplicated(keep=self.keep)
        else:
            mask = self._obj.duplicated(subset=self.subset, keep=self.keep)

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['subset', 'keep']


class custom(_Detector):
    r"""Detect errors using an user-defined callable

    Intended to be used by the detect method with the keyword 'custom'

    >>> df.cleaner.detect.custom(...)
    >>> df.cleaner.detect('custom',...)

    Parameters
    ----------
    error_func: Callable
        returns a boolean: True if the element/row is an error, False otherwise

    Raises
    ------
    TypeError:
        when error_func is not a callable
    ValueError
        when the number of arguments of error_func does not match the number of columns
    TypeError:
        when error_func does not return a boolean


    Examples
    --------
    >>> import pandas as pd
    >>> import pdcleaner

    with a lambda function

    >>> series = pd.Series([-1, 2, 3])
    >>> detector = series.cleaner.detect('custom', error_func=lambda x: x<0)
    >>> print(detector.is_error())
    0     True
    1    False
    2    False
    dtype: bool

    with a function

    >>> def f(x) -> bool:
            if x**2 > 5:
                return True
            return False
    >>> detector = series.cleaner.detect('custom', error_func=f)
    >>> print(detector.is_error())
    0    False
    1     True
    2     True
    dtype: bool

    with a dataframe, the callable should have the same number of inputs as the df.

    >>> df = pd.DataFrame({'col1' : [1,2,3], 'col2' : [1,3,9] })
    >>> bad_square = lambda x,y: x**2!=y
    >>> df.cleaner.detect('custom', error_func=bad_square).is_error()
    0    False
    1     True
    2    False
    dtype: bool

    """
    name = 'custom'

    def __init__(self,
                 obj,
                 detector_obj=None,
                 error_func=None,
                 ):

        super().__init__(obj)

        if not detector_obj:
            self._error_func = error_func
        else:
            self._error_func = detector_obj.error_func

        if self.error_func is None:
            raise ValueError('error_func must be defined')

        if not isinstance(self.error_func, Callable):
            raise TypeError('error_func sould be a callable')

        if isinstance(obj, pd.Series):
            n_cols = 1
        else:
            n_cols = len(obj.columns)

        if n_cols != nb_of_args(self.error_func):
            raise ValueError('error_func does not have the required number of arguments')

    @property
    def error_func(self):
        """Custom error function"""
        return self._error_func

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        if isinstance(self._obj, pd.Series):
            mask = self._obj.apply(self._error_func)
        else:
            mask = self._obj.apply(lambda x: self._error_func(*x), axis=1)

        if mask.dtype != bool:
            raise TypeError('error_func must return a boolean')

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return []


class quantiles(bounded):
    r"""Detect errors values in a Series using quantiles.

    Intended to be used by the detect method with the keyword 'quantiles'

    >>> series.cleaner.detect.quantiles(...)
    >>> series.cleaner.detect('quantiles',...)

    This detection method flags values as errors wherever the corresponding
    Series element is outside the range between the values at given quantiles

    Notes
    -----
    NA values are not treated as errors.

    Parameters
    ----------
    lowerq: float (Default = 0)
        The lower quantile, which can lie in range: 0 <= lowerq <= 1.
    upperq: float (Default = 1)
        The upper quantile, which can lie in range: 0 <= upperq <= 1.
    inclusive: {“both”, “neither”, “left”, “right”}, default "both"
        Include boundaries. Whether to set each bound as closed or open.

    Raises
    ------
    ValueError
        when lower or upperq are not in the range [0, 1]
    Warning
        when lowerq = 0 and higherq = 1

    Examples
    --------

    >>> s = pd.Series([0, 0, 0, 0, -1, 1, -1, 1, -5, 5])
    >>> q_errors = s.cleaner.detect.quantiles(lowerq=.1, upperq=.9)
    >>> q_errors.n_errors
    2

    """
    name = 'quantiles'

    def __init__(self,
                 obj,
                 detector_obj=None,
                 lowerq=0,
                 upperq=1,
                 inclusive="both"
                 ):
        super().__init__(obj, lower=np.nan, upper=np.nan)

        if not isinstance(lowerq, numbers.Number):
            raise ValueError("lowerq must be a number.")
        if not isinstance(upperq, numbers.Number):
            raise ValueError("upperq must be a number.")

        _raise_if_invalid_sided_or_inclusive_args(inclusive=inclusive)

        if not detector_obj:
            self._lowerq = lowerq
            self._upperq = upperq
            self._lower = self._obj.quantile(self.lowerq)
            self._upper = self._obj.quantile(self.upperq)
            self._inclusive = inclusive
            self._sided = "both"
        else:
            self._lowerq = detector_obj.lowerq
            self._upperq = detector_obj.upperq
            self._lower = detector_obj.lower
            self._upper = detector_obj.upper
            self._inclusive = detector_obj.inclusive
            self._sided = detector_obj._sided

        if (self._lowerq == 0) & (self._upperq == 1):
            warnings.warn("Neither lower or upper quantile specified")

        if self._lowerq >= self._upperq:
            raise ValueError("Lower quantile is >= upper quantile")

    @property
    def lowerq(self):
        """Lower quantile value"""
        return self._lowerq

    @property
    def upperq(self):
        """Upper quantile value"""
        return self._upperq

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['lowerq', 'upperq', 'lower', 'upper', 'inclusive', 'sided']

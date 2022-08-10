"""
Strings detectors
"""

import pandas as pd
from dateutil.parser import parse

from pdcleaner.detection._base import _DateTypeSeriesDetector
from pdcleaner.utils.utils import raise_if_not_in


class DateRangeDetector(_DateTypeSeriesDetector):
    r""" 'date_range': Detect if date value is between a given range.

    Intended to be used by the detect method with the keyword 'date_range'

    >>> series.cleaner.detect.date_range(...)
    >>> series.cleaner.detect('date_range',...)

    This detection method flags values as potential errors wherever the corresponding Series
    element is outside the date range.

    Note
    ----

    NA values are not treated as errors.

    Parameters
    ----------
    lower: datetime
        Lower bound
    upper : datetime
        Upper bound
    inclusive : {“both”, “neither”, “left”, “right”}, default "both"
        Include boundaries. Whether to set each bound as closed or open.

    Examples
    --------

    >>> my_series = pd.Series(['2022-10-01', '2021-06-11', '2019-04-03',' 2020-09-25'])
    >>> my_series= pd.to_datetime(my_series)
    >>> my_detector = my_series.cleaner.detect.date_range(lower='2020-06-15', upper='2022-08-05')
    >>> print(my_detector.is_error())
    0     True
    1    False
    2     True
    3    False
    dtype: bool

    With only one bound specified

    >>> my_detector = my_series.cleaner.detect.date_range(upper='2022-08-05')
    >>> print(my_detector.is_error())
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    """
    name = 'date_range'

    @staticmethod
    def is_date(date_str):
        """Check if value is in date format"""
        try:
            return bool(parse(date_str))
        except ValueError:
            return False

    def __init__(self,
                 obj,
                 detector_obj=None,
                 lower=pd.Timestamp.min,
                 upper=pd.Timestamp.max,
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
        else:
            self._lower = detector_obj.lower
            self._upper = detector_obj.upper
            self._inclusive = detector_obj.inclusive

        if self._lower != pd.Timestamp.min:
            if not self.is_date(self._lower):
                raise TypeError("Lower bound must be date format")

        if self._upper != pd.Timestamp.max:
            if not self.is_date(self._upper):
                raise TypeError("Upper bound must be date format")

        if (self._lower == pd.Timestamp.min) & (self._upper == pd.Timestamp.max):
            raise ValueError("Neither lower nor upper specified")

        if pd.to_datetime(self._lower) >= pd.to_datetime(self._upper):
            raise ValueError("Lower bound is >= upper bound")

    @property
    def lower(self) -> str:
        "Lower bound"
        return self._lower

    @property
    def upper(self) -> str:
        "Upper bound"
        return self._upper

    @property
    def inclusive(self) -> str:
        """Keyword to indicate if boundaries are included  {“both”, “neither”, “left”, “right”}"""
        return self._inclusive

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
        return ['lower', 'upper', 'inclusive']

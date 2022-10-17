"""
Base classes for detectors
"""
import re

from datetime import date, datetime

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_datetime64_dtype
from pdcleaner.utils.df_utils import (is_numerics_df,
                                      is_twocats_df,
                                      is_cats_df
                                      )

from pdcleaner.utils.report_utils import print_line, print_name_value, print_fixed_width


class _Detector():

    def __init__(self, obj):
        self._obj = obj

    @property
    def obj(self):
        """The object (Series or DataFrame) containing the data to which the detection is applied"""
        return self._obj

    def detected(self):
        """Series or DataFrame containing only the detected errors"""
        return self._obj.loc[self.index]

    def valid(self):
        """Series or DataFrame containing only the valid values"""
        return self._obj[self.not_error()]

    def is_error(self) -> pd.Series:
        """Return a boolean same-sized object indicating if the values are flagged as errors"""
        return pd.Series(self._obj.index.isin(self.index),
                         index=self._obj.index
                         )

    def not_error(self) -> pd.Series:
        """Return a boolean same-sized object indicating if the values are NOT flagged as errors"""
        return pd.Series(~self._obj.index.isin(self.index),
                         index=self._obj.index
                         )

    @property
    def n_errors(self) -> int:
        """Number of rows detected as errors"""
        return len(self.index)

    def __repr__(self):

        return (f"Detector\n- Method: {self.name}\n- Detected errors: {self.n_errors} among"
                f" {len(self._obj)} samples"
                f"\n(Use .report() for more details)"
                )

    def report(self):
        """prints a detection report"""

        param_dict = {
            'Method:': self.name,
            'Nb samples:': len(self._obj),
            'Date:': date.today().strftime('%B %d,%Y'),
            'Nb errors:': self.n_errors,
            'Time:': datetime.now().strftime('%H:%M:%S'),
            'Nb rows with NaN:': self.obj.isna().values.ravel().sum()
            }

        print(f"{'Detection report':^78}")
        print_line(symbol='=', nb=78)

        cpt = 0
        for name, value in param_dict.items():
            if (cpt % 2) == 0:
                print(f"{name: <18}{value: >18}"+" "*6, end='')
            else:
                print(f"{name: <18}{value: >18}")
            cpt += 1

        if hasattr(self, '_reported'):
            print_line(cpt=cpt, symbol='-', nb=78)

            cpt = 0
            for name in self._reported:
                if '_'+name in vars(self):
                    value = vars(self)['_'+name]

                    # Avoid error with method 'types'
                    if isinstance(value, type):
                        value = str(value.__name__)

                    if isinstance(value, re.Pattern):
                        value = 'compiled'

                    print_name_value(name=name, value=str(value), cpt=cpt)
                    cpt += 1

        if self.name == 'by_category':
            title = self._detector.name + " parameters"
            params = ['threshold', 'transform', 'inclusive', 'sided']

            print_line(cpt=cpt, symbol='-', nb=78)
            print(f"{title:^78}")

            cpt = 0
            for name in params:
                if '_'+name in vars(self._detector):
                    value = vars(self._detector)['_'+name]

                    print_name_value(name=name, value=str(value), cpt=cpt)
                    cpt += 1

        if hasattr(self, '_add_report'):

            title, params = self._add_report

            print_line(cpt=cpt, symbol='-', nb=78)
            print(f"{title:^78}")

            cpt = 0
            for name in params:
                if '_'+name in vars(self):
                    value = vars(self)['_'+name]

                    # Avoid error with method 'types'
                    if isinstance(value, type):
                        value = str(value.__name__)

                    if isinstance(value, re.Pattern):
                        value = 'compiled'

                    print_name_value(name=name, value=str(value), cpt=cpt)
                    cpt += 1

        if hasattr(self, '_report_comments'):
            print_line(cpt=cpt, symbol='-', nb=78)
            print_fixed_width(text=self._report_comments, width=78)
            cpt = 0

        print_line(cpt=cpt, symbol='=', nb=78)

        return None

    def has_errors(self) -> bool:
        """Returns True if any error has been detected, False otherwise"""
        return any(self.is_error())


class _SeriesDetector(_Detector):
    """Series base class detectors"""

    def __init__(self, obj):
        super().__init__(obj)

        if not isinstance(self._obj, pd.core.series.Series):
            raise TypeError("This detector applies to Series.")


class _NumericalSeriesDetector(_SeriesDetector):
    """Numerical Series base classes detectors"""

    def __init__(self, obj):
        super().__init__(obj)

        if not is_numeric_dtype(self._obj):
            raise TypeError("This detector applies to numerical Series.")


class _ObjectTypeSeriesDetector(_SeriesDetector):
    """Categorical or String Series base class detectors"""

    def __init__(self, obj):
        super().__init__(obj)

        if not is_object_dtype(self._obj):
            raise TypeError("This detector applies to "
                            "categorical/string Series.")


class _DateTypeSeriesDetector(_SeriesDetector):
    """Datetime Series base class detectors"""

    def __init__(self, obj):
        super().__init__(obj)

        if not is_datetime64_dtype(self._obj):
            raise TypeError("This detector applies to datetime Series")


class _DataFramesDetector(_Detector):
    """DataFrames base class detectors"""

    def __init__(self, obj):
        super().__init__(obj)

        if not isinstance(self._obj, pd.core.frame.DataFrame):
            raise TypeError("This detector applies to DataFrames.")


class _QuantiDataFramesDetector(_DataFramesDetector):
    """DataFrames with only numeric columns base detector"""

    def __init__(self, obj):
        super().__init__(obj)

        if not is_numerics_df(self._obj):
            raise TypeError("This detector applies to DataFrames "
                            "with numeric columns only.")


class _ObjectTypeDataFramesDetector(_DataFramesDetector):
    """DataFrames with only object (or categorical) columns"""

    def __init__(self, obj):
        super().__init__(obj)

        if not is_cats_df(self._obj):
            raise TypeError("This detector applies to DataFrames"
                            " with string/text/categorical columns")


class _TwoColsCategoricalDataFramesDetector(_ObjectTypeDataFramesDetector):
    """DataFrames with two object (or categorical) columns"""

    def __init__(self, obj):
        super().__init__(obj)

        if not is_twocats_df(self._obj):
            raise ValueError("This detector applies to DataFrames"
                             " with two object/categorical columns")


class _NumericalAndCategoricalDataFramesDetector(_DataFramesDetector):
    """DataFrames with one numerical column and one categorical column"""

    def __init__(self, obj):
        super().__init__(obj)

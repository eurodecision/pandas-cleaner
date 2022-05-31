"""Top-level package for Pandas Outliers."""

# __version__ = "unknown"
# try:
#     from ._version import __version__
# except ImportError:
#     # We're running in a tree that didn't come with a _version.py,
#     #  so we don't know what our version is.
#     pass

from inspect import getdoc, getmembers, isfunction
import pandas as pd

from pdcleaner.utils.df_utils import is_nums_and_cat_df
from pdcleaner.utils.utils import (add_method,
                                   all_subclasses,
                                   is_valid_detection_method_name)

from pdcleaner.detection._base import _Detector, _NumericalSeriesDetector

from pdcleaner.detection.basic import *
from pdcleaner.detection.strings import *
from pdcleaner.detection.web import *
from pdcleaner.detection.gaussian import *
from pdcleaner.detection.multivariate import *
from pdcleaner.detection.by_category import *
from pdcleaner.detection.types import *
from pdcleaner.detection.values import *
from pdcleaner.detection.datetimes import *


# Import plotting methods

from pdcleaner.plots.numseries import *
from pdcleaner.plots.keycollision import *
from pdcleaner.plots.multicategories import *
from pdcleaner.plots.freqandcount import *

# Import cleaning methods

from pdcleaner.cleaning import cleaning

# version
from .__version__ import __version__


# Define accessors: detect and clean


@pd.api.extensions.register_series_accessor("cleaner")
@pd.api.extensions.register_dataframe_accessor("cleaner")
class _ErrorsDetectAndClean:
    """
    A class representing a pandas DataFrame and Series accessor extension.
    cf. https://pandas.pydata.org/pandas-docs/stable/development/extending.html

    ...

    Attributes
    ----------
    _obj : pandas object : a Series or a DataFrame
        The pandas object decorated/handled by this accessor.
    detect : DetectorAccessor
        The object used for detection
    clean : CleanerAccessor
        The object used for cleaning

    Methods
    -------
    detect():
        detects the errors in the handled pandas object
        c.f. DetectorAccessor.__call__
    clean():
        cleans the pandas object using the specified method
        c.f. CleanerAccessor.__call__
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.detect = self.DetectorAccessor(self._obj)
        self.clean = self.CleanerAccessor(self._obj)

    class DetectorAccessor():
        """
        a callable class providing detection methods:

        Methods
        -------

        __call__(*args, **kwargs)
            methods to detect errors by providing the detector name

        Note
        ----
        multiple detection methods are added dynamically to the class
        """

        def __init__(self, obj):
            self._obj = obj

        def __call__(self, *args, **kwargs):
            """The function called when DetectorAccessor class is called.
            Mainly, when cleaner.detect is invoked

            Args:
                method (str or _Detector): The detection method to be used.
                *args: Variable length argument list.
                **kwargs: keyword arguments passed to the detector.

            Returns:
                a _Detector corresponding to the specified detection method

            Raises:
              TypeError: If the method is neither a string nor a _Detector.
              ValueError: If method is not a recognized detection method.

            """

            if not args:
                raise ValueError("A detection method must be provided")

            if isinstance(args[0], _Detector):
                method = args[0].name
                detector_obj = args[0]
            elif isinstance(args[0], str):
                method = args[0]
                detector_obj = None
            else:
                raise TypeError(f"{args[0]} is not "
                                "a valid argument for detect")

            if method not in detection_classes.keys():
                raise ValueError(f"{args[0]} is not a valid detection method."
                                 "Possible candidates are: "
                                 f"{', '.join(detection_classes.keys())}")

            if ((method in num_series_detector_classes)
               and (isinstance(self._obj, pd.DataFrame))):
                if is_nums_and_cat_df(self._obj):
                    detector_instance = \
                        detection_classes['by_category'](self._obj,
                                                         detector_obj=detector_obj,
                                                         method=method,
                                                         method_kwargs=kwargs)
                else:
                    raise TypeError("Dataframe must contain one numerical column"
                                    " and one categorical column")
            else:
                detector_instance = \
                    detection_classes[method](self._obj,
                                              detector_obj=detector_obj,
                                              **kwargs)

            return detector_instance

    class CleanerAccessor():
        """
        a callable class providing cleaning methods:

        Methods
        -------

        __call__(*args, **kwargs)
            methods to clean errors by providing the cleaning method name

        Note
        ----
            multiple cleaning methods are added dynamically to the class
        """

        def __init__(self, obj):
            self._obj = obj

        def __call__(self, method, detector_obj, **kwargs):
            """The function called when CleanerAccessor class is called.
            Mainly, when cleaner.clean is invoked

            Args:
              method (str): The cleaning method to be used.
              detector_obj (_Detector) : The detector used to spot the errors
              **kwargs: keyword arguments passed to the cleaning method.

            Returns:
              a _Detector corresponding to the specified detection method

            Raises:
              TypeError: If the provided method is not a string.
              ValueError: If the cleaning method is not a recognized one.

            """

            if not isinstance(method, str):
                raise TypeError(
                    f"{method} is not a valid argument for clean()")

            if method not in series_cleaning_methods.keys():
                raise ValueError(f"{method} is not a valid cleaning method."
                                 "Possible candidates are: "
                                 f"{', '.join(series_cleaning_methods.keys())}")

            cleaned_obj = \
                series_cleaning_methods[method](self,
                                                detector_obj=detector_obj,
                                                **kwargs)
            return cleaned_obj

    def detected(self, detector_obj):
        """Returns the errors detected by a detector"""
        return self._obj[detector_obj.is_error()]

    def valid(self, detector_obj):
        """Returns the values considered as valid (not errors) by a detector"""
        return self._obj[detector_obj.not_error()]


# Register detection methods

detection_classes = {}
for detector_subclass in all_subclasses(_Detector):
    try:
        detection_classes[detector_subclass.name] = detector_subclass
    except AttributeError:
        pass

num_series_detector_classes = []
for detector_subclass in all_subclasses(_NumericalSeriesDetector):
    try:
        num_series_detector_classes.append(detector_subclass.name)
    except AttributeError:
        pass


# allows calls like series.cleaner.detect.iqr(threshold=-1)
def register_detection_method(detector_name):
    """
    Creates a function
    Register it as a method accessor for pandas Series with the given name
    Retrieve and attach the docstring in the corresponding class
    """
    if detector_name not in detection_classes:
        raise ValueError(f"{detector_name} was not linked to a detector class."
                         " Check pdcleaner.detection_classes!")

    if not is_valid_detection_method_name(detector_name):
        raise ValueError(f"detector name '{detector_name}' is invalid!"
                         " name must be alphanumeric, start with a letter,"
                         " be all lowercase and may include underscores")

    @add_method(_ErrorsDetectAndClean.DetectorAccessor, detector_name)
    def method(self, **kwargs):
        return self(detector_name, **kwargs)
    method.__doc__ = getdoc(detection_classes[detector_name])


for detector_name in detection_classes:
    register_detection_method(detector_name)

# Register cleaning functions

list_series_cleaning_methods = ['clip', 'drop', 'to_na', 'replace', 'bykeys', 'cast', 'strip']
series_cleaning_methods = {}

for cleaning_member in getmembers(cleaning, isfunction):
    cleaning_function_name = cleaning_member[0]
    cleaning_function = cleaning_member[1]
    if cleaning_function_name in list_series_cleaning_methods:
        setattr(_ErrorsDetectAndClean.CleanerAccessor,
                cleaning_function_name,
                cleaning_function)
        cleaning_method = getattr(_ErrorsDetectAndClean.CleanerAccessor,
                                  cleaning_function_name)
        series_cleaning_methods[cleaning_function_name] = cleaning_method

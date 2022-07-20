import pandas as pd

from pdcleaner.detection._base \
    import _NumericalAndCategoricalDataFramesDetector


class ByCategoryDataFrameDetector(_NumericalAndCategoricalDataFramesDetector):
    r"""Detect errors in a numerical/categorical dataframe by applying a 1D
    numerical method to each category.

    Note
    ----

    This detector is not intended to be called directly,
    but when a 1D numerical method (such as iqr, zscore ...)
    detector is applied to a 2 cols num/cat dataframe (see example below)

    Parameters
    ----------
    method : str
        1d numerical method name, such as 'iqr', 'zscore' eg
    method_kwargs : dict
        Parameters for the 1d method

    Raises
    ------
    Value Error
        if called with a pre-existing detector object
    Errors
        depending on the 1d method

    Examples
    --------

    >>> import pandas as pd
    >>> import pdcleaner
    >>> df = pd.DataFrame({
            'col1' : [0, 0, 0, 0, -1, 1, -1, 1, -2, 2, 5,
                    6, 6, 6 ,6, 5, 7, 5, 7, 4, 8],
            'col2' : ['I'] * 11 + ["II"] * 10
            })
    >>> detector = df.cleaner.detect('iqr')
    >>> print(detector.detected)
        col1 col2
    10    5    1
    """

    name = "by_category"

    def __init__(self,
                 obj,
                 detector_obj=None,
                 method=None,
                 method_kwargs=None
                 ):

        super().__init__(obj)

        if not detector_obj:
            self._method = method
            self._method_kwargs = method_kwargs
        else:
            raise ValueError("This detection method can not be used "
                             "with an existing detector as an input.")

    @property
    def method(self):
        """Detection method by category"""
        return self._method

    @property
    def method_kwargs(self):
        """Parameters of the detection method"""
        return self._method_kwargs

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        # Missing values are not considered as errors
        df = self._obj.dropna()

        # Wich col is numerical ? which is a category ?
        num_col = df.select_dtypes(include='number').columns.item()
        cat_col = df.select_dtypes(exclude='number').columns.item()

        idxs = pd.Index([])

        for category in df[cat_col].unique():
            series = (df[df[cat_col] == category][num_col])
            self._detector =  \
                series.cleaner.detect(self._method,
                                      **self._method_kwargs
                                      )
            idxs = idxs.append(self._detector.index)

        return idxs

    @property
    def _reported(self):
        return ['method']

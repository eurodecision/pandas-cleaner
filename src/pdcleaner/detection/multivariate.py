"""
Multivariate detectors
"""

import numbers

import pandas as pd

from sklearn.cluster import DBSCAN as _DBSCAN


from pdcleaner.detection._base import _QuantiDataFramesDetector


class outliers(_QuantiDataFramesDetector):
    r"""Detects outliers in a  numeric DataFrame using a clustering DBScan algorithm

    This detection methods flags outliers in N-dimensional numerical datasets.
    The detection is performed using a density based clustering method DBScan (with
    its scikit-learn's implementation).

    The DBSCAN algorithm is performed on a column-scaled values of the initial datasets.
    A defaut set of rules is used for the DBSCAN parameters: eps is set to the max standard
    deviation of the scaled columns and min_samples is set to 2. These values
    can be modified to fit particular purposes.

    The samples that are not part of a cluster are flagged as potential errors.

    Rows with missing values are not considered and not flagged as errors.

    Parameters
    ----------
    eps: float
        The maximum euclidean distance between two samples in the normalized dataset
        for one to be considered as in the neighborhood of the other.
        By default, it is set to maximum standard deviation among all normalized variables.

    min_samples: int, default 2
        The number of samples to form a cluster.

    Raises
    ------
    TypeError
        when eps is not a number
        when min_samples is not an integer

    ValueError
        when sklearn's DBSCAN throws an exception for the given dataset and set of parameters

    References
    ----------

    [1] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Examples
    --------
    >>> import pandas as pd

    >>> df = pd.DataFrame({'x': [1, 1.1, 4],
                           'y': [1.1, 1, 4],
                           'z': [1, 1.1, 4]})
    >>> detector = df.cleaner.detect.outliers()
    >>> print(detector.is_error())
        0    False
        1    False
        2     True
        dtype: bool

    Rows with missing values are ignored and not flagged as errors

    >>> import numpy as np
    >>> df = pd.DataFrame({'x': [1, 1.1, 4, np.nan],
                           'y': [1.1, 1, 4, 5],
                           'z': [1, 1.1, 4, 5]})
    >>> detector = df.cleaner.detect.outliers()
    >>> print(detector.is_error())
        0    False
        1    False
        2     True
        3    False
        dtype: bool
    """
    name = 'outliers'

    def __init__(self, obj, detector=None, eps=None, min_samples=2):
        super().__init__(obj)

        if detector:
            raise ValueError("This detection method can not be used "
                             "with an existing detector as an input.")

        if (eps is not None) and (not isinstance(eps, numbers.Number)):
            raise TypeError("eps must be a number")

        if not isinstance(min_samples, int):
            raise TypeError("min_samples must be an integer")

        if not detector:
            if eps is None:
                self._eps = ((self._obj - self._obj.mean())
                             / self._obj.std()).std().max()
            else:
                self._eps = eps
            self._min_samples = min_samples

    @property
    def eps(self):
        """epsilon value see [1]"""
        return self._eps

    @property
    def min_samples(self):
        """min_samples value see [1]"""
        return self._min_samples

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        # Scale DataFrame columns
        df_norm = (self._obj - self._obj.mean()) / self._obj.std()

        # exclude rows containing at least one missing value
        df_norm.dropna(inplace=True)

        try:
            mask = (_DBSCAN(eps=self._eps,
                           min_samples=self._min_samples
                           )
                    .fit(df_norm)
                    .labels_ == -1
                    )
        except Exception:
            raise ValueError('DBScan error: see sklearn documentation for help')

        return df_norm[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['eps', 'min_samples']

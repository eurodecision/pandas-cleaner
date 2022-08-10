"""
Gaussian detectors
"""

import numbers
import warnings

import numpy as np
import pandas as pd

from scipy import stats

from pdcleaner.detection.basic import BoundedSeriesDetector
from pdcleaner.detection.basic import _raise_if_invalid_sided_or_inclusive_args
from pdcleaner.utils.utils import raise_if_not_in


def _inverse_boxcox(xt: float, shift: float = 0, lambda_: float = 1) -> float:
    """Box-cox inverse transformation with a shift to deal only with positive values"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if lambda_ == 0:
            X = np.exp(xt) - shift
        else:
            X = ((xt) * lambda_ + 1) ** (1 / lambda_) - shift
    return X


def _inverse_yeojohnson(xt: float, lambda_: float = 1) -> float:
    """Yeo-Johnson inverse transformation"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if xt >= 0 and lambda_ == 0:
            X = np.exp(xt) - 1
        elif xt >= 0 and lambda_ != 0:
            X = (xt * lambda_ + 1) ** (1 / lambda_) - 1
        elif xt < 0 and lambda_ != 2:
            X = 1 - (-(2 - lambda_) * xt + 1) ** (1 / (2 - lambda_))
        elif xt < 0 and lambda_ == 2:
            X = 1 - np.exp(-xt)
    return X


class _GaussianSeriesDetector(BoundedSeriesDetector):
    r"Base class for outliers detection based on a normal/gaussian distribution assumption."

    name = 'gaussian'

    def __init__(self,
                 obj,
                 detector_obj=None,
                 threshold=1.5,
                 inclusive="both",
                 sided="both",
                 normaltest='ignore',
                 pvalue=1e-3,
                 transform=None,
                 ):

        super().__init__(obj, lower=np.nan, upper=np.nan)

        _raise_if_invalid_sided_or_inclusive_args(inclusive=inclusive,
                                                  sided=sided
                                                  )

        valid_normaltest = ['ignore', 'warn', 'error']
        raise_if_not_in(normaltest,
                        valid_normaltest,
                        f"normaltest must be {' or '.join([str(v) for v in valid_normaltest])}")

        valid_transform = [None, 'boxcox', 'yeojohnson']
        raise_if_not_in(transform,
                        valid_transform,
                        f"transform must be {' or '.join([str(v) for v in valid_transform])}"
                        )

        if not isinstance(pvalue, numbers.Number):
            raise TypeError('pvalue must be a number')
        if pvalue <= 0:
            raise ValueError('pvalue must be positive')

        if not detector_obj:
            self._threshold = threshold
            self._inclusive = inclusive
            self._sided = sided
            self._normaltest = normaltest
            self._pvalue = pvalue
            self._transform = transform
            if len(self.obj) > 8:
                _, self._pval = stats.normaltest(self.obj)
                self._isnormal = self._pval > self.pvalue
            else:
                self._isnormal = False
        else:
            self._threshold = detector_obj.threshold
            self._inclusive = detector_obj.inclusive
            self._sided = detector_obj._sided
            self._normaltest = detector_obj.normaltest
            self._pvalue = detector_obj.pvalue
            self._transform = detector_obj.transform
            self._isnormal = detector_obj.isnormal

        if not isinstance(self._threshold, numbers.Number):
            print(self._threshold)
            raise TypeError("Threshold must be a number")
        if self._threshold < 0:
            raise ValueError("Threshold must be >= 0")

        if not self.isnormal:
            if len(self.obj) <= 8:
                self._msg = "Not enough rows to test normality. Must be > 8"
            else:
                self._msg = "Series distribution is not normal/gaussian"
            if self.normaltest == 'warn':
                warnings.warn(self._msg)
            elif self.normaltest == 'error':
                raise Exception(self._msg)
        else:
            self._msg = f"Series distribution has been tested as normal with p={self.pvalue}"

        self._transformations = \
            {"boxcox": lambda x: stats.boxcox(x + 1 - self._obj.min()),
             "yeojohnson": stats.yeojohnson,
             }

        if self.transform is not None and not self.isnormal:
            self._transformed, self._lmbda = self._transformations[self._transform](self._obj)
            self._transformed = pd.Series(self._transformed)

        self._inverse_tranf = \
            {"boxcox": lambda x: _inverse_boxcox(x,
                                                 shift=(1 - self._obj.min()),
                                                 lambda_=self._lmbda
                                                 ),
             "yeojohnson": lambda x: _inverse_yeojohnson(x, lambda_=self._lmbda)
             }

    @property
    def threshold(self):
        """Threshold value used to detect outliers"""
        return self._threshold

    @property
    def normaltest(self) -> str:
        """Normality test result behavior"""
        return self._normaltest

    @property
    def pvalue(self) -> float:
        """pvalue for normality test"""
        return self._pvalue

    @property
    def isnormal(self) -> bool:
        """is the series normal/gaussian according to the test ?"""
        return self._isnormal

    @property
    def transform(self) -> str or None:
        """Distribution transformation"""
        return self._transform

    @property
    def lmbda(self):
        """The lambda that maximizes the log-likelihood function of the transformation"""
        return self._lmbda

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['lower', 'upper', 'inclusive', 'sided']

    @property
    def _report_comments(self) -> str:
        """Comments section in detection reports"""
        if self.transform is not None:
            if not self.isnormal:
                return self._msg + f" (A {self.transform} transformation has been applied)"
            else:
                return self._msg + "(The transformation has not been applied)"
        return self._msg


class IqrSeriesDetector(_GaussianSeriesDetector):
    r"""'iqr': Detect outliers as potential errors in a Series using the IQR method.

    Intended to be used by the detect method with the keyword 'iqr'

    >>> series.cleaner.detect.iqr(...)
    >>> series.cleaner.detect('iqr',...)

    This detection method flag values as errors wherever the corresponding Series element
    is outside the range defined by:

    [Q25 - threshold x IQR; Q75 + threshold x IQR]

    where

    - Q25: the 25th percentile

    - Q75: the 75th percentile

    - IQR = Q75 - Q25: the interquartile range (the 75th percentile minus the 25th percentile)

    - threshold: cut-off value (default: 1.5)

    The upper and lower are the values of the whiskers extremities in a boxplot.

    Note
    ----
    NA values are not treated as errors.

    Warning
    -------
    A normality test is performed
    [see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html ]

    If the distribution does not follow a gaussian/normal distribution:

    - This is ignored if `normaltest='ignore'` (default)

    - A warning is raised if `normaltest = 'warning'`

    - An exception is raised if `normaltest = 'error'`

    If the series length is no more than 8, it is considered as not normal

    Tip
    ---
    The series can be "normalized" before applying the detector, using a power-series
    transformation:

    - Box-cox with a shift to deal with positive values [see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html]

    - Yeo-Johnson [see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html]
    Using the `scipy.stats` implementations, the optimal parameter lambda is
    calculated and used for the transformations and its inverse functions.

    When a transformation is applied, some parameters are expressed for the
    transformed series and informations are made available via `report()`.

    Parameters
    ----------
    threshold: float, default 1.5
    inclusive: {“both”, “neither”, “left”, “right”}, default "both"
        Include boundaries. Whether to set each bound as closed or open.
    sided: {“both”, “left”, “right”}, default "both"
        Specifies which limits should be applied.
        If "left", only apply lower limit
        If "right", only apply upper limit
        If "both", apply both upper and lower limits
    normaltest: {'ignore', 'warn', 'error'} default: 'ignore'
        wether to ignore, raise a warning or raise en exception if the normality test fails
    pvalue: float, default 1e-3
        pvalue associated with the normality test

    Raises
    ------
    TypeError
        when threshold is not a number
    TypeError
        when pvalue is not a number
    ValueError
        when threshold is negative
    ValueError
        when pvalue is negative
    ValueError
        if sided or inclusive has an unvalid value
    ValueError
        if normaltest is not 'ignore', 'warn' or 'error'
    UserWarning
        if the series is not normal and normaltest = 'warn'
    Exception
        if the series is not normal and normaltest = 'error'

    Examples
    --------

    >>> my_series = pd.Series([1,2,100,3])
    >>> my_detector = my_series.cleaner.detect.iqr()
    >>> print(my_detector.is_error())
        0    False
        1    False
        2     True
        3    False
        dtype: bool

    Using a transformation

    >>> s = pd.Series([0, 0, 0, 0, -100, 1, -1, 1, -6, 6])
    >>> iqr_detector = s.cleaner.detect.iqr(transform='boxcox')
    >>> iqr_detector.report()
                                Detection report
    ==============================================================================
    Method:                          iqr      Nb samples:                       10
    Date:                  March 23,2022      Nb errors:                         3
    Time:                       10:25:35      Nb rows with NaN:                  0
    ------------------------------------------------------------------------------
    lower             -3.045719539555506      upper             2.9570647561524908
    inclusive                       both      sided                           both
    ------------------------------------------------------------------------------
                    iqr parameters after boxcox transformation
    q25                7102.650530146201      q75                7325.937837774465
    iqr                 223.287307628264      threshold                        1.5
    transform                     boxcox      lmbda             2.0840437865755472
    ------------------------------------------------------------------------------
    Series distribution is not normal/gaussian (A boxcox transformation has been
    applied)
    ==============================================================================

    If the series is tested as normal, the transformation is not useful hence not applied

    >>> s = pd.Series([0, 0, 0, 0, -1, 1, -1, 1, -6, 6])
    >>> iqr_detector = s.cleaner.detect.iqr(transform='boxcox')
    >>> iqr_detector.report()
                                Detection report
    ==============================================================================
    Method:                          iqr      Nb samples:                       10
    Date:                  March 23,2022      Nb errors:                         2
    Time:                       10:12:38      Nb rows with NaN:                  0
    ------------------------------------------------------------------------------
    lower                           -3.0      upper                            3.0
    inclusive                       both      sided                           both
    ------------------------------------------------------------------------------
                                    iqr parameters
    q25                            -0.75      q75                             0.75
    iqr                              1.5      threshold                        1.5
    ------------------------------------------------------------------------------
    Series distribution has been tested as normal with p=0.001(The transformation
    has not been applied)
    ==============================================================================
    """
    name = 'iqr'

    def __init__(self,
                 obj,
                 detector_obj=None,
                 threshold=1.5,
                 inclusive="both",
                 sided="both",
                 normaltest='ignore',
                 pvalue=1e-3,
                 transform=None,
                 ):

        super().__init__(obj,
                         detector_obj=detector_obj,
                         threshold=threshold,
                         inclusive=inclusive,
                         sided=sided,
                         normaltest=normaltest,
                         pvalue=pvalue,
                         transform=transform,
                         )

        if not detector_obj:
            if transform is None or self.isnormal:
                self._q25 = self._obj.quantile(.25)
                self._q75 = self._obj.quantile(.75)
            else:
                self._q25 = self._transformed.quantile(.25)
                self._q75 = self._transformed.quantile(.75)
        else:
            self._q25 = detector_obj.q25
            self._q75 = detector_obj.q75

        self._iqr = self.q75 - self.q25

        self._lower = self.q25 - self.threshold * self._iqr
        self._upper = self.q75 + self.threshold * self._iqr
        if transform is not None and not self.isnormal:
            self._lower = self._inverse_tranf[self._transform](self._lower)
            if np.isnan(self._lower):
                self._lower = np.NINF
            self._upper = self._inverse_tranf[self._transform](self._upper)
            if np.isnan(self._upper):
                self._upper = np.inf

    @property
    def q75(self):
        """Value of the 75th percentile used to detect outliers"""
        return self._q75

    @property
    def q25(self):
        """Value of the 25th percentile used to detect outliers"""
        return self._q25

    @property
    def iqr(self):
        """Value of the interquantile range used to detect outliers"""
        return self._iqr

    @property
    def _add_report(self):
        """Additional reports"""
        title = f"{self.name} parameters"
        if self.transform is not None and not self.isnormal:
            title += f" after {self.transform} transformation"
        params = ['q25', 'q75', 'iqr', 'threshold']
        if self.transform is not None and not self.isnormal:
            params += ['transform', 'lmbda']
        return title, params


class ZScoreSeriesDetector(_GaussianSeriesDetector):
    r"""'zscore': Detect outliers as potential errors in a Series using the Z-score method.

    Intended to be used by the detect method with the keyword 'zscore'

    >>> series.cleaner.detect.zscore(...)
    >>> series.cleaner.detect('zscore',...)

    This detection method flag values as errors wherever the corresponding
    Series element has a Z-score above a given threshold.

    Z-scores are the number of standard deviations above and below the mean that each value falls.

    Z = (value - mean) / (standard deviation)

    Z-scores are used to quantify the unusualness of an observation
    when data follow a normal distribution.
    The further away an observation’s Z-score is from zero, the more unusual it is.

    Standard cut-off values (thresholds) for finding outliers are Z-scores of

    - +/- 1.96 corresponding to a 5% confidence that the value is an outlier (default here)

    - +/-3 often used in practice

    Note
    ----
    NA values are not treated as errors.

    Warning
    -------
    A normality test is performed
    [see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html ]

    If the distribution does not follow a gaussian/normal distribution:

    - This is ignored if `normaltest='ignore'` (default)

    - A warning is raised if `normaltest = 'warning'`

    - An exception is raised if `normaltest = 'error'`

    If the series length is no more than 8, it is considered as not normal

    Tip
    ---
    The series can be "normalized" before applying the detector, using a power-series
    transformation:

    - Box-cox with a shift to deal with positive values [see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html]

    - Yeo-Johnson [see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html]
    Using the `scipy.stats` implementations, the optimal parameter lambda is
    calculated and used for the transformations and its inverse functions.

    When a transformation is applied, some parameters are expressed for the
    transformed series and informations are made available via `report()`.

    Parameters
    ----------
    threshold: float, default 1.96
    inclusive: {“both”, “neither”, “left”, “right”}, default "both"
        Include boundaries. Whether to set each bound as closed or open.
    sided: {“both”, “left”, “right”}, default "both"
        Specifies which limits should be applied.
        If "left", only apply lower limit
        If "right", only apply upper limit
        If "both", apply both upper and lower limits
    normaltest: {'ignore', 'warn', 'error'} default: 'ignore'
        wether to ignore, raise a warning or raise en exception if the normality test fails
    pvalue: float, default 1e-3
        pvalue associated with the normality test

    Raises
    ------
    TypeError
        when threshold is not a number
    TypeError
        when pvalue is not a number
    ValueError
        when threshold is negative
    ValueError
        when pvalue is negative
    ValueError
        if sided or inclusive has an unvalid value
    ValueError
        if normaltest is not 'ignore', 'warn' or 'error'
    UserWarning
        if the series is not normal and normaltest = 'warn'
    Exception
        if the series is not normal and normaltest = 'error'

    Examples
    --------

    >>> s = pd.Series([0, 0, 0, 0, -1, 1, -1, 1, -6, 6])
    >>> zscore_detector = s.cleaner.detect.zscore()
    >>> zscore_detector.n_errors
    2

    >>> zscore_detector.lower, zscore_detector.upper
    (-4.800999895855028, 4.800999895855028)

    >>> s_test = pd.Series([1, 100])
    >>> s_test.cleaner.detect(zscore_detector).is_error()
    0    False
    1     True
    dtype: bool

    Using a transformation

    >>> s = pd.Series([0, 0, 0, 0, -100, 1, -1, 1, -6, 6])
    >>> zscore_detector = s.cleaner.detect.modzscore(transform='boxcox')
    >>> zzscore_detector.report()
                                Detection report
    ==============================================================================
    Method:                       zscore      Nb samples:                       10
    Date:                  March 23,2022      Nb errors:                         1
    Time:                       10:22:20      Nb rows with NaN:                  0
    ------------------------------------------------------------------------------
    lower             -47.08667824994567      upper             23.077126492888723
    inclusive                       both      sided                           both
    ------------------------------------------------------------------------------
                    zscore parameters after boxcox transformation
    mean               6513.202636676919      std               2328.4345612025554
    threshold                       1.96      transform                     boxcox
    lmbda             2.0840437865755472
    ------------------------------------------------------------------------------
    Series distribution is not normal/gaussian (A boxcox transformation has been
    applied)
    ==============================================================================

    If the series is tested as normal, the transformation is not useful hence not applied

    >>> s = pd.Series([0, 0, 0, 0, -1, 1, -1, 1, -6, 6])
    >>> zscore_detector = s.cleaner.detect.zscore(transform='boxcox')
    >>> zscore_detector.report()
                                Detection report
    ==============================================================================
    Method:                       zscore      Nb samples:                       10
    Date:                  March 23,2022      Nb errors:                         2
    Time:                       10:15:30      Nb rows with NaN:                  0
    ------------------------------------------------------------------------------
    lower             -5.695627952893147      upper              5.695627952893147
    inclusive                       both      sided                           both
    ------------------------------------------------------------------------------
                                zscore parameters
    mean                             0.0      std               2.9059326290271157
    threshold                       1.96
    ------------------------------------------------------------------------------
    Series distribution has been tested as normal with p=0.001(The transformation
    has not been applied)
    ==============================================================================
    """
    name = 'zscore'

    def __init__(self,
                 obj,
                 detector_obj=None,
                 threshold=1.96,
                 inclusive="both",
                 sided="both",
                 normaltest='ignore',
                 pvalue=1e-3,
                 transform=None,
                 ):

        super().__init__(obj,
                         detector_obj=detector_obj,
                         threshold=threshold,
                         inclusive=inclusive,
                         sided=sided,
                         normaltest=normaltest,
                         pvalue=pvalue,
                         transform=transform,
                         )

        if not detector_obj:
            if self.transform is None or self.isnormal:
                self._mean = self._obj.mean()
                self._std = self._obj.std()
            else:
                self._mean = self._transformed.mean()
                self._std = self._transformed.std()
        else:
            self._mean = detector_obj.mean
            self._std = detector_obj.std

        self._lower = self.mean - self.threshold * self.std
        self._upper = self.mean + self.threshold * self.std

        if transform is not None and not self.isnormal:
            self._lower = self._inverse_tranf[self._transform](self._lower)
            if np.isnan(self._lower):
                self._lower = np.NINF
            self._upper = self._inverse_tranf[self._transform](self._upper)
            if np.isnan(self._upper):
                self._upper = np.inf

    @property
    def mean(self) -> float:
        """Mean value used to calculate Z-scores"""
        return self._mean

    @property
    def std(self) -> float:
        """Standard deviation used to calculate Z-scores"""
        return self._std

    @property
    def _add_report(self):
        """Additional reports"""
        title = f"{self.name} parameters"
        if self.transform is not None and not self.isnormal:
            title += f" after {self.transform} transformation"
        params = ['mean', 'std', 'threshold', ]
        if self.transform is not None and not self.isnormal:
            params += ['transform', 'lmbda']
        return title, params


class ModZScoreSeriesDetector(_GaussianSeriesDetector):
    r"""'modzscore': Detect outliers as potential errors in a Series using the modified Z-score.

    Intended to be used by the detect method with the keyword 'modzscore'

    >>> series.cleaner.detect.modzscore(...)
    >>> series.cleaner.detect('modzscore',...)

    This detection method flag values as errors wherever the corresponding
    Series element has a modified Z-score above a given threshold.

    The modified Z-scores is used to quantify the unusualness of an observation
    when data follow a normal distribution. It is defined as:

    modified Z score = 0.6745 * (value - median) / (median absolute deviation)

    The further away an observation’s modified Z-score is from zero, the more unusual it is.

    A modified Z-score is more robust than a Z-score because it uses the median as opposed
    to the mean, which is known to be influenced by outliers.

    The standard cut-off values (threshold) for finding outliers are modified Z-scores
    of +/- 3.5 (default here).

    Note
    ----
    NA values are not treated as errors.

    Warning
    -------
    A normality test is performed
    [see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html ]

    If the distribution does not follow a gaussian/normal distribution:

    - This is ignored if `normaltest='ignore'` (default)

    - A warning is raised if `normaltest = 'warning'`

    - An exception is raised if `normaltest = 'error'`

    If the series length is no more than 8, it is considered as not normal

    Tip
    ---
    The series can be "normalized" before applying the detector, using a power-series
    transformation:

    - Box-cox with a shift to deal with positive values [see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html]

    - Yeo-Johnson [see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html]
    Using the `scipy.stats` implementations, the optimal parameter lambda is
    calculated and used for the transformations and its inverse functions.

    When a transformation is applied, some parameters are expressed for the
    transformed series and informations are made available via `report()`.

    Parameters
    ----------
    threshold: float, default 3.5
    inclusive: {“both”, “neither”, “left”, “right”}, default "both"
        Include boundaries. Whether to set each bound as closed or open.
    sided: {“both”, “left”, “right”}, default "both"
        Specifies which limits should be applied.
        If "left", only apply lower limit
        If "right", only apply upper limit
        If "both", apply both upper and lower limits
    normaltest: {'ignore', 'warn', 'error'} default: 'ignore'
        wether to ignore, raise a warning or raise en exception if the normality test fails
    pvalue: float, default 1e-3
        pvalue associated with the normality test

    Raises
    ------
    TypeError
        when threshold is not a number
    TypeError
        when pvalue is not a number
    ValueError
        when threshold is negative
    ValueError
        when pvalue is negative
    ValueError
        if sided or inclusive has an unvalid value
    ValueError
        if normaltest is not 'ignore', 'warn' or 'error'
    UserWarning
        if the series is not normal and normaltest = 'warn'
    Exception
        if the series is not normal and normaltest = 'error'

    Examples
    --------

    >>> s = pd.Series([0, 0, 0, 0, -1, 1, -1, 1, -6, 6])
    >>> modzscore_detector = s.cleaner.detect.modzscore()
    >>> modzscore_detector.n_errors
    2

    >>> modzscore_detector.lower, modzscore_detector.upper
    (-5.405405405405405, 5.405405405405405)

    >>> s_test = pd.Series([1, 100])
    >>> s_test.cleaner.detect(modzscore_detector).is_error()
    0    False
    1     True
    dtype: bool

    Using a transformation

    >>> s = pd.Series([0, 0, 0, 0, -100, 1, -1, 1, -6, 6])
    >>> modzscore_detector = s.cleaner.detect.modzscore(transform='boxcox')
    >>> modzscore_detector.report()
                                Detection report
    ==============================================================================
    Method:                    modzscore      Nb samples:                       10
    Date:                  March 23,2022      Nb errors:                         3
    Time:                       10:19:07      Nb rows with NaN:                  0
    ------------------------------------------------------------------------------
    lower             -5.571754332925835      upper              5.256895307993972
    inclusive                       both      sided                           both
    ------------------------------------------------------------------------------
                modzscore parameters after boxcox transformation
    median             7213.695043599123      mad               148.85820508550978
    threshold                        3.5      transform                     boxcox
    lmbda             2.0840437865755472
    ------------------------------------------------------------------------------
    Series distribution is not normal/gaussian (A boxcox transformation has been
    applied)
    ==============================================================================

    If the series is tested as normal, the transformation is not useful hence not applied

    >>> s = pd.Series([0, 0, 0, 0, -1, 1, -1, 1, -6, 6])
    >>> modzscore_detector = s.cleaner.detect.modzscore(transform='yeojohnson')
    >>> modzscore_detector.report()
                                Detection report
    ==============================================================================
    Method:                    modzscore      Nb samples:                       10
    Date:                  March 23,2022      Nb errors:                         2
    Time:                       10:16:45      Nb rows with NaN:                  0
    ------------------------------------------------------------------------------
    lower             -5.405405405405405      upper              5.405405405405405
    inclusive                       both      sided                           both
    ------------------------------------------------------------------------------
                                modzscore parameters
    median                           0.0      mad                              1.0
    threshold                        3.5
    ------------------------------------------------------------------------------
    Series distribution has been tested as normal with p=0.001(The transformation
    has not been applied)
    ==============================================================================
    """
    name = 'modzscore'

    def __init__(self,
                 obj,
                 detector_obj=None,
                 threshold=3.5,
                 inclusive="both",
                 sided="both",
                 normaltest='ignore',
                 pvalue=1e-3,
                 transform=None,
                 ):

        super().__init__(obj,
                         detector_obj=detector_obj,
                         threshold=threshold,
                         inclusive=inclusive,
                         sided=sided,
                         normaltest=normaltest,
                         pvalue=pvalue,
                         transform=transform,
                         )

        if not detector_obj:
            if self.transform is None or self.isnormal:
                self._median = self._obj.median()
                self._mad = ((self._obj - self._obj.median()).abs()).median()
            else:
                self._median = self._transformed.median()
                self._mad = ((self._transformed - self._transformed.median()).abs()).median()
        else:
            self._median = detector_obj.median
            self._mad = detector_obj.mad

        self._lower = self.median - self.threshold / 0.6475 * self.mad
        self._upper = self.median + self.threshold / .6475 * self.mad

        if transform is not None and not self.isnormal:
            self._lower = self._inverse_tranf[self._transform](self._lower)
            if np.isnan(self._lower):
                self._lower = np.NINF
            self._upper = self._inverse_tranf[self._transform](self._upper)
            if np.isnan(self._upper):
                self._upper = np.inf

    @property
    def median(self) -> float:
        """Median value used to calculate modified Z-scores"""
        return self._median

    @property
    def mad(self) -> float:
        """ Median absolute deviation of the distribution used to calculate modified Z-scores"""
        return self._mad

    @property
    def _add_report(self):
        """Additional reports"""
        title = f"{self.name} parameters"
        if self.transform is not None and not self.isnormal:
            title += f" after {self.transform} transformation"
        params = ['median', 'mad', 'threshold', ]
        if self.transform is not None and not self.isnormal:
            params += ['transform', 'lmbda']
        return title, params

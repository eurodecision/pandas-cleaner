r"""Web related detection methods:
        * email: Detect strings that do not match an email
        * url: Detect strings that do not match a url
        * ping: Detect strings that do not match a reachable url
"""

import re
import requests

import pandas as pd

from pdcleaner.detection.strings import PatternSeriesDetector
from pdcleaner.detection._base import _ObjectTypeSeriesDetector


class EmailSeriesDetector(PatternSeriesDetector):
    r"""'email': Detect strings that do not match an email.

    Intended to be used by the detect method with the keyword 'email'

    >>> series.cleaner.detect.email(...)
    >>> series.cleaner.detect('email',...)

    This detection method flags values as potential errors wherever the
    corresponding Series element does not match an email.

    Note
    ----
    Missing values (NaN) are not treated as errors

    Examples
    --------

    >>> my_series = pd.Series(['john_856_doe@gmail.com','john_doe','np.nan','john?doe@gmail.com'])
    >>> my_detector = my_series.cleaner.detect.email()
    >>> print(my_detector.detected)
    1              john_doe
    2                np.nan
    3    john?doe@gmail.com
    dtype: object
    """

    name = 'email'

    def __init__(self, obj, detector_obj=None):
        pattern_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        super().__init__(obj, detector_obj, pattern=pattern_email, mode="fullmatch")

    @property
    def _reported(self):
        r"""Generates a report of the detection"""
        return []


class UrlSeriesDetector(PatternSeriesDetector):
    r"""'url': Detect strings that do not match a url.

    Intended to be used by the detect method with the keyword 'url'

    >>> series.cleaner.detect.url(...)
    >>> series.cleaner.detect('url',...)

    This detection method flags values as potential errors wherever the
    corresponding Series element does not match a url.
    URLs can be a regular internet address, or an IP, or localhost

    Parameters
    ----------

    check_protocol: bool (Default = True)
        If True, the 'http/https' is mandatory in a regular url.

    Note
    ----
    Missing values (NaN) are not treated as errors

    Examples
    --------

    >>> my_series = pd.Series([
        'google.com','https://www.google.com/', 'https://127.0.0.1:80', 'dummy'])
    >>> my_detector = my_series.cleaner.detect.url()
    >>> print(my_detector.detected)
    0    google.com
    3         dummy
    dtype: object

    If protocol is not mandatory

    >>> my_series = pd.Series(['google.com','https://www.google.com/'])
    >>> my_detector = my_series.cleaner.detect('url', check_protocol=False)
    >>> print(my_detector.is_error())
    0   False
    1   False
    dtype: bool
    """

    name = 'url'

    def __init__(self, obj, detector_obj=None, check_protocol=True):

        if detector_obj is None:
            self._check_protocol = check_protocol
        else:
            self._check_protocol = detector_obj.check_protocol

        pattern_url = re.compile(
            (r'^(?:http|ftp)s?://' if check_protocol else
                r'(^(?:http|ftp)s?://)?') +  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'
            r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        super().__init__(obj, detector_obj, pattern=pattern_url, mode="fullmatch")

    @property
    def check_protocol(self):
        r"""If True, checks if the http or https protocol is present.
        Otherwise, the protocol is optional"""
        return self._check_protocol

    @property
    def _reported(self):
        r"""Generates a report of the detection"""
        return ['check_protocol']


class PingSeriesDetector(_ObjectTypeSeriesDetector):
    r"""'ping': Detect strings that do not match a reachable url.

    Intended to be used by the detect method with the keyword 'ping'

    >>> series.cleaner.detect.ping(...)
    >>> series.cleaner.detect('ping',...)

    This detection method flags values as potential errors wherever the
    corresponding Series element does not match a reachable url.

    Note
    ----
    Missing values (NaN) are not treated as errors

    Examples
    --------

    >>> my_series = pd.Series(['google.com','https://www.google.com/', 'dummy'])
    >>> my_detector = my_series.cleaner.detect.ping()
    >>> print(my_detector.detected)
    0    google.com
    2         dummy
    dtype: object
    """

    name = 'ping'

    def __init__(self, obj, detector_obj=None):
        # pylint: disable=unused-argument
        super().__init__(obj)

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        def ping(url):
            try:
                requests.get(url)
                return True
            except requests.exceptions.RequestException:
                return False

        mask = ~(self._obj.apply(ping))

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def _reported(self):
        r"""Generates a report of the detection"""
        return []

"""
Strings detectors
"""
import re
import warnings

import pandas as pd

from pdcleaner.detection._base import _ObjectTypeSeriesDetector
from pdcleaner.utils.utils import raise_if_not_in


class pattern(_ObjectTypeSeriesDetector):
    r"""Detect strings that do not match a given pattern.

    This detection method flags values as potential errors wherever the
    corresponding Series element does not match a given character sequence
    or regular expression.

    Matching methods 'match', 'fullmatch' or 'contains' (similar to python's re.search)
    can be used.

    Parameters
    ----------

    pattern: string
        Character sequence or regular expression.

    mode: string (Default = 'match')
        test wether:

        - 'match': there is a match that begins at the first character of the string

        - 'fullmatch': the entire string matches the regular expression

        - 'contains':  there is a match at any position within the string

    case: bool (Default = True)
        If True, the search is case sensitive.

    flags: int (Default = 0 = no flags)
        Regex module flags, e.g. re.IGNORECASE.

    Raises
    ------
    ValueError
        when pattern is empty
        when mode is neither 'match', 'fullmatch' nor 'contains'

    Note
    ----
    Missing values (NaN) are not treated as errors

    Examples
    --------

    Strings are to be not lower cases letters only

    >>> series = pd.Series(['Cat','cat','dog','bird','14',np.nan,""])
    >>> detector = series.cleaner.detect.pattern(pattern=r"[a-z]*", mode='fullmatch')
    >>> print(detector.detected())
    0    Cat
    4     14
    dtype: object

    Strings must contain a "d"

    >>> series = pd.Series(['Cat','cat','dog','bird','14',np.nan,""])
    >>> detector = series.cleaner.detect.values(pattern=r"d", mode='contains')
    >>> print(detector.detected())
    0    Cat
    1    cat
    4     14
    6
    dtype: object

    Strings should be 'cat' or 'dog' whenever the case

    >>> series = pd.Series(['Cat','cat','dog','bird','14',np.nan,""])
    >>> detector = series.cleaner.detect.values(pattern=r"cat|dog", mode='match', case=False)
    >>> print(detector.detected())
    3    bird
    4      14
    6
    dtype: object

    One can also use a compiled regex. In this case, the arguments `case` and `flag`
    are ignored

    >>> series = pd.Series(['Cat','cat','dog','bird','14',np.nan,""])
    >>> import re
    >>> regex = re.compile(r"[a-z]*")
    >>> detector = series.cleaner.detect.pattern(pattern=regex, mode='fullmatch', case=True)
    ... UserWarning: case and flag are ignored with a compiled regex
    >>> print(detector.detected())
    0    Cat
    4     14
    dtype: object

    """
    name = 'pattern'

    def __init__(self, obj,
                 detector=None,
                 pattern="",
                 mode="match",
                 case=True,
                 flags=0
                 ):
        super().__init__(obj)

        if not detector:
            self._pattern = pattern
            self._mode = mode
            self._case = case
            self._flags = flags
        else:
            self._pattern = detector.pattern
            self._mode = detector.mode
            self._case = detector.case
            self._flags = detector.flags

        if self._pattern == "":
            raise ValueError("The pattern is empty")

        if isinstance(self.pattern, re.Pattern):
            warnings.warn("case and flag are ignored with a compiled regex")

        modes = ['match', 'fullmatch', 'contains']

        if self._mode not in modes:
            raise ValueError(f"mode shoud be one of {modes}")

    @property
    def pattern(self):
        """Character sequence or regular expression used to detect errors"""
        return self._pattern

    @property
    def mode(self):
        """'match', 'fullmatch' or 'contains'"""
        return self._mode

    @property
    def case(self):
        """Case sensitivity"""
        return self._case

    @property
    def flags(self):
        """Usage of Regex module flags"""
        return self._flags

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        if isinstance(self.pattern, re.Pattern):
            kwargs = {'pat': self.pattern}
        else:
            kwargs = {
                'pat': self.pattern,
                'case': self.case,
                'flags': self.flags,
            }

        if self.mode == 'match':
            mask = ~self._obj.fillna('').str.match(**kwargs)
        elif self.mode == 'fullmatch':
            mask = ~self._obj.fillna('').str.fullmatch(**kwargs)
        elif self.mode == 'contains':
            mask = ~self._obj.fillna('').str.contains(**kwargs)

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        if isinstance(self.pattern, re.Pattern):
            return ['pattern', 'mode']
        return ['pattern', 'mode', 'case', 'flags']


class keycollision(_ObjectTypeSeriesDetector):
    r"""Detect strings that might be alternative representations of the same thing.

    This detection method is typically useful for people names or for brands.
    It works on a clustering approach by grouping strings that might be alternative
    representations of the same thing, but written a little bit diferently, e.g:
    'Linus Torvalds', 'linus.torvald', 'Torvalds, Linus', 'linus torvald'

    As explained in [1]: \"Key Collision methods are based on the idea of creating an alternative
    representation of a value (a "key") that contains only the most valuable or meaningful
    part of the string and "buckets" together different strings based on the fact that their
    key is the same (hence the name "key collision").

    As for now, the only available method is the fingerprinting method, explained in [1].
    In the aforementioned example, all strings would have the same key: 'linus torvalds'

    The detector flags values with a given key but not the most frequent formulation as errors.

    A dictionary associating the keys and the most frequent associated values is produced.
    {'linus torvalds': 'Linus Torvalds', ...}

    Reference
    ---------
    [1] https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth

    Parameters
    ----------
    keys: str (Default = 'fingerprint')
        method for generating the keys. Only 'fingerprint' currently available

    Raises
    ------
    ValueError
        when keys is not 'fingerprint'

    Note
    ----
    NA values are not treated as errors.

    Examples
    --------

    >>> series = pd.Series(['Linus Torvalds','linus.torvalds','Torvalds, Linus',
                               'Linus Torvalds', 'Bill Gates', ])
    >>> detector = series.cleaner.detect.keycollision()
    >>> print(detector.is_error())
    0    False
    1    True
    2    True
    3    False
    4    False
    dtype: bool
    >>> print(detector.dict_keys)
    {'linus torvalds': 'Linus Torvalds', 'bill gates': 'Bill Gates'}

    Missing values are not treated as errors.

    >>> series = pd.Series(['Linus Torvalds','linus.torvalds','Torvalds, Linus', np.nan ])
    >>> detector = series.cleaner.detect.keycollision()
    >>> print(detector.is_error())
    0    False
    1    True
    2    False
    3    False
    dtype: bool
    """
    name = 'keycollision'

    def __init__(self, obj, detector=None, keys='fingerprint'):
        super().__init__(obj)

        if not isinstance(keys, str):
            raise TypeError('keys must be a string')

        if keys not in ['fingerprint']:
            raise ValueError('Not a valid method. Only fingerprint method is implemented')

        if not detector:
            self._keys = keys
            self._dict_keys = self.dict_keys
        else:
            self._keys = detector.keys
            self._dict_keys = detector.dict_keys

    @property
    def keys(self):
        """returns the name of the method used to generate keys"""
        return self._keys

    @property
    def dict_keys(self) -> dict:
        """  A python dictionary associating the key with the most frequent formulation

        can be used for replacements"""

        keys = self.calc_keys(method=self._keys)

        df = pd.DataFrame({'orig': self._obj.astype(str), 'keys': keys})
        group = df.groupby('keys').agg(lambda s: s.value_counts().idxmax())

        return pd.Series(group.orig.values, index=group.index).to_dict()

    @staticmethod
    def fingerprints(series_: pd.Series) -> pd.Series:
        """ Calculate fingerprint key for each element of the series"""
        return (series_.fillna('')
                .str.replace('^.{2}:', ' ', regex=True)
                .str.strip()
                .str.lower()
                .str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode("utf-8")
                .str.replace('[^a-zA-Z]', ' ', regex=True)
                .str.replace(' +', ' ', regex=True)
                .str.split(" ")
                .apply(lambda l: sorted(list(dict.fromkeys(l))))
                .apply(lambda l: ' '.join(l))
                .str.strip()
                )

    def calc_keys(self, method='fingerprint') -> pd.Series:
        """Calculate keys with the given method"""
        if method == 'fingerprint':
            return self.fingerprints(self._obj)

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""

        # For each key, associate the most frequent corresponding original value
        associated_mod = self.calc_keys(self._keys).map(self._dict_keys)

        # Flag as errors when the actual value is not the most frequent for the key
        mask = self._obj.ne(other=associated_mod)

        mask[self._obj.isna()] = False  # NA are not errors

        return self._obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ['keys']


class spaces(_ObjectTypeSeriesDetector):
    r"""Detect elements whith extra spaces before and/or after the value.

    Intended to be used by the detect method with the keyword 'spaces'

    >>> series.cleaner.detect.spaces(...)
    >>> series.cleaner.detect('spaces',...)

    This detection method flags elements as potential errors wherever they
    contain an extra space at begininng or at the end.

    Parameters
    ----------
    side: {'leading', 'trailing', 'both'} (Default = 'both')
        The side where extraspaces will be detected

    Raises
    ------
    ValueError
        when unknown value is given to side parameter

    Note
    ----
    NA values are not treated as errors.

    Examples
    --------
    >>> series = pd.Series(['Paris','Paris ',' Lille', ' Lille ', 'Troyes'])
    >>> detector = series.cleaner.detect.spaces(side='leading')
    >>> print(detector.is_error())
    0    False
    1    False
    2     True
    3     True
    4    False
    dtype: bool

    >>> detector = series.cleaner.detect.spaces(side='trailing')
    >>> print(detector.is_error())
    0    False
    1     True
    2    False
    3     True
    4    False
    dtype: bool

    >>> detector = series.cleaner.detect.spaces(side='both')
    >>> print(detector.is_error())
    0    False
    1     True
    2     True
    3     True
    4    False
    dtype: bool
    """
    name = 'spaces'

    def __init__(self,
                 obj,
                 detector=None,
                 side='both'
                 ):
        super().__init__(obj)

        legal_values = ["leading", "trailing", "both"]
        raise_if_not_in(side, legal_values, f"Parameter side must be {' or '.join(legal_values)}")

        if not detector:
            self._side = side
        else:
            self._side = detector.side

    @property
    def side(self) -> str:
        """Side to check the presence of spaces"""
        return self._side

    @property
    def index(self) -> pd.Index:
        """Indices of the rows detected as errors"""
        if self.side == "leading":
            mask = self._obj.apply(lambda x: x.startswith(" "))
        elif self.side == "trailing":
            mask = self._obj.apply(lambda x: x.endswith(" "))
        elif self.side == "both":
            mask = self._obj.apply(lambda x: (x.startswith(" ") or x.endswith(" ")))

        mask[self._obj.isna()] = False

        return self.obj[mask].index

    @property
    def _reported(self):
        """Properties displayed by the report() method"""
        return ["side"]

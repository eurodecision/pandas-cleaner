![](https://github.com/eurodecision/pandas-cleaner/blob/master/docs/source/pandas-cleaner.png)

------

**Pandas-cleaner**

<img src=https://img.shields.io/pypi/v/pandas-cleaner.svg target=https://pypi.python.org/pypi/pandas-cleaner>
<img src=https://img.shields.io/pypi/l/pandas-cleaner.svg target=https://pypi.python.org/pypi/pandas-cleaner>
<img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ed-ialifinaritra/74e11571ef9b1a732e44fe9fbbdc7f0b/raw/pdcleaner_coverage.json">

Pandas-cleaner is a Python package, built on top of pandas, that provides methods detect, analyze and clean errors in datasets with different types of data (numerical, categorical, text, datetimes...).


## Features

Pandas-cleaner offers functionnalities to automatically :

:arrow_right: **detect** different kind of potential errors in datasets such as outliers, inconsistencies, typos, wrong-typed ..., given predefined rules or statistiscal estimations,  via an easy-to-use API extending pandas,

:arrow_right: **analyze** these errors, via reports and plots, to check the validity of the set and/or decide if any correction is needed,

:arrow_right: **clean** the datasets, either by dropping the lines with errors, emptying, correcting or replacing bad values,

:arrow_right: **reapply** the same rules to any other incoming fresh data.

## Usage

Import the package

```python
import pandas as pd
import pdcleaner
```

Create an example data series

```python
series = pd.Series([1, 5, -6, 100, 10])
```

Detect the errors in the series with a given method (such as `bounded`, `iqr`, `zscore` and many more depending the type of data...)

```python
detector = series.cleaner.detect('bounded', lower=0, upper=10)
```

Inspect the result:

```python
detector.report()
```

```none

                                 Detection report                               
==============================================================================
Method:                      bounded      Nb samples:                        5
Date:                January 24,2022      Nb errors:                         2
Time:                       16:06:08      Nb rows with NaN:                  0
------------------------------------------------------------------------------
lower                              0      upper                             10
inclusive                       both      sided                           both
==============================================================================
```

Check the potential errors that have been detected

```python
detector.detected()
```
```
 2     -6
 3    100
 dtype: int64
```

Clean the detected errors from the series using the chosen method among `drop`, `to_na`, `clip`
, `replace`...

```python
series.cleaner.clean("drop", detector, inplace=True)
   series
```
```
 0      1
 1      5
 4     10
 dtype: int64
```

## Contributing to pandas-cleaner

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

Issues and bugs can be reported at https://github.com/eurodecision/pandas-cleaner/issues

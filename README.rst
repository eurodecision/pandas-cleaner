
.. image:: https://edgitlab.eurodecision.com/data/pandas-cleaner/-/raw/dev/docs/source/pandas-cleaner.png

------

pandas-cleaner
==============


What is it ?
------------

Pandas-cleaner is a Python package, built on top of pandas, that provides methods detect, analyze and clean errors in datasets with different types of data (numerical, categorical, text, datetimes...).


Features
--------
Pandas-cleaner offers functionnalities to automatically :

+ **detect** different kind of potential errors in datasets such as outliers, inconsistencies, typos, wrong-typed ..., given predefined rules or statistiscal estimations,  via an easy-to-use API extending pandas,

+ **analyze** these errors, via reports and plots, to check the validity of the set and/or decide if any correction is needed,

+ **clean** the datasets, either by dropping the lines with errors, emptying, correcting or replacing bad values,

+ **reapply** the same rules to any other incoming fresh data.

Installation
------------

Pandas-cleaner is at present time an internal EURODECISION python package. To download and install it:

* Go to https://edgitlab.eurodecision.com/eurodecision/releases/-/packages
* Select the most recent version (or any older according to your needs)
* Download `pdcleaner-<version-number>-py3-none-any.whl`
* Install using pip

.. code-block:: shell

  pip install pdcleaner-<version-number>-py3-none-any.whl

Usage
-----

Import the package

.. code-block:: python

   import pandas as pd
   import pdcleaner

Create an example data series

.. code-block:: python

   series = pd.Series([1, 5, -6, 100, 10])

Detect the errors in the series with a given method (such as `bounded`, `iqr`, `zscore` and many more depending the type of data...)

.. code-block:: python

   detector = series.cleaner.detect('bounded', lower=0, upper=10)

Inspect the result:

.. code-block:: none

   detector.report()

.. code-block:: none

                                    Detection report                               
   ==============================================================================
   Method:                      bounded      Nb samples:                        5
   Date:                January 24,2022      Nb errors:                         2
   Time:                       16:06:08      Nb rows with NaN:                  0
   ------------------------------------------------------------------------------
   lower                              0      upper                             10
   inclusive                       both      sided                           both
   ==============================================================================

Check the potential errors that have been detected

.. code-block:: python

   detector.detected

.. parsed-literal::

    2     -6
    3    100
    dtype: int64

Clean the detected errors from the series using the chosen method among `drop`, `to_na`, `clip`
, `replace`...

.. code-block:: python

   series.cleaner.clean("drop", detector, inplace=True)
   series

.. parsed-literal::

    0      1
    1      5
    4     10
    dtype: int64

Documentation
-------------

The documentation is still a **work in progress**. 

* Clone the project

* Build the documentation using :

.. code-block:: bash

    cd docs
    make html

* Open `docs/build/html/index.html` in your browser

Contributing to pandas-cleaner
------------------------------

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

Issues and bugs can be reported at https://edgitlab.eurodecision.com/data/pandas-cleaner/-/issues

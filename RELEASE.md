### What's new in 0.0.3 (October 17, 2022)

+ **New detection methods**
    + `'value'` to detect cells different from a given value
    + ``'custom'`` to apply an user-defined python callable to detect the errors
    + ``'spaces'`` to detect extra spaces before or after strings
    + ``'duplicated'`` to detect duplicated records (whole row or subset of columns)
    + ``'date_range'`` to detect dates outside a given time range
+ **New cleaning method**
    + ``'cast'`` associated to the detector ``castable`` to cast strings into floats, integers, dates or NaNs
+ **Enhancements**
    + The detectors have two new useful methods `.detected()` and `.valid()` that return valid/unvalid rows
    + Gaussian detectors (``iqr``, ``zscore``, ``modzscore``) now have a systematic normality test and an option to transform the distribution
      (Box-Cox, Yeo-Johnson) before applying the detector
    + Reports are more detailed when using a multivariate qualitative/ quantitative detector
    + Detector ``castable`` now applies to booleans
    + Detector ``values`` is now called ``enum``
    + Detector ``keycollision`` 
        + is now called ``alternatives`` 
        + is much faster
        + now offers the choice to keep either most frequent representation or the first encountered
    + Plot method for ``freq`` and ``cat`` detectors now have options ``nfirst`` and ``nlast``
    + API for detector ``length`` has been simplified
+ **Bug fixes**
    + Detector ``pattern`` now works with compiled regex
    + Cleaning methods applied to DataFrames have been fixed
    + Booleans are better displayed in reports
    + Fixed bug with ``association`` detector plot method
+  **Code**
    + Detector classes now have much simplier names
    + ``tests`` folder has been reorganized  to mimic ``src`` structure

### What's new in 0.0.1 (February 25, 2022)

+ **New detection methods**
    + `'length'` to detect string or number cells that do not match a desired number of characters
    + `'missing'` to detect missing values in Series or DataFrames (partially or totally blank lines)
    + 3 new methods related to web formatted strings
        * `'email'` to check email format
        * `'url'` for url format
        * `'ping'` to check if urls are reachable
    + `'castable'` to check wether string have a good format to be casted as integers, floats or dates
+ **Enhancements**
    + The detectors now have an updated method `.report()` to display a formatted detection report
+ **Other**
    + README has been updated
    + Release notes added
+ **Bug fixes**
    + Key collision detector is now much faster


## Version 0.1

### What's new in 0.1.2 (February 25, 2022)

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


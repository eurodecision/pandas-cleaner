import numpy as np
import pandas as pd
import seaborn as sns

from pdcleaner.utils.utils import add_method
from pdcleaner.detection.values \
    import CategoriesAssociationsDataFramesDetector


@add_method(CategoriesAssociationsDataFramesDetector, 'plot')
def plot(self, color='green', errors_color='red', fmt=".0f"):
    """plot a colored matrix (heatmap) représentation of categories associations.

    Parameters
    ----------
    color : palette name (Default: "green")
        Color associated to legitimate associations.
        Should be something that can be interpreted by seaborn's color_palette()
    errors_color : palette name (Default: "red")
        Color associated to erroneous associations.
        Should be something that can be interpreted by seaborn's color_palette()
    fmt  : format (default : ".0f")
        String formatting code to use for the numbers.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.

    Example
    -------
    >>> import pandas as pd
    >>> import pdcleaner

    >>> df = pd.DataFrame({
                'col1': ['A'] * 10 + ['B'] * 10,
                'col2': ['a'] * 8 + ['c'] * 2 + ['b'] * 9 + ['a'],
        })
    >>> detector = df.cleaner.detect.associations(count=3)
    >>> detector.plot()

    .. image:: ../../_static/plot_association.png
    """
    data = self.obj

    crosstab = pd.crosstab(index=data.iloc[:, 0],
                           columns=[data.iloc[:, 1]]
                           )

    ax = sns.heatmap(crosstab.replace(0, np.nan),
                     cmap=[color],
                     vmin=0,
                     annot=True,
                     linewidths=5,
                     cbar=False,
                     fmt=fmt,
                     )

    data['error'] = ~data.apply(tuple, axis=1).isin(self.valid_associations)

    pivot_tbl = pd.pivot_table(data=data,
                               index=data.columns[0],
                               columns=[data.columns[1]],
                               values='error',
                               aggfunc=np.sum
                               )

    sns.heatmap(pivot_tbl.replace(0, np.nan),
                cmap=[errors_color],
                vmin=0,
                annot=True,
                linewidths=5,
                cbar=False,
                fmt=fmt,
                ax=ax
                )

    return ax

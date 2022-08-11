"""Plot method for values count and freq detectors"""
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pdcleaner.utils.utils import add_method
from pdcleaner.detection.values import counts, freq


@add_method(counts, 'plot')
@add_method(freq, 'plot')
def plot(self,
         nfirst=0,
         nlast=0,
         figsize=None,
         color='green',
         errors_color='red',
         not_displayed_color='grey',
         ):
    """plot a countplot of values frequency, with options to compact the graph

    Parameters
    ----------

    nfirst : int
        Number of top n values to display
    nlast : Bool (Default: True)
        Number of n last values to display
    figsize :  (float, float) (Default: None)
        width and height of the figure.
    color : palette name (Default: "green")
        Color associated to legitimate values.
        Should be something that can be interpreted by seaborn's color_palette()
    errors_color : palette name (Default: "red")
        Color associated to erroneous values.
        Should be something that can be interpreted by seaborn's color_palette()
    not_displayed_color : str, color name (Default = "grey")
        Box color for the number of hidden values

    Returns
    -------
    axs : matplotlib.axes._subplots.AxesSubplot
        matplotlib axes objects representing the plots

    Raises
    ------
        ValueError if nfirst or nlast is <0
        TypeError if nfirst or nlast is not an integer

    Examples
    --------
    >>> my_series = pd.Series(['cat','cat','dog', 'dog','dog','bird'])
    >>> detector = my_series.cleaner.detect.freq(freq=.2)
    >>> detector.plot()

    .. image:: ../../_static/plot_freq.png

    >>> detector.plot(nfirst=1, nlast=1)

    .. image:: ../../_static/plot_freq_nfirst_nlast.png

    """

    if not isinstance(nfirst, int):
        raise TypeError('nfirst should be an integer')

    if not isinstance(nlast, int):
        raise TypeError('nlast should be an integer')

    if nfirst < 0:
        raise ValueError('nfirst should be >=0')

    if nlast < 0:
        raise ValueError('nlast should be >=0')

    vals = self.obj.dropna().value_counts()

    not_displayed = len(vals) - nfirst - nlast

    if not_displayed != len(vals):
        vals_chunks = [vals.iloc[:nfirst]]
        if not_displayed > 0:
            vals_chunks.append(pd.Series([0], index=[f"{not_displayed}"]))
        if nlast > 0:
            vals_chunks.append(vals.iloc[-nlast:])
        compacted = pd.concat(vals_chunks)
    else:
        compacted = vals

    palette = [color if val in self.values else errors_color for val in compacted.index]

    _, ax = plt.subplots(figsize=figsize)
    sns.barplot(y=compacted.index, x=compacted, palette=palette, ax=ax)
    plt.ylabel('')

    if str(not_displayed) in compacted.index:
        pos = compacted.reset_index()[compacted.index == str(not_displayed)].index.values.item()
        ax.text(0,
                float(pos),
                f"    +{not_displayed}    ",
                color='white',
                weight='bold',
                ha='center',
                bbox=dict(facecolor=not_displayed_color,
                          edgecolor=not_displayed_color,),
                )

    return ax

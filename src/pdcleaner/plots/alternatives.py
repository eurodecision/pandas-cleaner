import pandas as pd

from pdcleaner.utils.utils import add_method
from pdcleaner.detection.strings import alternatives


@add_method(alternatives, 'plot')
def plot(self, cmap=None, not_displayed_color='red', nfirst=0, nlast=0, figsize=None):
    """plot a countplot of values frequency grouped by keys, with options to compact the graph

    Parameters
    ----------
    cmap : palette name (Default = Default matplotlib's palette)
        Should be something that can be interpreted by seaborn's color_palette()
    not_displayed_color : str, color name (Default = "red")
        Box color for the number of hidden values
    nfirst : int
        Number of top n values to display
    nlast : Bool (Default: True)
        Number of n last values to display
    figsize :  (float, float) (Default: None)
        width and height of the figure.

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

    >>> series = pd.Series(['Linus Torvalds',
                               'Torvalds, Linus',
                               'Linus Torvalds',
                               'Bill Gates',
                               'Bill Gates',
                               'Steve Jobs',
                               ])
    >>> detector = series.cleaner.detect.alternatives()
    >>> detector.plot()

    .. image:: ../../_static/plot_alternatives_1.png

    Display only the two most frequents

    >>> detector.plot(nfirst=2)

    .. image:: ../../_static/plot_alternatives_nfirst_2.png

    Display only the least frequent

    >>> detector.plot(nlast=1)

    .. image:: ../../_static/plot_alternatives_nlast_1.png
    """

    if not isinstance(nfirst, int):
        raise TypeError('nfirst should be an integer')

    if not isinstance(nlast, int):
        raise TypeError('nlast should be an integer')

    if nfirst < 0:
        raise ValueError('nfirst should be >=0')

    if nlast < 0:
        raise ValueError('nlast should be >=0')

    keys = self.fingerprints(self.obj)

    df = pd.DataFrame({'series': self.obj,
                       'keys': keys,
                       'value':  keys.map(self.dict_keys),
                       })

    pivot = (df.pivot_table(index='value', columns='series', aggfunc='count').fillna(0))

    not_displayed = len(pivot) - nfirst - nlast

    if (not_displayed != len(pivot)) and (not_displayed > 0):
        if nfirst == 0:
            nfirst = -len(pivot)

        compacted = pd.concat([
            pivot.loc[pivot.sum(axis=1).sort_values().index].iloc[0:nlast],
            pd.DataFrame(columns=pivot.columns, index=[not_displayed]).fillna(0),
            pivot.loc[pivot.sum(axis=1).sort_values().index].iloc[-nfirst:],
        ])

        ax = compacted.plot(kind='barh',
                            stacked=True,
                            legend=False,
                            cmap=cmap,
                            figsize=figsize
                            )

        pos = compacted.reset_index()[compacted.index == not_displayed].index.values.item()

        ax.text(0,
                pos,
                f"    +{not_displayed}    ",
                color='white',
                weight='bold',
                ha='center',
                bbox=dict(facecolor=not_displayed_color,
                          edgecolor=not_displayed_color,),
                )
    else:
        ax = (pivot.loc[pivot.sum(axis=1).sort_values().index]
                   .plot(kind='barh',
                         stacked=True,
                         legend=False,
                         cmap=cmap,
                         figsize=figsize
                         )
              )

    ax.set_ylabel('')

    return ax

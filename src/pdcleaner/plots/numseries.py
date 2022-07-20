import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pdcleaner.utils.utils import add_method
from pdcleaner.detection._base import _NumericalSeriesDetector


@add_method(_NumericalSeriesDetector, 'plot')
def plot(self,
         color='green',
         errors_color='red',
         compact=False,
         limits=True,
         figsize=None
         ):
    """plot a visualization representing an overview of the treated data and colored
    according to the validity of the values:

    - a scatter plot representing the values in the treated series.
    - a histogram representing the distribution of values.
    - a kernel density estimate plot visualizing the distribution of values.
    - a boxplot showing the distribution of values.

    Parameters
    ----------
    color : palette name (Default: "green")
        Color associated to legitimate values.
        Should be something that can be interpreted by seaborn's color_palette()
    errors_color : palette name (Default: "red")
        Color associated to erroneous values.
        Should be something that can be interpreted by seaborn's color_palette()
    compact : Bool (Default: False)
        If True, compact the plots around valid values and show the number of erroneous values
        on the scatter plot
    limits : Bool (Default: True)
        If True, draw horizontal lines showing the lower and upper values delimiting
        the allowed values
    figsize :  (float, float) (Default: None)
        width and height of the figure.


    Returns
    -------
    axs : array of matplotlib.axes._subplots.AxesSubplot
        an array of length 4 containing the matplotlib axes representing the plots

    Examples
    --------

    >>> my_series = pd.Series([-5, 1, 2 , 3, 8, 12])
    >>> my_detector = my_series.cleaner.detect.bounded(lower=0, upper=10)
    >>> my_detector.plot()

    .. image:: ../../_static/plot_numseries.png
    """
    df = pd.DataFrame({'data': self.obj, 'error': self.is_error()})

    _, axs = plt.subplots(1, 4,
                          sharey=True,
                          gridspec_kw={"width_ratios": (.7, .1, .1, .1)},
                          figsize=figsize,
                          )

    if self.is_error().all():
        palette = [errors_color]
    elif self.not_error().all():
        palette = [color]
    else:
        palette = [color, errors_color]

    linestyle = ':'

    sns.scatterplot(data=df,
                    x=df.index,
                    y='data',
                    hue='error',
                    palette=palette,
                    legend=False,
                    ax=axs[0],
                    )

    sns.histplot(data=df,
                 y='data',
                 ax=axs[1],
                 hue='error',
                 legend=False,
                 palette=palette,
                 )

    sns.kdeplot(data=df,
                y='data',
                color=color,
                ax=axs[2],
                fill=True,
                clip=(self.lower, self.upper)
                )
    if self.lower != np.NINF:
        sns.kdeplot(data=df,
                    y='data',
                    color=errors_color,
                    ax=axs[2],
                    fill=True,
                    clip=(None, self.lower)
                    )
    if self.upper != np.inf:
        sns.kdeplot(data=df,
                    y='data',
                    color=errors_color,
                    ax=axs[2],
                    fill=True,
                    clip=(self.upper, None)
                    )

    sns.boxplot(data=df,
                y='data',
                palette=palette,
                ax=axs[3],
                flierprops=dict(markerfacecolor=errors_color, markeredgecolor=errors_color),
                showfliers=False,
                )
    sns.stripplot(data=df[self.is_error()],
                  y='data',
                  color=errors_color,
                  ax=axs[3],
                  )

    # Get left axis position to position lower and upper labels
    xmin = axs[0].get_xlim()[0]

    # Compact graphic around valid values and show the number of potential errors
    if compact:
        extension = 0.5 * (self.obj[self.not_error()].max() - self.obj[self.not_error()].min())
        if np.isnan(extension):
            extension = 0.

        if not np.isinf(self.lower):

            # Compact the graph
            axs[0].set_ylim([self.lower - extension, axs[0].get_ylim()[1]])
            ymin = axs[0].get_ylim()[0]

            axs[0].text(0, max(ymin, self.obj.min()),
                        f"min: {float(self.obj.min()):.3}",
                        color=errors_color,
                        va='bottom',
                        ha='left',
                        bbox=dict(facecolor='white',
                                  edgecolor=errors_color,
                                  ),
                        )

            for ax_i in axs:
                ymin = ax_i.get_ylim()[0]
                if ymin > self.obj.min():
                    ax_i.spines['bottom'].set_visible(False)
                    ax_i.axhline(ymin, linestyle='--', color='black')

            axs[0].text(len(self.obj)/2.,
                        ymin+0.5 * extension,
                        f"{len(self.obj[self.obj < self.lower])}",
                        color='white',
                        weight='bold',
                        bbox=dict(facecolor=errors_color,
                                  edgecolor=errors_color,
                                  boxstyle='circle,pad=0.5'
                                  ),
                        )

        if not np.isinf(self.upper):
            # Compact the graph
            axs[0].set_ylim([axs[0].get_ylim()[0], self.upper + extension, ])
            ymax = axs[0].get_ylim()[1]

            axs[0].text(0, min(ymax, self.obj.max()),
                        f"max: {float(self.obj.max()):.3}",
                        color=errors_color,
                        va='top',
                        ha='left',
                        bbox=dict(facecolor='white',
                                  edgecolor=errors_color),
                        )

            axs[0].text(len(self.obj)/2., ymax-0.5 * extension,
                        f"{len(self.obj[self.obj > self.upper])}",
                        color='white',
                        weight='bold',
                        bbox=dict(facecolor=errors_color,
                                  edgecolor=errors_color,
                                  boxstyle='circle,pad=0.5'
                                  ),
                        )

            for ax_i in axs:
                ymax = ax_i.get_ylim()[1]
                if ymax < self.obj.max():
                    ax_i.spines['top'].set_visible(False)
                    ax_i.axhline(ymax, linestyle='--', color='black')

    if limits:

        if not np.isinf(self.lower):

            for ax_i in axs:
                ax_i.axhline(self.lower, c=errors_color, ls=linestyle)

            axs[0].text(xmin,
                        self.lower,
                        f" {float(self.lower):.3}",
                        ha='right',
                        color=errors_color,
                        bbox=dict(facecolor='white',
                                  edgecolor=errors_color,
                                  )
                        )

        if not np.isinf(self.upper):

            for ax_i in axs:
                ax_i.axhline(self.upper, c=errors_color, ls=linestyle)

            axs[0].text(xmin,
                        self.upper,
                        f" {float(self.upper):.3}",
                        color=errors_color,
                        ha='right',
                        bbox=dict(facecolor='white',
                                  edgecolor=errors_color,
                                  )
                        )

    return axs

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from torch import Tensor


def plot_vtrace(
    t: Tensor,
    Vt: Tensor,
    It: Optional[Tensor] = None,
    axes: Axes = None,
    figsize: Tuple[int, int] = (12, 4),
    title: str = "",
    timewindow: Tuple[int, int] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """Plot voltage and possibly current trace.

    Args:
        axes: The set of axes this plot will be added to.
        figsize: Changes the figure size of the plot.
        title: Adds a custom title to the figure.
        timewindow: The voltage and current trace will only be plotted
            between (t1,t2). To be specified in secs.

    Returns:
        axes: The set of axes of this plot in order to add further plots to
            the same set of axes."""

    for i in torch.arange(t.shape[0]):
        start, end = (0, -1)
        dt = t[i, 0] - t[i, 1]
        if timewindow != None:
            start, end = torch.tensor(timewindow) / dt
            start = int(start)
            end = int(end)

        ax1, ax2 = (0, 0)
        time = t[i, start:end].numpy()
        voltage = Vt[i, start:end].numpy()

        if It == None:
            if axes == None:
                fig = plt.figure(figsize=figsize)
                axes = plt.subplot(111)

            axes.plot(time, voltage, **kwargs)
            axes.set_xlabel("t [ms]", fontsize=14)
            axes.set_ylabel("V [mV]", fontsize=14)
            if "label" in kwargs:
                axes.legend(loc=1)

        else:

            current = It[i, start:end].numpy()

            if axes == None:
                fig = plt.figure(1, figsize=figsize)
                gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1], figure=fig)
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])
                axes = [ax1, ax2]
            if axes != None:
                ax1, ax2 = axes

            ax1.plot(time, voltage, **kwargs)
            ax1.set_ylabel("V [mV]")
            ax1.set_title(title)
            if "label" in kwargs:
                ax1.legend(loc=1)
            # plt.setp(ax1, xticks=[], yticks=[-80, -20, 40])

            ax2.plot(time, current, lw=2, c="black", **kwargs)
            ax2.set_xlabel("t [ms]")
            ax2.set_ylabel("I [nA]")

            ax2.set_yticks([0, torch.max(current)])
            ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))

    return fig, axes

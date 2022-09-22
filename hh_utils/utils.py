import re
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from scipy.io import loadmat
from torch import Tensor


# return sigmoid of input x
def sigmoid(x: Tensor, offset: int = 1, steepness: int = 1) -> Tensor:
    """Implements the sigmoid function.

    Args:
        x: Where to evaluate the sigmoid.
        offset: translation of x.
        steepness: time constant of the exponential.

    Returns:
        Sigmoid evaluated at x.
    """
    # offset to shift the sigmoid centre to 1
    return 1 / (1 + torch.exp(-steepness * (x - offset)))


def constant_stimulus(
    duration: float or int,
    dt: float,
    stim_onset: float or int,
    stim_end: float or int,
    magn: float or int,
    noise: float = 0.0,
    return_ts: bool = False,
) -> Tuple[Tensor, Tensor] or Tensor:
    """Creates a constant stimulus current.

    Based on the input parameters, a stimulus current is generated, that can be fed
    to a HH simulator in order to compute the stimulus response of a HH neuron.

    Args:
        duration: Duration of the whole current pulse in ms.
        dt: Time steps in ms, 1/dt = sampling frequency in MHz.
        stim_onset: At which timepoint [ms] the stimulus is applied, i.e. I != 0.
        stim_end: At which timepoint [ms] the stimulus is stopped, i.e. I = 0 again.
        magn: The magnitude of the constant stimulus in pA.
        noise: Noise added ontop of the input current. Measured in mV
        return_ts: Whether to also return the time axis along with the current.

    Returns:
        t: Corresponding time axis of the stimulus [ms].
        I_t: Stimulus current [pA].
    """
    t_start = 0.0
    t_end = duration

    t = torch.arange(t_start, t_end, dt)
    I_t = torch.zeros_like(t)
    stim_at = torch.logical_and(t > stim_onset, t < stim_end)
    I_t[stim_at] = magn
    I_t += noise * torch.randn(len(I_t)) / (dt**0.5)

    if return_ts:
        return t, I_t
    else:
        return I_t


def ints_from_str(string: str) -> List[int]:
    """Extracts list of integers present in string."""
    number_strs = re.findall(r"\d+", string)
    return [int(s) for s in number_strs]


def import_and_select_trace(
    path2mat: str, Iinj: dict = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """Import electrophysiological recordings stored in .mat files into t, V(t) Tensors.

    This function takes the Path to a .mat file as input and optionally the
    current parameters of the stimulation protocol 'stim_start', 'stim_end', 'duration'
    of the whole I(t). If provided, I(t) will be returned along with V(t), otherwise
    it will be left blank.

    WARNING: Some Values in this function are hard coded, thus works
    with specific recordings only at the moment.

    Args:
        path2mat: Path to the file location.
        Iinj: Dictionary containing current specifics.

    Returns:
        t: Time axis in steps of dt in ms.
        Vt: Membrane voltage in mV.
        It: Stimulation current in pA.
    """
    data = loadmat(path2mat)
    trace_keys = [key for key in data.keys() if "Trace" in key]
    trace_tags = torch.vstack(
        [torch.tensor(ints_from_str(x)) for x in trace_keys if ints_from_str(x) != []]
    )
    # print(tags)
    num_electrodes = int(torch.max(trace_tags[:, -1]))
    num_samples = int(len(trace_tags) / num_electrodes)
    num_bins = len(list(data.values())[10])  # arbitratry

    t = torch.zeros(num_electrodes, num_samples, num_bins)
    Vt = torch.zeros(num_electrodes, num_samples, num_bins)
    It = torch.zeros(num_electrodes, num_samples, num_bins)

    for tags, key in zip(trace_tags, trace_keys):
        trace = torch.tensor(data[key])
        elec_idx = tags[-1] - 1
        sample_idx = tags[-2] - 1
        if trace.ndim > 1:
            t[elec_idx, sample_idx, :], Vt[elec_idx, sample_idx, :] = trace.T

    if Iinj != None:
        It = torch.ones_like(Vt)
        dt = (t[:, :, 1] - t[:, :, 0]) * 1000
        if "stim_onset" in Iinj.keys():
            t1 = torch.max((Iinj["stim_onset"] / dt).long())
            t2 = torch.max((Iinj["stim_end"] / dt).long())
        else:
            t1 = torch.max((Iinj["stim_onset [ms]"] / dt).long())
            t2 = torch.max((Iinj["stim_end [ms]"] / dt).long())

        if "duration" in Iinj.keys():
            T = torch.max((Iinj["duration"] / dt).long())
        else:
            if "t_start [ms]" in Iinj.keys():
                T = torch.max(((Iinj["t_end [ms]"] - Iinj["t_start [ms]"]) / dt).long())

        I = torch.arange(
            -200, num_samples * 20 - 200, 20
        )  # params are hardcoded for now
        It[:, :, :t1] = 0
        It[:, :, t2:] = 0
        It[:, :, t1:t2] = I.reshape(1, num_samples, 1)

        t = t[:, :, :T]
        Vt = Vt[:, :, :T]
        It = It[:, :, :T]

    t = t * 1000
    Vt = Vt * 1000
    return t, Vt, It


def plot_vtrace(
    t: Tensor,
    Vt: Tensor,
    It: Optional[Tensor] = None,
    figsize: Tuple = (6, 4),
    title="",
    timewindow=None,
    **plot_kwargs
) -> Tuple[Figure, Axes]:
    """Plot voltage and possibly current trace.

    Args:
        t: Time axis in steps of dt.
        Vt: Membrane voltage.
        It: Stimulus.
        figsize: Changes the figure size of the plot.
        title: Adds a custom title to the figure.
        timewindow: The voltage and current trace will only be plotted
            between (t1,t2). To be specified in secs.
    Returns:
        fig: plt.Figure.
        axes: plt.Axes.
    """

    start, end = (0, -1)

    if timewindow != None:
        dt = t[1] - t[0]
        start, end = (torch.tensor(timewindow) / dt).int()

    Vts = Vt[:, start:end]
    ts = t[:, start:end]

    if It == None:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
        for t_i, Vt_i in zip(ts, Vts):
            axes[0].plot(t_i.numpy(), Vt_i.numpy(), lw=2, **plot_kwargs)

    else:
        Its = It[:, start:end]

        fig, axes = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )
        for t_i, Vt_i in zip(ts, Vts):
            axes[0].plot(t_i.numpy(), Vt_i.numpy(), lw=2, **plot_kwargs)

        for t_i, It_i in zip(ts, Its):
            axes[1].plot(t_i.numpy(), It_i.numpy(), lw=2, c="grey")

    for i, ax in enumerate(axes):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)

        ax.set_xticks(torch.linspace(t[0, 0], t[0, -1], 3).numpy())
        ax.set_xticklabels(torch.linspace(t[0, 0], t[0, -1], 3).numpy())
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))
        ax.xaxis.set_tick_params(width=2)

        ax.set_yticks(
            torch.linspace(
                torch.round(torch.min(Vt)), torch.round(torch.max(Vt)), 2
            ).numpy()
        )
        ax.set_yticklabels(torch.linspace(torch.min(Vt), torch.max(Vt), 2).numpy())
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))
        ax.yaxis.set_tick_params(width=2)

        if i == 0:
            ax.set_title(title)
            ax.set_ylabel("voltage (mV)", fontsize=12)
            if len(axes) == 1:
                ax.set_xlabel("time (ms)", fontsize=12)
            else:
                ax.xaxis.set_ticks_position("none")
            if "label" in plot_kwargs.keys():
                ax.legend(loc=1)
        if i == 1:
            ax.set_xlabel("time (ms)", fontsize=12)
            ax.set_ylabel("input (pA)", fontsize=12)
            ax.set_yticks([0, torch.max(It)])
    plt.tight_layout()

    return fig, axes

"""Helpers for plotting results."""

from __future__ import annotations

from math import sqrt

import matplotlib
import matplotlib.axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from loguru import logger

MAX_HEIGHT_INCHES = 8.0


def latexify(
    fig_width: float | None = None, fig_height: float | None = None, columns: int = 1
) -> None:
    """Set up matplotlib's RC params for LaTeX plotting.

    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """
    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    if columns not in (
        1,
        2,
    ):
        raise ValueError(f"Columns need to be either 1 or 2, but got {columns}")

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    if fig_height > MAX_HEIGHT_INCHES:
        logger.warning(
            f"fig_height too large: {fig_height} so will reduce to {MAX_HEIGHT_INCHES} "
            "inches."
        )
        fig_height = MAX_HEIGHT_INCHES

    params = {
        "backend": "ps",
        "text.latex.preamble": "\\usepackage{gensymb}",
        "axes.labelsize": 30,  # fontsize for x and y labels (was 30)
        "axes.titlesize": 30,  # was 30
        "font.size": 30,  # was 30
        "legend.fontsize": 30,  # was 30
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "text.usetex": False,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
        "axes.formatter.limits": [-3, 3],
        "axes.formatter.use_mathtext": True,
    }

    matplotlib.rcParams.update(params)


def create_fig_and_ax(
    num_fi: int = 1,
) -> tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, ...]]:
    """Creates base figure to plot."""
    ratios = [0.9]
    ratios.extend([0.03] * (num_fi))
    return plt.subplots(
        nrows=1,
        ncols=(num_fi + 1),
        figsize=(12, 6),
        gridspec_kw={"width_ratios": ratios},
    )


def plot_individual_fi(
    fig: matplotlib.figure.Figure,
    ax: tuple[matplotlib.axes.Axes, ...],
    individual_fi: npt.NDArray,
    x_range: npt.NDArray,
    other_param_range: npt.NDArray,
    color: str,
    cb_ax: int,
    label: str,
    hide_ticks: bool = True,
    rotation: int = 0,
    offset: float = 0.0,
) -> cm.ScalarMappable:
    """Plots qfi for a specific parameter.

    Produces multiple plots depending on the other parameter value as a color gradient.
    """
    norm = mcolors.Normalize(
        vmin=np.min(other_param_range), vmax=np.max(other_param_range)
    )
    cmap = cm.get_cmap(color)
    cmap = cmap(np.linspace(0.3, 1.0, 256))  # clip out the darkest 20%
    cmap = mcolors.ListedColormap(cmap)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(other_param_range)

    colors = cmap(norm(other_param_range))

    for i, _ in enumerate(other_param_range):
        _ = ax[0].plot(x_range, individual_fi[:, i], "-", color=colors[i])

    cb = fig.colorbar(sm, cax=ax[cb_ax])
    if hide_ticks:
        cb.set_ticks([])
    ax[cb_ax].set_xlabel(label).set_rotation(rotation)
    x, y = ax[cb_ax].xaxis.label.get_position()
    ax[cb_ax].xaxis.set_label_coords(x, y - offset)  # adjust offset to taste

    return sm

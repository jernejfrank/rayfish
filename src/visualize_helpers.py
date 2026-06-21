"""Helpers for plotting results."""

from __future__ import annotations

import itertools
from math import sqrt

import matplotlib
import matplotlib.axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from beartype.typing import Callable, Sequence
from loguru import logger
from matplotlib.lines import Line2D
from numpy.polynomial.polynomial import polyval

# Default colormap families, one per cross-talk level (index-aligned with the
# eps values passed to ``plot_fi_comparison``). Same order used across the
# example notebooks so figures stay visually consistent.
DEFAULT_FI_CMAPS = ("Greens", "Reds", "Purples", "Oranges", "Blues")

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
        "legend.fontsize": 20,  # was 30
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "text.usetex": False,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
        "axes.formatter.limits": [-3, 3],
        "axes.formatter.use_mathtext": True,
    }

    matplotlib.rcParams.update(params)


def measurement_mode_field(
    mode: dict[str, npt.NDArray],
    extent: float = 4.0,
    resolution: int = 200,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Reconstruct the real-space field of a single measurement mode.

    A mode is phi(x, y) = g(x) f(y) sqrt(1/pi) exp(-(x^2 + y^2) / 2), where g and f are
    polynomials whose ascending-power coefficients are stored under the "x" and "y" keys
    of the mode dict (as produced by ``measurement_basis``). The Gaussian envelope is
    the point-spread function the modes are matched to.

    Args:
        mode: a single ``{"x": coeffs, "y": coeffs}`` entry of the measurement basis.
        extent: half-width of the square plotting window in normalized units.
        resolution: number of grid points per axis.

    Returns:
        The ``(X, Y, field)`` meshgrid arrays, with ``field`` the (real) mode amplitude.
    """
    axis = np.linspace(-extent, extent, resolution)
    x_grid, y_grid = np.meshgrid(axis, axis)
    envelope = np.sqrt(1.0 / np.pi) * np.exp(-(x_grid**2 + y_grid**2) / 2.0)
    field = polyval(x_grid, mode["x"]) * polyval(y_grid, mode["y"]) * envelope
    return x_grid, y_grid, np.real(field)


def plot_measurement_modes(
    modes: list[dict[str, npt.NDArray]],
    mode_degree: int | None = None,
    extent: float = 4.0,
    resolution: int = 200,
    intensity: bool = False,
    cmap: str | None = None,
) -> tuple[matplotlib.figure.Figure, npt.NDArray]:
    """Plot every measurement mode on a ``mode_degree x mode_degree`` grid.

    Handy for eyeballing how the mode-mixing strength epsilon distorts the
    Hermite-Gaussian modes: pass the (possibly mixed) basis straight from
    ``measurement_basis``. Subplot ``(i, j)`` shows mode ``modes[i * mode_degree + j]``.
    Amplitude is drawn with a symmetric diverging colormap so sign changes are visible;
    set ``intensity=True`` to show ``|phi|^2`` instead.

    Args:
        modes: the measurement basis list (length ``mode_degree**2``).
        mode_degree: grid size; inferred from ``len(modes)`` when omitted.
        extent: half-width of each subplot window in normalized units.
        resolution: number of grid points per axis.
        intensity: plot ``|phi|^2`` instead of the signed amplitude.
        cmap: matplotlib colormap name; defaults to a diverging/intensity map.

    Returns:
        The created ``(figure, axes)``, where ``axes`` is a 2D array of shape
        ``(mode_degree, mode_degree)``.
    """
    if mode_degree is None:
        mode_degree = round(len(modes) ** 0.5)
    if cmap is None:
        cmap = "viridis" if intensity else "RdBu_r"

    fig, axes = plt.subplots(
        mode_degree,
        mode_degree,
        figsize=(2.2 * mode_degree, 2.2 * mode_degree),
        squeeze=False,
    )
    window = [-extent, extent, -extent, extent]
    for i in range(mode_degree):
        for j in range(mode_degree):
            _, _, field = measurement_mode_field(
                modes[i * mode_degree + j], extent, resolution
            )
            data = field**2 if intensity else field
            ax = axes[i, j]
            if intensity:
                ax.imshow(data, origin="lower", extent=window, cmap=cmap)
            else:
                vmax = np.max(np.abs(data)) or 1.0
                ax.imshow(
                    data,
                    origin="lower",
                    extent=window,
                    cmap=cmap,
                    vmin=-vmax,
                    vmax=vmax,
                )
            ax.set_title(f"({i},{j})", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    return fig, axes


def plot_mode_mixing_sweep(
    base_modes: list[dict[str, npt.NDArray]],
    epsilons: list[float],
    mix_fn: Callable,
    mode_indices: list[int] | None = None,
    mode_degree: int | None = None,
    extent: float = 4.0,
    resolution: int = 160,
    intensity: bool = False,
    cmap: str | None = None,
) -> tuple[matplotlib.figure.Figure, npt.NDArray]:
    """Grid of modes (rows) versus mixing strength (columns).

    Each row fixes one measurement mode and shows how it is progressively distorted as
    the mixing strength epsilon increases left to right, so cross-talk transferred into
    a mode is read along a single row. The amplitude colour scale is shared within each
    row so brightness changes across epsilon are comparable for that mode.

    Args:
        base_modes: the unmixed measurement basis (length ``mode_degree**2``).
        epsilons: mixing strengths for the columns, lowest to highest.
        mix_fn: callable ``mix_fn(modes, epsilon) -> mixed_modes`` (e.g. the basis
            module's ``_mix_modes_via_unitary``).
        mode_indices: flat indices of the modes to show as rows; defaults to all.
        mode_degree: grid size; inferred from ``len(base_modes)`` when omitted.
        extent: half-width of each subplot window in normalized units.
        resolution: number of grid points per axis.
        intensity: plot ``|phi|^2`` instead of the signed amplitude.
        cmap: matplotlib colormap name; defaults to a diverging/intensity map.

    Returns:
        The created ``(figure, axes)``, where ``axes`` is a 2D array of shape
        ``(len(mode_indices), len(epsilons))``.
    """
    if mode_degree is None:
        mode_degree = round(len(base_modes) ** 0.5)
    if mode_indices is None:
        mode_indices = list(range(len(base_modes)))
    if cmap is None:
        cmap = "viridis" if intensity else "RdBu_r"

    # Mix once per column, then index the chosen modes out of each mixed basis.
    mixed_per_eps = [mix_fn(base_modes, eps) for eps in epsilons]
    window = [-extent, extent, -extent, extent]

    n_rows, n_cols = len(mode_indices), len(epsilons)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.7 * n_cols, 1.7 * n_rows),
        squeeze=False,
    )
    for row, mode_idx in enumerate(mode_indices):
        fields = [
            measurement_mode_field(mixed[mode_idx], extent, resolution)[2]
            for mixed in mixed_per_eps
        ]
        data = [f**2 if intensity else f for f in fields]
        # Shared colour scale per row so the same mode is comparable across epsilon.
        row_vmax = max((np.max(np.abs(d)) for d in data), default=1.0) or 1.0
        for col in range(n_cols):
            ax = axes[row, col]
            if intensity:
                ax.imshow(
                    data[col],
                    origin="lower",
                    extent=window,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=row_vmax,
                )
            else:
                ax.imshow(
                    data[col],
                    origin="lower",
                    extent=window,
                    cmap=cmap,
                    vmin=-row_vmax,
                    vmax=row_vmax,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                i, j = divmod(mode_idx, mode_degree)
                ax.set_ylabel(f"({i},{j})", fontsize=10, rotation=0, labelpad=18)
            if row == 0:
                ax.set_title(rf"$\epsilon$={epsilons[col]:g}", fontsize=10)
    fig.tight_layout()
    return fig, axes


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


def eps_to_db(eps: float) -> float:
    """Convert a mode-mixing angle ``eps`` to cross-talk in decibels.

    The baseline ``eps == 0`` maps to ``-inf`` (no cross-talk).
    """
    return 20 * np.log10(np.sin(eps)) if eps > 0 else -np.inf


def _clipped_cmap(name: str) -> mcolors.ListedColormap:
    """Colormap with the darkest 30% clipped out (matches ``plot_individual_fi``)."""
    return mcolors.ListedColormap(cm.get_cmap(name)(np.linspace(0.3, 1.0, 256)))


def _rep_color(name: str, t: float = 0.85) -> tuple[float, float, float, float]:
    """A single representative color from a colormap family, for legend swatches."""
    return _clipped_cmap(name)(t)


def plot_fi_comparison(
    fi_fields: Sequence[npt.NDArray],
    x_range: npt.NDArray,
    gradient_range: npt.NDArray,
    eps_values: Sequence[float],
    *,
    cmaps: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    gradient_label: str = "",
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    legend_title: str = "cross-talk (dB)",
    legend_ncol: int = 2,
    baseline_first: bool = False,
    gradient_cmap: str = "Greys",
) -> tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, ...]]:
    """Overlay Fisher-information curves for several cross-talk levels.

    Each entry in ``fi_fields`` is a 2D array of shape
    ``[len(x_range), len(gradient_range)]`` drawn as a family of curves: one
    line per value of ``gradient_range``, colored light->dark within a single
    colormap family. The *family* (hue) encodes the cross-talk level and is
    decoded by the legend; the light->dark progression encodes
    ``gradient_range`` and is decoded by a single shared (neutral) colorbar --
    replacing the redundant per-level colorbars, since every level shares the
    same ``gradient_range`` normalization.

    Parameters
    ----------
    fi_fields :
        One 2D field per cross-talk level, index-aligned with ``eps_values``.
    x_range :
        Horizontal-axis values (shared by every curve).
    gradient_range :
        The swept parameter encoded by the within-family gradient / colorbar.
    eps_values :
        Mode-mixing angles, index-aligned with ``fi_fields``; ``eps == 0`` is
        the baseline and is labelled as such (and placed last in the legend).
    cmaps :
        Colormap family names, index-aligned with ``fi_fields``. Defaults to
        :data:`DEFAULT_FI_CMAPS`.
    legend_ncol :
        Number of legend columns. Entries read left-to-right, top-to-bottom in
        order of increasing cross-talk (dB), with the baseline last.
    baseline_first :
        If True, the baseline sits alone in the first row (its row padded with
        blank entries) and the cross-talk levels follow in ascending dB order.
    """
    if cmaps is None:
        cmaps = DEFAULT_FI_CMAPS[: len(fi_fields)]
    if not (len(fi_fields) == len(eps_values) == len(cmaps)):
        raise ValueError(
            "fi_fields, eps_values and cmaps must have the same length, got "
            f"{len(fi_fields)}, {len(eps_values)}, {len(cmaps)}"
        )

    fig, axs = create_fig_and_ax(1)
    main, cax = axs[0], axs[1]

    norm = mcolors.Normalize(
        vmin=float(np.min(gradient_range)), vmax=float(np.max(gradient_range))
    )

    for field, name in zip(fi_fields, cmaps):
        colors = _clipped_cmap(name)(norm(gradient_range))
        for j in range(len(gradient_range)):
            main.plot(x_range, field[:, j], "-", color=colors[j])

    # single neutral colorbar showing the gradient direction
    sm = cm.ScalarMappable(cmap=_clipped_cmap(gradient_cmap), norm=norm)
    sm.set_array(gradient_range)
    fig.colorbar(sm, cax=cax)
    cax.set_ylabel(gradient_label)

    # legend: color family -> cross-talk level, ordered by increasing dB with
    # the baseline last. Reorder so matplotlib's column-major fill reads
    # left-to-right, top-to-bottom.
    labelled = [
        (
            eps,
            Line2D(
                [],
                [],
                color=_rep_color(name),
                lw=6,
                label="baseline" if eps == 0 else f"{eps_to_db(eps):.2f}",
            ),
        )
        for eps, name in zip(eps_values, cmaps)
    ]
    nonbaseline = sorted((h for h in labelled if h[0] != 0), key=lambda h: h[0])
    baseline = [h for h in labelled if h[0] == 0]
    if baseline_first and baseline:
        # baseline alone on the top row; pad the rest of that row with blanks
        blanks = [
            Line2D([], [], linestyle="none", label="")
            for _ in range(max(0, legend_ncol - len(baseline)))
        ]
        reading = [h for _, h in baseline] + blanks + [h for _, h in nonbaseline]
    else:
        reading = [handle for _, handle in (*baseline, *nonbaseline)]
    column_major = list(
        itertools.chain(*[reading[i::legend_ncol] for i in range(legend_ncol)])
    )
    main.legend(
        handles=column_major,
        title=legend_title,
        frameon=False,
        loc="best",
        ncol=legend_ncol,
        handlelength=0.5,
        handletextpad=0.4,
        columnspacing=0.5,
    )

    if ylim is not None:
        main.set_ylim(*ylim)
    if xlim is not None:
        main.set_xlim(*xlim)

    main.set_xlabel(xlabel)
    main.set_ylabel(ylabel)

    pos_main = axs[0].get_position()
    pos_cbar = axs[1].get_position()
    gap = 0.01  # in figure fraction; smaller = tighter
    axs[1].set_position((
        pos_main.x1 + gap,
        pos_cbar.y0,
        pos_cbar.width,
        pos_cbar.height,
    ))

    return fig, axs

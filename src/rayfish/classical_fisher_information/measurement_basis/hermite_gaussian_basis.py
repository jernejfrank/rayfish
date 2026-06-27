"""Hermite Gaussian measurement basis.

The basis phi(x,y)_{j,l} = H_j(x) H_l(y) exp{-(x^2 + y^2)/2} / sqrt{pi 2^j j! 2^l l!},
is normalized to match the point spread function of the imaging system.
"""

from __future__ import annotations

from math import factorial

import numpy as np
import numpy.typing as npt
from beartype.typing import Dict, List
from hamilton.function_modifiers import pipe_output, source, step
from scipy.special import hermite

# Public API
__all__ = (
    "measurement_basis",
    "mode_mixing_strength",
)


def _mix_modes_via_unitary(
    modes: List[Dict[str, npt.NDArray[np.float64]]], mode_mixing_strength: float
) -> List[Dict[str, npt.NDArray[np.float64]]]:
    """Applies a Givens rotations that mixes nearest neighbor modes.

    For now only nearest neighbor modes are mixed since this is the most strong case in
    SPADE like device imperfections.
    """
    mode_order = len(modes[0]["x"])
    old_modes_x = np.stack([modes[i]["x"] for i in range(0, len(modes), mode_order)])
    old_modes_y = np.stack([modes[i]["y"] for i in range(mode_order)])
    unitary = _mixing_unitary(mode_order, mode_mixing_strength)

    new_modes_x = unitary @ old_modes_x
    new_modes_y = unitary @ old_modes_y

    basis = []
    for i in range(mode_order):
        for j in range(mode_order):
            b = {}
            b["x"] = new_modes_x[i]
            b["y"] = new_modes_y[j]
            basis.append(b)

    return basis


def _mixing_unitary(mode_degree: int, epsilon: float) -> npt.NDArray[np.float64]:
    unitary = np.eye(mode_degree)
    cos, sin = np.cos(epsilon), np.sin(epsilon)
    for n in range(mode_degree - 1):  # couple order n <-> n+1
        givens = np.eye(mode_degree)
        givens[n, n] = cos
        givens[n, n + 1] = -sin
        givens[n + 1, n] = sin
        givens[n + 1, n + 1] = cos
        unitary = givens @ unitary
    return unitary


def mode_mixing_strength(mode_mixing_eps: float = 0.0) -> float:
    """Defaults to 0 for backwards compatibility."""
    return mode_mixing_eps


def _select_mode_set(
    modes: List[Dict[str, npt.NDArray[np.float64]]], mode_set: tuple
) -> List[Dict[str, npt.NDArray[np.float64]]]:
    basis = []
    mode_order = round(len(modes) ** 0.5)
    for i in range(mode_order):
        for j in range(mode_order):
            if (i, j) in mode_set:
                mode_x = modes[i * mode_order + j]["x"]
                mode_y = modes[i * mode_order + j]["y"]
                basis.append({"x": mode_x, "y": mode_y})

    return basis


@pipe_output(
    step(_mix_modes_via_unitary, mode_mixing_strength=source("mode_mixing_strength")),
    step(_select_mode_set, mode_set=source("filter_mode_set")).when(filter_modes=True),
)
def measurement_basis(
    mode_degree: int,
) -> List[Dict[str, npt.NDArray[np.float64]]]:
    """Encodes the coefficients of the Hermite-Gaussian modes."""
    basis = []
    for i in range(mode_degree):
        for j in range(mode_degree):
            b = {}
            padded_x = np.zeros(mode_degree)
            hg_x = hermite(i).coeffs[::-1] / np.sqrt(2**i * factorial(i))
            padded_x[: len(hg_x)] = hg_x
            padded_y = np.zeros(mode_degree)
            hg_y = hermite(j).coeffs[::-1] / np.sqrt(2**j * factorial(j))
            padded_y[: len(hg_y)] = hg_y

            b["x"] = padded_x
            b["y"] = padded_y
            basis.append(b)
    return basis

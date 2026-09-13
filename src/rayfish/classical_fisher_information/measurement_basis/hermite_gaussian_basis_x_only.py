"""Hermite Gaussian measurement basis (x-only variant).

The basis phi(x,y)_{j,0} = H_j(x) H_0(y) exp{-(x^2 + y^2)/2} / sqrt{pi 2^j j!},
is normalized to match the point spread function of the imaging system. This is the
1D (width-only) restriction of the PSF-adapted mode basis used for the mode-sorting
Fisher information, keeping only the y=0 modes.
"""

from __future__ import annotations

from beartype.typing import Dict, List

# Public API
__all__ = ("measurement_basis",)

from math import factorial

import numpy as np
import numpy.typing as npt
from scipy.special import hermite


def measurement_basis(number_of_modes: int) -> List[Dict[str, npt.NDArray[np.float64]]]:
    """Encodes the coefficients of the Hermite-Gaussian modes."""
    basis = [{"x": np.array([1]), "y": np.array([1])}]
    for i in range(1, number_of_modes):
        b = {}
        coeffs = hermite(i).coeffs[::-1]
        b["x"] = coeffs / np.sqrt(2**i * factorial(i))
        b["y"] = np.array([1])
        basis.append(b)

    return basis

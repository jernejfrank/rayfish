"""Hermite Gaussian measurement basis.

The basis phi(x,y)_{j,l} = H_j(x) H_l(y) exp{-(x^2 + y^2)/2} / sqrt{pi 2^j j! 2^l l!},
is normalized to match the point spread function of the imaging system.
"""

from __future__ import annotations

from beartype.typing import Dict, List

# Public API
__all__ = ("measurement_basis",)

from math import factorial

import numpy as np
import numpy.typing as npt
from scipy.special import hermite


def measurement_basis(mode_degree: int) -> List[Dict[str, npt.NDArray[np.float64]]]:
    """Encodes the coefficients of the Hermite-Gaussian modes."""
    basis = []
    for i in range(mode_degree):
        for j in range(mode_degree):
            b = {}
            b["x"] = hermite(i).coeffs[::-1] / np.sqrt(2**i * factorial(i))
            b["y"] = hermite(j).coeffs[::-1] / np.sqrt(2**j * factorial(j))
            basis.append(b)

    return basis

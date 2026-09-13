"""Estimating the transverse separation of two incoherent point sources.

Two point sources separated along the x-axis and kept in focus (z = 0). Replicates the
canonical result [1]: a Hermite-Gaussian mode-sorting measurement (SPADE) saturates the
quantum Fisher information for the separation at every distance -- the FI stays flat
instead of collapsing as the sources merge (no Rayleigh's curse), so MS = QFI.

[1] M. Tsang, R. Nair, and X.-M. Lu, "Quantum theory of superresolution for two
    incoherent optical point sources," Phys. Rev. X 6, 031033 (2016).
"""

import numpy as np
import numpy.typing as npt

# Public API
__all__ = (
    "number_of_point_sources",
    "coordinates",
    "coordinate_jacobian",
)


def number_of_point_sources() -> int:
    """Define number of point sources.

    While in general this will be a user input at runtime. We hard-code it here since
    it is meant to be static.
    """
    return 2


def coordinates(
    grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Position coordinates of the point source distributions.

    The two sources are separated along the x-axis and left in the focal plane (no z
    displacement), so the grid -- which already carries the equidistant x-y spacing --
    is returned unchanged.

    Args:
        grid: (M,3N) array, where by default the grid[M,:2N] are populated with
            equidistant spacing in the x-y plane.

    Returns:
        A (M,3N) array, identical to the input grid.
    """
    return grid


def coordinate_jacobian() -> npt.NDArray[np.float64]:
    """Jacobian B of the re-parametrization to the single parameter of interest.

    Collapses the source-coordinate (Q)FIM onto the transverse separation d via
    H̃ = B H B^T. The estimated parameter d is the full separation between the two
    sources, which sit symmetrically at ∓d/2 along the x-axis, so ∂s1/∂d = -1/2 and
    ∂s2/∂d = +1/2. The 8 columns are the ordered source coordinates (s1,s2, t1,t2,
    p1,p2) followed by the two intensity parameters (zeroed since only the separation
    is estimated); only the x-columns are non-zero.
    """
    return np.array([[-1 / 2, 1 / 2, 0, 0, 0, 0, 0, 0]])

"""Replicating the results for estimating distance between 2 point sources.

This module creates 2 point sources positioned along the x-axis. We will replicate the
known results:
[1] M. Tsang, R. Nair, and X.-M. Lu, “Quantum theory of superresolution for two
incoherent optical point sources,” Phys. Rev. X 6, 031033 (2016).

It also serves as a template for now. Follow the same steps wand define the problem
how you need to get out parameters of interest to you.
"""

import numpy as np
import numpy.typing as npt
from hamilton.function_modifiers import pipe_output, source, step

from rayfish.parameter_catalogue import _matrix_multiplication, jacobian_1d

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
    """Position coordinates of point source distributions.

    Two point sources displaced along the x-axis only to replicate the results of Tsang,
    et. al., https://arxiv.org/abs/1511.00552.


    Args:
        grid: (M,3N) array, where by default the grid[M,:2N] are populated with
            equidistant spacing in the x-y plane.

    Returns:
        A (M,3N) array with in-place populated z-coordinates.
    """
    return grid


@pipe_output(
    step(jacobian_1d).named("z_dimension"),
    step(_matrix_multiplication, matrix=source("jacobian_to_collapse_xy")).named(
        "xy_dimension"
    ),
    step(_matrix_multiplication, matrix=source("jacobian_to_correct_sign")).named(
        "sign"
    ),
)
def coordinate_jacobian() -> npt.NDArray[np.float64]:
    """Jacobian B of the re-parametrization to the parameters of interest.

    Encodes the change of basis B_{α,b} = ∂θ_b/∂θ̃_α applied as
    H̃ = B H B^T (resp. J̃ = B J B^T), collapsing the 3N source-coordinate matrix to the
    parameters of interest. Standard transforms are chained via the above `pipe_output`
    scheme; here we keep only the in-plane distance parameter along the x-axis.
    """
    return np.array([[0, 0]])

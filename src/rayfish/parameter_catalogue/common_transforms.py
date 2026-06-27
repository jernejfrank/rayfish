"""Common re-parametrization transformations.

This module provides common transformations and the associated Jacobians B for
reparametrization that will be used in every parameter estimation problem. Each Jacobian
is the change of basis B_{α,b} = ∂θ_b/∂θ̃_α, applied to the (Q)FIM as H̃ = B H B^T
(resp. J̃ = B J B^T).

We model the underlying problem by a discrete distribution of point sources that are
displaced in the positive x-, y-, and z-directions. The Jacobians in this module correct
for sign changes and provide the option to collapse parameters of individual point
sources to common parameters (such as fixed distance between particles in a grid
ordering).

We assume the parameter order: [x1,x2,...,xN,y1,y2,...,yN,,z1,z2,...,zN].
"""

from enum import Enum

import numpy as np
import numpy.typing as npt

# Public API
__all__ = (
    "ParameterCatalogue",
    "_matrix_multiplication",
    "jacobian_to_correct_sign",
    "jacobian_to_collapse_xy",
    "jacobian_to_collapse_xy_separate",
    "jacobian_1d",
    "jacobian_2d",
)


class ParameterCatalogue(Enum):
    """Collecting all reparametrization implementations."""

    custom = "custom"
    dist_2_src = "distance_2_sources"


def _matrix_multiplication(
    user_jacobian: npt.NDArray[np.float64], matrix: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Helper function to multiply two matrices."""
    return np.matmul(user_jacobian, matrix)


def jacobian_to_correct_sign(
    grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Jacobian for the reparametrisation used in constructing the grid.

    We assume the Gaussian beams to be displace by positive (dx,dy,dz)-values in
    calculating the closed-form overlaps. The grid is created symmetrically around the
    origin of the coordinate system, so half the values are negative. We capture
    this reparametrisation get captured here in the form of the Jacobian which has
    +1 or -1 on the diagonal.

    Args:
        grid: (M,3N) array of particle distributions in 3D space

    Returns:
        A (3N,3N) array with the diagonal +1 or -1 depending if the particle is
        displaced by a positive or negative number.

    """
    sign_vector = np.sign(grid[0]) + (grid[0] == 0)
    return np.diag(sign_vector)


def jacobian_to_collapse_xy(
    number_of_point_sources: int,
) -> npt.NDArray[np.float64]:
    """Jacobian for reparametrizing the x-y coordinates to a single distance parameter.

    The grid in the x-y is isotropic so we can reduce the estimation parameter to only
    capture the relative distance between points. We place an identity for the
    parameters in the z-dimension since we will apply a user specif transformation
    afterwards to get the parameters of interest.

    Args:
        number_of_point_sources: N. Number of point sources distributed in space.

    Returns:
        A (1+N,3N) array with the first dimension reflecting the grid positioning in
        the x-y plane and the next N dimension being the identity.

    """
    jacobian = np.zeros((1 + number_of_point_sources, 3 * number_of_point_sources))
    factor = 1 / 2
    shift_by = 0
    if number_of_point_sources % 2 == 1:
        shift_by = 1
        jacobian[0, 0] = 1
        jacobian[0, number_of_point_sources] = 1
        jacobian[1, 2 * number_of_point_sources] = 1

    for point_source in range(shift_by, number_of_point_sources, 2):
        # x-dimnesion
        jacobian[0, point_source] = factor
        jacobian[0, point_source + 1] = factor
        # # y-dimnesion
        jacobian[0, number_of_point_sources + point_source] = factor
        jacobian[0, number_of_point_sources + point_source + 1] = factor

        factor += 1 / 2

    for point_source in range(shift_by, number_of_point_sources):
        # We leave z-dim alone here, user needs to specify how to collapse
        jacobian[1 + point_source, 2 * number_of_point_sources + point_source] = 1

    return jacobian


def jacobian_to_collapse_xy_separate(
    number_of_point_sources: int,
) -> npt.NDArray[np.float64]:
    """Jacobian for reparametrizing the x-y coordinates individual distance parameters.

    We reduce the distance in the x- and y-dimension seperately in case the grid is not
    symmetric in the two dimensions. We place an identity for the parameters in the
    z-dimension since we will apply a user specif transformation afterwards to get the
    parameters of interest.

    Args:
        number_of_point_sources: N. Number of point sources distributed in space.

    Returns:
        A (2+N,3N) array with the first dimension reflecting the grid distance in x and
        the second dimension the grid distance in y and the next N dimension being the
        identity.
    """
    jacobian = np.zeros((2 + number_of_point_sources, 3 * number_of_point_sources))
    factor = 1 / 2
    shift_by = 0
    if number_of_point_sources % 2 == 1:
        shift_by = 1
        jacobian[0, 0] = 1
        jacobian[1, number_of_point_sources] = 1
        jacobian[2, 2 * number_of_point_sources] = 1

    for point_source in range(shift_by, number_of_point_sources, 2):
        # x-dimnesion
        jacobian[0, point_source] = factor
        jacobian[0, point_source + 1] = factor
        # y-dimnesion
        jacobian[1, number_of_point_sources + point_source] = factor
        jacobian[1, number_of_point_sources + point_source + 1] = factor

        factor += 1 / 2

    for point_source in range(shift_by, number_of_point_sources):
        # We leave z-dim alone here, user needs to specify how to collapse
        jacobian[2 + point_source, 2 * number_of_point_sources + point_source] = 1

    return jacobian


def jacobian_1d(user_input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Creating Jacobian from user input by adding the distance parameter in plane.

    Embedding user_input array into the correct dimensional array to chain together the
    necessary Jacobian matrix transformation. We assume the x-y plane got collapsed into
    a single distance parameter between the points.

    Args:
        user_input: at least a (1,1) array that specifies how to reparametrize the
            parameters along the z-dimension

    Returns:
        An array of one dimension higher that add the x-y plane distance parameter.
    """
    jacobian = np.zeros((user_input.shape[0] + 1, user_input.shape[1] + 1))
    # x-y parameter
    jacobian[0, 0] = 1
    # z-dimension parameters
    jacobian[1:, 1:] = user_input
    return jacobian


def jacobian_2d(user_input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Creating Jacobian from user input by adding the distance parameters for x and y.

    Embedding user_input array into the correct dimensional array to chain together the
    necessary Jacobian matrix transformation. We assume both x- and y-dimension got
    collapsed into their respective distance parameters along the axes.

    Args:
        user_input: at least a (1,1) array that specifies how to reparametrize the
            parameters along the z-dimension

    Returns:
        An array of two dimension higher that add the x-dimension distance parameter and
        y-dimension distance parameter.
    """
    jacobian = np.zeros((user_input.shape[0] + 2, user_input.shape[1] + 2))
    # x-dimension parameter
    jacobian[0, 0] = 1
    # y-dimension parameter
    jacobian[1, 1] = 1
    # z-dimension parameters
    jacobian[2:, 2:] = user_input
    return jacobian

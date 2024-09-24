"""Modelling the imaging scenario with point sources.

This module creates point source distributions that model the underlying object.

For now we represent the 3D object as a uniform grid with each point having a varying
height (figure 1 in: https://www.azooptics.com/Article.aspx?ArticleID=1238). We compute
the jacobian of the grid parametrization as a pre-processing step for the covariance
matrix to perform dimensionality reduction to a single distance parameter in the x-y
plane.

We can choose to model 1D (line) or 2D (area) objects. Note that for a 1D model the
number of point sources N corresponds directly, but for 2D we will have N^2 point
sources (N in each dimension).
"""

import numpy as np
import numpy.typing as npt

# Public API
__all__ = (
    "LengthError",
    "spacing",
    "grid__1d",
    "coordinate_jacobian_sign",
    "coordinate_jacobian_collapse_plane_parameters",
)

# Parameter order
# [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN]


class LengthError(Exception):
    """Cannot distribute all the particles with minimal distance."""

    def __init__(
        self, number_of_point_sources: int, minimal_spacing: int, maximal_length: int
    ) -> None:
        """Raising error message."""
        super().__init__(
            f"Cannot fit {number_of_point_sources} "
            "point sources that have to be at"
            f"least {minimal_spacing} apart into a "
            f"line of length {maximal_length}."
        )


def spacing(
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    minimal_spacing: float,
    maximal_length: float,
) -> npt.NDArray[np.float64]:
    """Calculate the spacing parameter of the grid.

    We start with each point source being a distance `minimal_spacing` apart from
    each other. We than increase this distance to look at the parameter estimation
    behavior w.r.t. varying the spacing of the point sources. Each variation of the
    spacing is a unique distribution. In principle we have a continuous family of
    parameters between the minimal and maximal spacing permitted. We sample a discrete
    number of distributions.

    Args:
        number_of_point_sources: number of point sources we use to model the object
        number_of_point_source_distributions: number of discrete sampling states
        minimal_spacing: lower bound of how close point sources can be located
        maximal_length: upper bound of how far apart the outer sources can be located

    Returns:
        A (M,) array, where M = number_of_point_source_distributions, is the number of
        discrete sampling points in varying the distance between point sources.
    """
    if maximal_length / number_of_point_sources < minimal_spacing:
        raise LengthError(
            number_of_point_sources=number_of_point_sources,
            minimal_spacing=minimal_spacing,
            maximal_length=maximal_length,
        )

    return (
        np.linspace(
            minimal_spacing,
            maximal_length / number_of_point_sources,
            number_of_point_source_distributions,
            endpoint=True,
        )
        / 2
    )


def grid__1d(
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    spacing: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Creates 1D grid in the x-dimension.

    W.L.O.G. we can assume the 1D axis to be along the x-axis (due to rotational
    symmetry of the plane and the Gaussian beam). The point sources are distributed
    in a symmetric manner around the origin.

    If even number of sources: they are placed at +spacing and -spacing from the
    origin.
    If odd number of sources: one point source is placed at origin (0,0).

    Args:
        number_of_point_sources: number of point sources we use to model the object
        number_of_point_source_distributions: number of discrete sampling states
        spacing: (M,) array of distances between point sources

    Returns:
        A (M,3N) array with the first (M,N) values corresponding to the distributions
        of point sources for varying spacings. This grid is meant to be populated
        by a similar (M,N) array for the z-dimension
        grid__1D[M,2N:] = height_distribution.
    """
    grid = np.zeros((number_of_point_source_distributions, 3 * number_of_point_sources))

    for i, distance in enumerate(spacing):
        # Placing one point source at the origin of the coordinate system
        shift_by = 0
        if number_of_point_sources % 2 == 1:
            shift_by = 1

        second_index = 0
        for point_source in range(number_of_point_sources - shift_by):
            prefactor = 1
            if point_source % 2 == 0:
                prefactor = -1
                second_index += 1

            grid[i, point_source + shift_by] = prefactor * second_index * distance

    return grid


def coordinate_jacobian_sign(
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


def coordinate_jacobian_collapse_plane_parameters(
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
        # y-dimnesion
        jacobian[0, number_of_point_sources + point_source] = factor
        jacobian[0, number_of_point_sources + point_source + 1] = factor

        factor += 1 / 2

    for point_source in range(shift_by, number_of_point_sources):
        # We leave z-dim alone here, user needs to specify how to collapse
        jacobian[1 + point_source, 2 * number_of_point_sources + point_source] = 1

    return jacobian

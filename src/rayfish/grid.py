"""Modelling the imaging scenario with point sources.

This module creates point source distributions that model the underlying object.

For now we represent the 3D object as a uniform grid with each point having a varying
height (figure 1 in: https://www.azooptics.com/Article.aspx?ArticleID=1238).

We can choose to model 1D (line) or 2D (area) objects. Note that for a 1D model the
number of point sources N corresponds directly, but for 2D we will have N^2 point
sources (N in each dimension).
"""

import numpy as np
import numpy.typing as npt
from hamilton.function_modifiers import config

# Public API
__all__ = (
    "LengthError",
    "NumberNotSquareError",
    "spacing",
    "grid__1d",
    "grid__2d",
)

# Parameter order
# [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN]


class LengthError(Exception):
    """Cannot distribute all the particles with minimal distance."""

    def __init__(
        self,
        number_of_point_sources: int,
        minimal_spacing: float,
        maximal_length: float,
    ) -> None:
        """Raising error message."""
        super().__init__(
            f"Cannot fit {number_of_point_sources} "
            "point sources that have to be at"
            f"least {minimal_spacing} apart into a "
            f"line of length {maximal_length}."
        )


class NumberNotSquareError(Exception):
    """Number of point sources needs to be a perfect square for a 2D grid.."""

    def __init__(self, number_of_point_sources: int) -> None:
        """Raising error message."""
        super().__init__(
            f"Cannot create a grid out of {number_of_point_sources} "
            "point sources. Has to be perfect square."
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

    return np.linspace(
        minimal_spacing,
        maximal_length / number_of_point_sources,
        number_of_point_source_distributions,
        endpoint=True,
    )


@config.when_not(dimension=2)
def grid__1d(
    number_of_point_sources: int,
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
        spacing: (M,) array of distances between point sources

    Returns:
        A (M,3N) array with the first (M,N) values corresponding to the distributions
        of point sources for varying spacings. This grid is meant to be populated
        by a similar (M,N) array for the z-dimension
        grid__1D[M,2N:] = height_distribution.
    """
    grid_relative_spacing = np.zeros(3 * number_of_point_sources)

    # Placing one point source at the origin of the coordinate system
    shift_by = 0
    if number_of_point_sources % 2 == 1:
        shift_by = 1

    relative_position = 0
    for point_source in range(number_of_point_sources - shift_by):
        prefactor = 1
        if point_source % 2 == 0:
            prefactor = -1
            relative_position += 1

        grid_relative_spacing[point_source + shift_by] = prefactor * relative_position

    return spacing.reshape(-1, 1) * grid_relative_spacing.reshape(1, -1)


@config.when(dimension=2)
def grid__2d(
    number_of_point_sources: int,
    spacing: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Creates 2D grid in the x-y plane.

    If even number of sources: they are placed at +spacing and -spacing from the
    origin.
    If odd number of sources: one point source is placed at origin (0,0).

    Args:
        number_of_point_sources: number of point sources we use to model the object
        spacing: (M,) array of distances between point sources

    Returns:
        A (M,3N) array with the first (M,N) values corresponding to the distributions
        of point sources for varying spacings. This grid is meant to be populated
        by a similar (M,N) array for the z-dimension
        grid__2D[M,2N:] = height_distribution.
    """
    if number_of_point_sources < 4:
        raise ValueError("Minimal number of point sources for 2D is 4.")  # noqa

    # Make sure it is a perfect square - should work for realistic use cases but not
    # for gigantic numbers
    if number_of_point_sources**0.5 % 1 != 0:
        raise NumberNotSquareError(number_of_point_sources=number_of_point_sources)

    _sqrt_number_of_point_sources = int(np.sqrt(number_of_point_sources))

    _rel_x = np.zeros(_sqrt_number_of_point_sources)

    # Placing one point source at the origin of the coordinate system
    shift_by = 0
    if _sqrt_number_of_point_sources % 2 == 1:
        shift_by = 1

    relative_position = 0
    for point_source in range(_sqrt_number_of_point_sources - shift_by):
        prefactor = 1
        if point_source % 2 == 0:
            prefactor = -1
            relative_position += 1
        _rel_x[point_source + shift_by] = prefactor * relative_position

    x, y = np.meshgrid(_rel_x, _rel_x)
    grid_relative_spacing = np.concatenate(
        (
            x.flatten(order="F"),
            y.flatten(order="F"),
            np.zeros_like(x).flatten(order="F"),
        ),
        axis=0,
    )
    return spacing.reshape(-1, 1) * grid_relative_spacing.reshape(1, -1)

"""Simulation of precision bounds for 3 point source distribution.

The 3 sources are located along the x-axis. The middle source is at the coordinate
system origin, but displaced in the z-direction. The remaining two point sources are
displaced by equal amount left and right from the origin of the coordinate system in the
x-direction:

                 |
                 |
                 |
------X----------o----------X------
                 |
                 |
                 X
                 |

where `X`'s are point sources and `o` is the origin of the coordinate system.
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
    return 3


def coordinates(
    grid: npt.NDArray[np.float64], min_z_displacement: float, max_z_displacement: float
) -> npt.NDArray[np.float64]:
    """Position coordinates of point source distributions.

    For now we just use the same distance scale as for x.

    Args:
        grid: (M,3N) array, where by default the grid[M,:2N] are populated with
            equidistant spacing in the x-y plane.
        min_z_displacement: minimum displacement along the z-axis.
        max_z_displacement: maximum displacement along the z-axis.

    Returns:
        A (M,3N) array with in-place populated z-coordinates.
    """
    points = []
    z_displacement_range = np.linspace(
        min_z_displacement, max_z_displacement, grid.shape[0], endpoint=True
    )
    for row in grid:
        for displacement in z_displacement_range:
            single_xy_coords = row.copy()
            single_xy_coords[6] = -displacement

            points.append(single_xy_coords)

    return np.stack(points)


def coordinate_jacobian() -> npt.NDArray[np.float64]:
    """Specifiying parameters of interest.

    This can be any reparametrization from 3N -> parameter number of interest. We have
    some standard transformations that can be applied through the above `pipe_output`
    scheme, that chains reparametrizations together.

    In this case we just specify that we are interested in the distance between two
    point sources along the z-axis, and the above `pipe` collapses the covariance matrix
    to another distance parameter along the x-axis.
    """
    return np.array([
        [0, -1 / 2, 1 / 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0],
    ])

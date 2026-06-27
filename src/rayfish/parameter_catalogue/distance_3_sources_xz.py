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
    """Jacobian B of the re-parametrization to the crack width and depth.

    Encodes the change of basis B_{α,b} = ∂θ_b/∂θ̃_α that collapses the 3N
    source-coordinate (Q)FIM to the two parameters of interest, the crack width δx and
    depth δz,

        δx := δx_2 - δx_1,   δz := -δz_3,

    via H̃ = B H B^T. Rows correspond to (δx, δz) and the columns to the ordered source
    coordinates (s1,s2,s3, t1,t2,t3, p1,p2,p3).
    """
    return np.array([
        [0, -1 / 2, 1 / 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0],
    ])

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
from hamilton.function_modifiers import config

# Public API
__all__ = (
    "number_of_point_sources",
    "coordinates",
    "coordinate_jacobian__no_intensity",
    "coordinate_jacobian__with_intensity",
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

    Sweeps the z-displacement of the middle source (coordinate p1) linearly between
    ``min_z_displacement`` and ``max_z_displacement``, producing one distribution per
    combination of input grid row and z-displacement value.

    Args:
        grid: (M,3N) array, where by default the grid[M,:2N] are populated with
            equidistant spacing in the x-y plane.
        min_z_displacement: minimum displacement along the z-axis.
        max_z_displacement: maximum displacement along the z-axis.

    Returns:
        A (M*M,3N) array (M input rows times M z-displacement samples) with the middle
        source's z-coordinate populated.
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


@config.when_not(estimate_intensity=True)
def coordinate_jacobian__no_intensity() -> npt.NDArray[np.float64]:
    """Jacobian B of the re-parametrization to the crack width and depth.

    Encodes the change of basis B_{α,b} = ∂θ_b/∂θ̃_α that collapses the source-coordinate
    (Q)FIM to the two parameters of interest, the crack width δx and depth δz,

        δx := (δx_3 - δx_2) / 2,   δz := -δz_1,

    via H̃ = B H B^T. Rows correspond to (δx, δz) and the 12 columns to the ordered
    source coordinates (s1,s2,s3, t1,t2,t3, p1,p2,p3) followed by the three intensity
    parameters (zeroed here since intensity is not estimated).
    """
    return np.array([
        [0, -1 / 2, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    ])


@config.when(estimate_intensity=True)
def coordinate_jacobian__with_intensity(
    point_source_intensities: list[float],
    target_point_source_idx: int,
) -> npt.NDArray[np.float64]:
    """Jacobian B of the re-parametrization to the crack width, depth and intensity.

    Encodes the change of basis B_{α,b} = ∂θ_b/∂θ̃_α that collapses the source-coordinate
    (Q)FIM to the three parameters of interest, the crack width δx, depth δz and an
    intensity parameter,

        δx := (δx_3 - δx_2) / 2,   δz := -δz_1,

    via H̃ = B H B^T. Rows correspond to (δx, δz, intensity) and the 12 columns to the
    ordered source coordinates (s1,s2,s3, t1,t2,t3, p1,p2,p3) followed by the three
    intensity parameters, into which the intensity row maps.

    Args:
        point_source_intensities: relative intensities I_v of the sources.
        target_point_source_idx: index of the source whose intensity is treated as the
            parameter of interest.
    """
    intensity_jacobian = []
    for i, intensity in enumerate(point_source_intensities):
        if i == target_point_source_idx:
            other = (
                point_source_intensities[i + 1]
                if i < len(point_source_intensities)
                else point_source_intensities[i - 1]
            )
            intensity_jacobian.append((len(point_source_intensities) - 1) * other**2)
        else:
            intensity_jacobian.append(-(intensity**2))

    return np.array([
        [0, -1 / 2, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        [*[0 for _ in range(9)], *intensity_jacobian],
    ])

"""Testing object modelling.

We model objects through a distribution of point sources with (x,y,z) coordinates in
3D space.

We test that generating point sources distributions behaves as expected.
"""

import numpy as np
import numpy.typing as npt

from rayfish import parameter_catalogue


def test_jacobian__basic_xdim(
    mock_grid_1d_three_sources: npt.NDArray[np.float64],
) -> None:
    """Test Jacobian is diagonal with correct signs for x-dim."""
    expected = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])
    actual = parameter_catalogue.coordinate_jacobian_sign(
        grid=mock_grid_1d_three_sources
    )

    assert np.array_equiv(expected, actual)


def test_jacobian__basic_general() -> None:
    """Test Jacobian is diagonal with correct signs for all three dimensions."""
    grid = np.array([
        [-3, 3, -3, 3, -5, 5],
        [-6, 6, -3, 3, -5, 5],
        [-9, 9, -9, 9, -5, 5],
        [-12, 12, -12, 12, -5, 5],
    ])

    expected = np.array([
        [-1, 0, 0, 0, 0, 0],
        [
            0,
            1,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            -1,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            1,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            -1,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            1,
        ],
    ])
    actual = parameter_catalogue.coordinate_jacobian_sign(grid=grid)

    assert np.array_equiv(expected, actual)


def test_coordinate_jacobian_collapse_plane_parameters_2_sources() -> None:
    """Test Jacobian collapses xy even number of point sources."""
    expected = np.array([
        [0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    actual = parameter_catalogue.coordinate_jacobian_collapse_plane_parameters(2)
    assert np.array_equiv(expected, actual)


def test_coordinate_jacobian_collapse_plane_parameters_3_sources() -> None:
    """Test Jacobian collapses xy odd number of point sources."""
    expected = np.array([
        [1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    actual = parameter_catalogue.coordinate_jacobian_collapse_plane_parameters(3)
    assert np.array_equiv(expected, actual)

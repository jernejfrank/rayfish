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
    actual = parameter_catalogue.jacobian_to_correct_sign(
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
    actual = parameter_catalogue.jacobian_to_correct_sign(grid=grid)

    assert np.array_equiv(expected, actual)


def testjacobian_to_collapse_xy_2_sources() -> None:
    """Test Jacobian collapses xy even number of point sources."""
    expected = np.array([
        [0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    actual = parameter_catalogue.jacobian_to_collapse_xy(2)
    assert np.array_equiv(expected, actual)


def testjacobian_to_collapse_xy_3_sources() -> None:
    """Test Jacobian collapses xy odd number of point sources."""
    expected = np.array([
        [1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    actual = parameter_catalogue.jacobian_to_collapse_xy(3)
    assert np.array_equiv(expected, actual)


def testjacobian_to_collapse_xy_separate_2_sources() -> None:
    """Test Jacobian collapses xy even number of point sources."""
    expected = np.array([
        [0.5, 0.5, 0, 0, 0.0, 0.0],
        [0, 0, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    actual = parameter_catalogue.jacobian_to_collapse_xy_separate(2)
    assert np.array_equiv(expected, actual)


def testjacobian_to_collapse_xy_separate_3_sources() -> None:
    """Test Jacobian collapses xy odd number of point sources."""
    expected = np.array([
        [1.0, 0.5, 0.5, 0, 0, 0, 0.0, 0.0, 0.0],
        [0, 0, 0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    actual = parameter_catalogue.jacobian_to_collapse_xy_separate(3)
    assert np.array_equiv(expected, actual)


def test_create_jacobian_1d_single_z() -> None:
    """Test correct dimensional Jacobian matrix for x-y and single z."""
    expected = np.array([[1, 0], [0, 1 / 13]])

    user_input = np.array([[1 / 13]])

    actual = parameter_catalogue.jacobian_1d(user_input=user_input)
    assert np.array_equiv(expected, actual)


def test_create_jacobian_1d_collapse_z() -> None:
    """Test correct dimensional Jacobian matrix for x-y and collapsing z."""
    expected = np.array([[1, 0, 0], [0, 1 / 13, 1 / 13]])

    user_input = np.array([[1 / 13, 1 / 13]])

    actual = parameter_catalogue.jacobian_1d(user_input=user_input)
    assert np.array_equiv(expected, actual)


def test_create_jacobian_1d_multiple_z() -> None:
    """Test the correct dimensional Jacobian matrix for x-y and multiple z."""
    expected = np.array([[1, 0, 0, 0], [0, 1 / 13, 1 / 13, 0], [0, 0, 0, 3]])

    user_input = np.array([[1 / 13, 1 / 13, 0], [0, 0, 3]])

    actual = parameter_catalogue.jacobian_1d(user_input=user_input)
    assert np.array_equiv(expected, actual)


def test_create_jacobian_2d_single_z() -> None:
    """Test correct dimensional Jacobian matrix for x-y and single z."""
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1 / 13]])

    user_input = np.array([[1 / 13]])

    actual = parameter_catalogue.jacobian_2d(user_input=user_input)
    assert np.array_equiv(expected, actual)


def test_create_jacobian_2d_collapse_z() -> None:
    """Test correct dimensional Jacobian matrix for x-y and collapsing z."""
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 / 13, 1 / 13]])

    user_input = np.array([[1 / 13, 1 / 13]])

    actual = parameter_catalogue.jacobian_2d(user_input=user_input)
    assert np.array_equiv(expected, actual)


def test_create_jacobian_2d_multiple_z() -> None:
    """Test correct dimensional Jacobian matrix for x-y and multiple z."""
    expected = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1 / 13, 1 / 13, 0],
        [0, 0, 0, 0, 3],
    ])

    user_input = np.array([[1 / 13, 1 / 13, 0], [0, 0, 3]])

    actual = parameter_catalogue.jacobian_2d(user_input=user_input)
    assert np.array_equiv(expected, actual)

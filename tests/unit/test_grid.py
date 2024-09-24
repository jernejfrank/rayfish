"""Testing object modelling.

We model objects through a distribution of point sources with (x,y,z) coordinates in
3D space.

We test that generating point sources distributions behaves as expected.
"""

import numpy as np
import numpy.typing as npt
import pytest

import rayfish


class TestSpacing:
    """Testing correct spacing vector gets created."""

    def test_two_sources(
        self,
        mock_spacing_two_sources: npt.NDArray[np.float64],
    ) -> None:
        """Test two sources get correct distributions in a line."""
        expected = mock_spacing_two_sources

        actual = rayfish.spacing(
            number_of_point_sources=2,
            number_of_point_source_distributions=3,
            minimal_spacing=0.2,
            maximal_length=1.2,
        )

        assert np.array_equal(expected, actual)

    def test_three_sources(
        self,
        mock_spacing_three_sources: npt.NDArray[np.float64],
    ) -> None:
        """Test three sources get correct distributions in a line."""
        expected = mock_spacing_three_sources

        actual = rayfish.spacing(
            number_of_point_sources=3,
            number_of_point_source_distributions=4,
            minimal_spacing=3,
            maximal_length=36,
        )

        assert np.array_equal(expected, actual)

    def test_not_a_valid_distribution_error(self) -> None:
        """Test exception gets raised if wrong input."""
        n_point_sources = 4
        min_spacing = 0.5
        max_length = 1
        with pytest.raises(rayfish.LengthError) as excinfo:
            actual = rayfish.spacing(  # noqa: F841
                number_of_point_sources=n_point_sources,
                number_of_point_source_distributions=4,
                minimal_spacing=0.5,
                maximal_length=1,
            )
        assert str(excinfo.value) == (
            f"Cannot fit {n_point_sources} point sources that have to be at"
            f"least {min_spacing} apart into a line of length {max_length}."
        )


class TestGrid:
    """Testing point sources get distributed into the grid correctly."""

    def test_grid_1d_2_sources(
        self,
        mock_grid_1d_two_sources: npt.NDArray[np.float64],
        mock_spacing_two_sources: npt.NDArray[np.float64],
    ) -> None:
        """Testing correct grid for two point sources."""
        expected = mock_grid_1d_two_sources

        actual = rayfish.grid__1d(
            number_of_point_sources=2,
            number_of_point_source_distributions=3,
            spacing=mock_spacing_two_sources,
        )

        assert np.array_equiv(expected, actual)

    def test_grid_1d_3_sources(
        self,
        mock_grid_1d_three_sources: npt.NDArray[np.float64],
        mock_spacing_three_sources: npt.NDArray[np.float64],
    ) -> None:
        """Testing correct grid for two point sources."""
        expected = mock_grid_1d_three_sources

        actual = rayfish.grid__1d(
            number_of_point_sources=3,
            number_of_point_source_distributions=4,
            spacing=mock_spacing_three_sources,
        )

        assert np.array_equiv(expected, actual)


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
    actual = rayfish.coordinate_jacobian_sign(grid=mock_grid_1d_three_sources)

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
    actual = rayfish.coordinate_jacobian_sign(grid=grid)

    assert np.array_equiv(expected, actual)


def test_coordinate_jacobian_collapse_plane_parameters_2_sources() -> None:
    """Test Jacobian collapses xy even number of point sources."""
    expected = np.array([
        [0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    actual = rayfish.coordinate_jacobian_collapse_plane_parameters(2)
    assert np.array_equiv(expected, actual)


def test_coordinate_jacobian_collapse_plane_parameters_3_sources() -> None:
    """Test Jacobian collapses xy odd number of point sources."""
    expected = np.array([
        [1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    actual = rayfish.coordinate_jacobian_collapse_plane_parameters(3)
    assert np.array_equiv(expected, actual)

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
            number_of_grids=3,
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
            number_of_grids=4,
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
                number_of_grids=4,
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
            spacing=mock_spacing_three_sources,
        )

        assert np.array_equiv(expected, actual)

    def test_not_a_perfect_square_error(self) -> None:
        """Test exception if number of point sources is not a perfect square."""
        n_point_sources = 55
        with pytest.raises(rayfish.NumberNotSquareError) as excinfo:
            actual_error = rayfish.grid__2d(  # noqa: F841
                number_of_point_sources=n_point_sources,
                spacing=np.array([0.5]),
            )

        assert str(excinfo.value) == (
            f"Cannot create a grid out of {n_point_sources} "
            "point sources. Has to be perfect square."
        )

    def test_too_few_point_sources_error(self) -> None:
        """Test exception if number of point sources is not a perfect square."""
        n_point_sources = 2
        with pytest.raises(ValueError) as excinfo:
            actual_error = rayfish.grid__2d(  # noqa: F841
                number_of_point_sources=n_point_sources,
                spacing=np.array([0.5]),
            )

        assert str(excinfo.value) == ("Minimal number of point sources for 2D is 4.")

    def test_grid_2d_4_sources(
        self,
        mock_spacing_two_sources: npt.NDArray[np.float64],
    ) -> None:
        """Testing correct grid for two point sources."""
        expected = np.array([
            [-0.2, -0.2, 0.2, 0.2, -0.2, 0.2, -0.2, 0.2, 0.0, 0.0, 0.0, 0.0],
            [-0.4, -0.4, 0.4, 0.4, -0.4, 0.4, -0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
            [-0.6, -0.6, 0.6, 0.6, -0.6, 0.6, -0.6, 0.6, 0.0, 0.0, 0.0, 0.0],
        ])

        actual = rayfish.grid__2d(
            number_of_point_sources=4,
            spacing=mock_spacing_two_sources,
        )

        assert np.array_equiv(expected, actual)

    def test_grid_2d_9_sources(
        self,
        mock_spacing_three_sources: npt.NDArray[np.float64],
    ) -> None:
        """Testing correct grid for two point sources."""
        expected = np.array([
            [
                0,
                0,
                0,
                -3,
                -3,
                -3,
                3,
                3,
                3,
                0,
                -3,
                3,
                0,
                -3,
                3,
                0,
                -3,
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                -6,
                -6,
                -6,
                6,
                6,
                6,
                0,
                -6,
                6,
                0,
                -6,
                6,
                0,
                -6,
                6,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                -9,
                -9,
                -9,
                9,
                9,
                9,
                0,
                -9,
                9,
                0,
                -9,
                9,
                0,
                -9,
                9,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                -12,
                -12,
                -12,
                12,
                12,
                12,
                0,
                -12,
                12,
                0,
                -12,
                12,
                0,
                -12,
                12,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ])
        actual = rayfish.grid__2d(
            number_of_point_sources=9,
            spacing=mock_spacing_three_sources,
        )

        assert np.array_equiv(expected, actual)

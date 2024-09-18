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
        with pytest.raises(rayfish.LengthError) as excinfo:
            actual = rayfish.spacing(  # noqa: F841
                number_of_point_sources=4,
                number_of_point_source_distributions=4,
                minimal_spacing=0.5,
                maximal_length=1,
            )
        assert str(excinfo.value) == (
            "Cannot fit 4 point sources that have to be at"
            "least 0.5 apart into a line of length 1."
        )

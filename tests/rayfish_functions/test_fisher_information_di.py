"""Integration tests for fisher information of direct imaging."""

import numpy as np
import numpy.typing as npt
import pytest

from rayfish.di_fisher_information.incoherent.fisher_information_matrix import (
    _probability_derivatives,
    _total_probability,
    fisher_information_matrix_for_entire_distribution,
)


@pytest.fixture
def grid() -> tuple[npt.NDArray[np.float64], ...]:
    """Creates numerical grid."""
    grid_x = np.linspace(-5, 5)
    grid_ = np.meshgrid(grid_x, grid_x, indexing="xy")
    return grid_


@pytest.fixture
def coords() -> npt.NDArray[np.float64]:
    """Coordinates."""
    s = 0.23
    t = 0.21
    p = 0.121
    return np.array([
        [
            0,
            -s,
            s,
            0,
            -t,
            t,
            -p,
            0,
            0,
        ]
    ])


@pytest.fixture
def n_sources() -> int:
    """Number of sources."""
    return 3


def test_total_propability(
    coords: npt.NDArray[np.float64], n_sources: int, tolerance: float
) -> None:
    """Test the sum over individual point source probabilities."""
    single_x = np.array([-1.1])
    single_y = np.array([0.7])
    x_, y_ = np.meshgrid(single_x, single_y, indexing="xy")
    expected = 0.166664
    actual = _total_probability(x_, y_, coords, n_sources)
    assert abs(actual - expected) < tolerance


def test_propability_derivatives(
    coords: npt.NDArray[np.float64], n_sources: int, tolerance: float
) -> None:
    """Test individual derivatives for 3 point sources."""
    single_x = np.array([-1.1])
    single_y = np.array([0.7])
    x_, y_ = np.meshgrid(single_x, single_y, indexing="xy")
    expected = np.array([
        [
            0.127351,
            0.113562,
            0.113512,
            -0.0810413,
            -0.0418387,
            -0.118731,
            -0.00946236,
            0.0,
            0.0,
        ]
    ])

    actual = _probability_derivatives(x_, y_, coords, n_sources)
    np.testing.assert_allclose(
        actual=np.squeeze(actual), desired=np.squeeze(expected), atol=tolerance
    )


def test_di_fi_e2e(
    grid: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    coords: npt.NDArray[np.float64],
    tolerance: float,
) -> None:
    """Integration test for computing fisher information matrix of direct imaging."""
    expected = np.array([
        [0.218087, 0.0602803, 0.0197338],
        [0.0602803, 0.207105, 0.0180178],
        [0.0197338, 0.0180178, 0.00586729],
    ])

    x_, y_ = grid

    n_sources = 3
    n_dist = 1
    jacobian = np.array([
        [0, -1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0],
    ])

    actual = fisher_information_matrix_for_entire_distribution(
        coordinates=coords,
        number_of_point_sources=n_sources,
        number_of_point_source_distributions=n_dist,
        coordinate_jacobian=jacobian,
        numerical_grid=(x_, y_),
        batch_size=1,
    )
    assert len(actual) == 1
    np.testing.assert_allclose(actual=actual[0], desired=expected, atol=tolerance)

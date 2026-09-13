"""Integration tests for fisher information of direct imaging."""

import numpy as np
import numpy.typing as npt
import pytest
from hamilton import driver

import rayfish
from rayfish.classical_fisher_information import incoherent_di
from rayfish.classical_fisher_information.incoherent_di import (
    _probability_derivatives,
    _total_probability,
    fisher_information_matrix_for_entire_distribution,
)
from rayfish.parameter_catalogue import distance_3_sources_xz, intensity_distribution


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
    # Unweighted sum Σ_v |<psi|psi_v>|^2; equal intensities apply the 1/V = 1/3 weight.
    expected = 0.166664 / 3
    actual = _total_probability(
        x_, y_, coords, n_sources, point_source_intensities=[1 / 3, 1 / 3, 1 / 3]
    )
    assert abs(actual - expected) < tolerance


def test_propability_derivatives(
    coords: npt.NDArray[np.float64], n_sources: int, tolerance: float
) -> None:
    """Test individual derivatives for 3 point sources."""
    single_x = np.array([-1.1])
    single_y = np.array([0.7])
    x_, y_ = np.meshgrid(single_x, single_y, indexing="xy")
    # Unweighted single-source derivatives; equal intensities give the 1/V = 1/3 weight.
    # The trailing intensity block holds the raw (unweighted) single-source
    # probabilities, which double as the derivatives w.r.t. the intensities I_v.
    expected = np.concatenate(
        [
            np.array([
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
            / 3,
            np.array([[0.0587341, 0.0426926, 0.0652369]]),
        ],
        axis=1,
    )

    actual = _probability_derivatives(
        x_, y_, coords, n_sources, point_source_intensities=[1 / 3, 1 / 3, 1 / 3]
    )
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
    # 12 columns: 3N source coordinates followed by the N intensity parameters
    # (zeroed, intensity not estimated), matching the parameter catalogue layout.
    jacobian = np.array([
        [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    ])

    actual = fisher_information_matrix_for_entire_distribution(
        coordinates=coords,
        number_of_point_sources=n_sources,
        number_of_point_source_distributions=n_dist,
        coordinate_jacobian=jacobian,
        numerical_grid=(x_, y_),
        point_source_intensities=[1 / 3, 1 / 3, 1 / 3],
        batch_size=1,
    )
    assert len(actual) == 1
    np.testing.assert_allclose(actual=actual[0], desired=expected, atol=tolerance)


def test_di_fi_e2e_with_parameter_catalogue() -> None:
    """DI Fisher information wired through the parameter catalogue, as examples do.

    Guards the DI module against drifting out of sync with the catalogue's Jacobian
    layout (3N coordinates + N intensities): the catalogue Jacobians have 4N columns,
    so the DI derivative stack must expose the matching 4N parameter rows.
    """
    di_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            incoherent_di,
            distance_3_sources_xz,
            intensity_distribution,
        )
        .with_config(config={"simulation_dimension": 1})
    ).build()

    n_grids = 2
    result = di_dr.execute(
        inputs={
            "number_of_grids": n_grids,
            "maximal_length": 2.0,
            "minimal_spacing": 0.5,
            "min_z_displacement": 0.1,
            "max_z_displacement": 0.5,
            "numerical_integration_resolution": 4,
            "numerical_integration_support": 3.0,
            "batch_size": 4,
        },
        final_vars=["fisher_information_matrix_for_entire_distribution"],
    )

    fi = result["fisher_information_matrix_for_entire_distribution"]
    assert fi.shape == (n_grids * n_grids, 2, 2)
    assert np.all(np.isfinite(fi))
    np.testing.assert_allclose(fi, np.transpose(fi, (0, 2, 1)))
    assert np.all(np.diagonal(fi, axis1=1, axis2=2) > 0)

"""Regression tests for the mode-sorting Fisher information with remainder channel."""

import numpy as np
from hamilton import driver

import rayfish
from rayfish.classical_fisher_information import incoherent_mode_sorting
from rayfish.classical_fisher_information.incoherent_mode_sorting import (
    covariance_matrix_for_entire_distribution,
    covariance_matrix_for_entire_distribution_nuisance_intensity,
    covariance_matrix_for_entire_distribution_spatial,
)
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis,
)
from rayfish.parameter_catalogue import distance_2_sources_x

# Tsang's two-source result [PRX 6, 031033 (2016)] in these units: the QFI for the
# transverse separation is the constant 1/2, and Hermite-Gaussian mode sorting
# saturates it at every separation (no Rayleigh's curse).
TSANG_QFI = 0.5


def test_two_source_ms_fi_saturates_tsang_qfi() -> None:
    """MS FIM (incl. remainder channel) pins to Tsang's analytic two-source result.

    Protects the remainder-channel index bookkeeping in the per-channel FI assembly:
    the summed mode-sorting FI must stay within a fraction of a percent of the
    analytic QFI = 1/2 sub-Rayleigh, and must never exceed it (quantum bound).
    """
    n_grids = 4
    classical_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            incoherent_mode_sorting,
            distance_2_sources_x,
            hermite_gaussian_basis,
        )
        .with_config(config={"simulation_dimension": 1, "filter_modes": False})
    ).build()

    result = classical_dr.execute(
        inputs={
            "wave_vector": 2.0,
            "rayleigh_length": 1.0,
            "number_of_grids": n_grids,
            "maximal_length": 1.0,
            "minimal_spacing": 0.1,
            "point_source_intensities": [0.5, 0.5],
            "mode_degree": 6,
            "mode_mixing_eps": 0.0,
        },
        final_vars=["fisher_information_matrix_for_entire_distribution"],
    )

    # (M, B + 1, 1, 1): per-channel FI including the remainder bucket.
    per_channel = np.real(result["fisher_information_matrix_for_entire_distribution"])
    assert per_channel.shape == (n_grids, 6 * 6 + 1, 1, 1)

    fi_total = per_channel.sum(axis=1).reshape(n_grids)
    assert np.all(fi_total <= TSANG_QFI * (1 + 1e-9))
    assert np.all(fi_total >= 0.99 * TSANG_QFI)


def _synthetic_fi_stack() -> np.ndarray:
    """Two well-conditioned symmetric PSD 3x3 FIMs, split over 2 mock channels."""
    rng = np.random.default_rng(7)
    fis = []
    for _ in range(2):
        a = rng.normal(size=(3, 3))
        fis.append(a @ a.T + 3 * np.eye(3))
    full = np.stack(fis)
    # Split each FIM over two channels; the covariance helpers sum over axis 1.
    return np.stack([0.25 * full, 0.75 * full], axis=1)


def test_covariance_is_inverse_of_channel_sum() -> None:
    """Covariance equals the inverse of the channel-summed FIM."""
    fi = _synthetic_fi_stack()
    cov = covariance_matrix_for_entire_distribution(fi)
    np.testing.assert_allclose(cov, np.linalg.inv(fi.sum(axis=1)), atol=1e-12)


def test_nuisance_intensity_matches_full_inverse_block() -> None:
    """Schur-complement marginal equals the spatial block of the full inverse.

    The invert-then-slice identity (H^{-1})_SS = (H_SS - H_SI H_II^{-1} H_IS)^{-1}
    is the correctness condition for treating the trailing parameters as nuisance.
    """
    fi = _synthetic_fi_stack()
    n_spatial = 2
    cov = covariance_matrix_for_entire_distribution_nuisance_intensity(
        number_of_spatial_params=n_spatial,
        fisher_information_matrix_for_entire_distribution=fi,
    )
    full_inverse = np.linalg.inv(fi.sum(axis=1))
    np.testing.assert_allclose(cov, full_inverse[:, :n_spatial, :n_spatial], atol=1e-12)


def test_spatial_covariance_ignores_intensity_block() -> None:
    """Spatial covariance inverts only the leading block (known intensities)."""
    fi = _synthetic_fi_stack()
    n_spatial = 2
    cov = covariance_matrix_for_entire_distribution_spatial(
        fisher_information_matrix_for_entire_distribution=fi,
        number_of_spatial_params=n_spatial,
    )
    spatial_block = fi.sum(axis=1)[:, :n_spatial, :n_spatial]
    np.testing.assert_allclose(cov, np.linalg.inv(spatial_block), atol=1e-12)

"""Tests for the QFIM covariance helpers, incl. the complex-input Schur complement."""

import warnings

import numpy as np
import numpy.typing as npt

from rayfish.quantum_fisher_information.quantum_fisher_information import (
    covariance_matrix_for_entire_distribution,
    covariance_matrix_for_entire_distribution_nuisance_intensity,
    covariance_matrix_for_entire_distribution_spatial,
)


def _synthetic_complex_fim() -> npt.NDArray[np.complex128]:
    """Symmetric PSD FIMs with the small imaginary part a real QFIM pipeline carries.

    The QFIM is the real part of the complex Lyapunov covariance; numerically the
    computed matrix retains an O(1e-14) imaginary component that the covariance
    helpers must discard *before* inverting, not silently cast afterwards.
    """
    rng = np.random.default_rng(11)
    fims = []
    for _ in range(3):
        a = rng.normal(size=(4, 4))
        real = a @ a.T + 4 * np.eye(4)
        imag = 1e-14 * rng.normal(size=(4, 4))
        fims.append(real + 1j * imag)
    return np.stack(fims)


def test_covariance_real_and_inverse() -> None:
    """Plain covariance is the inverse of the real part, with real dtype."""
    fim = _synthetic_complex_fim()
    cov = covariance_matrix_for_entire_distribution(fim)
    assert cov.dtype == np.float64
    np.testing.assert_allclose(cov, np.linalg.inv(np.real(fim)), atol=1e-10)


def test_nuisance_schur_takes_real_part_first() -> None:
    """Schur marginal equals the spatial block of the real full inverse, warning-free.

    Regression for the complex Schur complement: the real part must be taken of the
    full FIM before the Schur complement so no ComplexWarning-casting occurs and the
    result matches the invert-then-slice ground truth.
    """
    fim = _synthetic_complex_fim()
    n_spatial = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a ComplexWarning would fail the test
        cov = covariance_matrix_for_entire_distribution_nuisance_intensity(
            number_of_spatial_params=n_spatial,
            fisher_information_matrix_for_entire_distribution=fim,
        )
    assert cov.dtype == np.float64
    full_inverse = np.linalg.inv(np.real(fim))
    np.testing.assert_allclose(cov, full_inverse[:, :n_spatial, :n_spatial], atol=1e-10)


def test_spatial_covariance_inverts_leading_block() -> None:
    """Spatial covariance inverts the real leading block only."""
    fim = _synthetic_complex_fim()
    n_spatial = 2
    cov = covariance_matrix_for_entire_distribution_spatial(
        fisher_information_matrix_for_entire_distribution=fim,
        number_of_spatial_params=n_spatial,
    )
    spatial_block = np.real(fim)[:, :n_spatial, :n_spatial]
    np.testing.assert_allclose(cov, np.linalg.inv(spatial_block), atol=1e-10)


def test_singular_fim_maps_to_infinite_covariance() -> None:
    """Singular distributions get an infinite covariance instead of crashing."""
    fim = _synthetic_complex_fim()
    fim[1] = 0.0  # make the middle distribution singular
    cov = covariance_matrix_for_entire_distribution(fim)
    assert np.all(np.isinf(cov[1]))
    np.testing.assert_allclose(cov[0], np.linalg.inv(np.real(fim[0])), atol=1e-10)

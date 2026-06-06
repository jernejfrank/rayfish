"""Unit tests for amplitude-level classical measurement helpers."""

import numpy as np

from rayfish import gaussian_overlaps
from rayfish.classical_fisher_information.incoherent import (
    amplitude_measurement_model,
)
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis_x_only,
)


def test_mode_amplitudes_match_overlap_primitive(
    tolerance: float,
    mock_wave_vector: int,
    mock_rayleigh_length: int,
) -> None:
    """Amplitude helper should reuse the closed-form overlap exactly."""
    coordinates = np.array([[0.13, -0.07, 0.0, 0.0, 0.09, -0.04]])
    basis = hermite_gaussian_basis_x_only.measurement_basis(number_of_modes=2)

    actual = amplitude_measurement_model.mode_amplitudes(
        coordinates=coordinates,
        number_of_point_sources=2,
        measurement_basis=basis,
        wave_vector=mock_wave_vector,
        rayleigh_length=mock_rayleigh_length,
    )

    expected = np.zeros((1, 2, 2), dtype=np.complex128)
    for mode_index, mode in enumerate(basis):
        for point_source in range(2):
            expected[0, mode_index, point_source] = (
                gaussian_overlaps.inner_product_measurement_mode_point_source_overlap(
                    s=coordinates[:, point_source],
                    t=coordinates[:, 2 + point_source],
                    p=coordinates[:, 4 + point_source],
                    k=mock_wave_vector,
                    zr=mock_rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            ).item()

    np.testing.assert_allclose(actual=actual, desired=expected, atol=tolerance)


def test_mode_amplitude_derivatives_match_overlap_primitives(
    tolerance: float,
    mock_wave_vector: int,
    mock_rayleigh_length: int,
) -> None:
    """Amplitude derivative helper should reuse the analytic derivatives exactly."""
    coordinates = np.array([[0.13, -0.07, 0.0, 0.0, 0.09, -0.04]])
    basis = hermite_gaussian_basis_x_only.measurement_basis(number_of_modes=1)

    actual_s, actual_t, actual_p = (
        amplitude_measurement_model.mode_amplitude_derivatives(
            coordinates=coordinates,
            number_of_point_sources=2,
            measurement_basis=basis,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )
    )

    expected_s = np.zeros((1, 1, 2), dtype=np.complex128)
    expected_t = np.zeros((1, 1, 2), dtype=np.complex128)
    expected_p = np.zeros((1, 1, 2), dtype=np.complex128)
    for point_source in range(2):
        expected_s[0, 0, point_source] = (
            gaussian_overlaps.derivate_s_inner_product_measurement_mode_point_source_overlap(
                s=coordinates[:, point_source],
                t=coordinates[:, 2 + point_source],
                p=coordinates[:, 4 + point_source],
                k=mock_wave_vector,
                zr=mock_rayleigh_length,
                mode_polynomial_coefficients=basis[0],
            )
        ).item()
        expected_t[0, 0, point_source] = (
            gaussian_overlaps.derivate_t_inner_product_measurement_mode_point_source_overlap(
                s=coordinates[:, point_source],
                t=coordinates[:, 2 + point_source],
                p=coordinates[:, 4 + point_source],
                k=mock_wave_vector,
                zr=mock_rayleigh_length,
                mode_polynomial_coefficients=basis[0],
            )
        ).item()
        expected_p[0, 0, point_source] = (
            gaussian_overlaps.derivate_p_inner_product_measurement_mode_point_source_overlap(
                s=coordinates[:, point_source],
                t=coordinates[:, 2 + point_source],
                p=coordinates[:, 4 + point_source],
                k=mock_wave_vector,
                zr=mock_rayleigh_length,
                mode_polynomial_coefficients=basis[0],
            )
        ).item()

    np.testing.assert_allclose(actual=actual_s, desired=expected_s, atol=tolerance)
    np.testing.assert_allclose(actual=actual_t, desired=expected_t, atol=tolerance)
    np.testing.assert_allclose(actual=actual_p, desired=expected_p, atol=tolerance)


def test_probabilities_from_amplitudes_matches_manual_average() -> None:
    """Probability helper should square amplitudes and average over sources."""
    amplitudes = np.array(
        [[[1 + 1j, 2 - 1j], [3 + 0j, 0 + 4j]]],
        dtype=np.complex128,
    )

    actual = amplitude_measurement_model.probabilities_from_amplitudes(
        amplitudes=amplitudes, number_of_point_sources=2
    )

    expected = np.array([[(2 + 5) / 2, (9 + 16) / 2]], dtype=np.float64)
    np.testing.assert_allclose(actual=actual, desired=expected)


def test_probability_derivatives_from_amplitudes_matches_manual_expression() -> None:
    """Derivative helper should apply the real conjugate-product rule."""
    amplitudes = np.array([[[1 + 2j, 2 - 1j]]], dtype=np.complex128)
    derivatives_s = np.array([[[0.5 - 1j, -1 + 0.5j]]], dtype=np.complex128)
    derivatives_t = np.array([[[1 + 0j, 0 + 1j]]], dtype=np.complex128)
    derivatives_p = np.array([[[0.25 + 0.5j, -0.75 + 0.25j]]], dtype=np.complex128)

    actual_s, actual_t, actual_p = (
        amplitude_measurement_model.probability_derivatives_from_amplitudes(
            amplitudes=amplitudes,
            derivatives_s=derivatives_s,
            derivatives_t=derivatives_t,
            derivatives_p=derivatives_p,
            number_of_point_sources=2,
        )
    )

    prefactor = 1.0
    conjugated = np.conjugate(amplitudes)
    expected_s = np.real(prefactor * conjugated * derivatives_s)
    expected_t = np.real(prefactor * conjugated * derivatives_t)
    expected_p = np.real(prefactor * conjugated * derivatives_p)

    np.testing.assert_allclose(actual=actual_s, desired=expected_s)
    np.testing.assert_allclose(actual=actual_t, desired=expected_t)
    np.testing.assert_allclose(actual=actual_p, desired=expected_p)


def test_projected_fisher_information_per_mode_matches_manual_projection() -> None:
    """Projection helper should divide by probability and apply the Jacobian."""
    probability = np.array([[2.0, 0.5]])
    derivatives_s = np.array([[[2.0], [1.0]]])
    derivatives_t = np.array([[[1.0], [3.0]]])
    derivatives_p = np.array([[[0.0], [2.0]]])
    coordinate_jacobian = np.array([[2.0, -1.0, 0.5]])

    actual = amplitude_measurement_model.projected_fisher_information_per_mode(
        probability=probability,
        derivatives_s=derivatives_s,
        derivatives_t=derivatives_t,
        derivatives_p=derivatives_p,
        coordinate_jacobian=coordinate_jacobian,
    )

    expected = np.zeros((1, 2, 1, 1))
    all_derivatives = np.concatenate(
        (derivatives_s, derivatives_t, derivatives_p), axis=2
    )
    for mode_index in range(2):
        derivative_vector = all_derivatives[0, mode_index]
        fisher = (
            np.outer(derivative_vector, derivative_vector) / probability[0, mode_index]
        )
        expected[0, mode_index, 0, 0] = (
            coordinate_jacobian @ fisher @ coordinate_jacobian.T
        ).item()

    np.testing.assert_allclose(actual=actual, desired=expected)

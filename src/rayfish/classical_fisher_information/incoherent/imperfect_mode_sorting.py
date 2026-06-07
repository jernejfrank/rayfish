"""Amplitude-level imperfect mode sorting for classical Fisher information."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from beartype.typing import Dict, List
from hamilton.function_modifiers import config

from rayfish import gaussian_overlaps

# Public API
__all__ = (
    "apply_mode_mixing",
    "append_loss_channel",
    "probability__1D",
    "fisher_information_matrix_for_entire_distribution",
)


def _validate_mode_mixing_matrix(
    mode_mixing_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    n_modes: int,
) -> None:
    if mode_mixing_matrix.shape != (n_modes, n_modes):
        raise ValueError(
            "mode_mixing_matrix has to have shape "
            f"({n_modes}, {n_modes}), got {mode_mixing_matrix.shape}."
        )


def _mode_amplitudes(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.complex128]:
    amplitudes = np.zeros(
        (coordinates.shape[0], len(measurement_basis), number_of_point_sources),
        dtype=np.complex128,
    )

    for mode_index, mode in enumerate(measurement_basis):
        for point_source in range(number_of_point_sources):
            amplitudes[:, mode_index, point_source] = (
                gaussian_overlaps.inner_product_measurement_mode_point_source_overlap(
                    s=coordinates[:, point_source],
                    t=coordinates[:, number_of_point_sources + point_source],
                    p=coordinates[:, 2 * number_of_point_sources + point_source],
                    k=wave_vector,
                    zr=rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            )

    return amplitudes


def _mode_amplitude_derivatives(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    wave_vector: float,
    rayleigh_length: float,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
]:
    shape = (coordinates.shape[0], len(measurement_basis), number_of_point_sources)
    derivatives_s = np.zeros(shape, dtype=np.complex128)
    derivatives_t = np.zeros(shape, dtype=np.complex128)
    derivatives_p = np.zeros(shape, dtype=np.complex128)

    for mode_index, mode in enumerate(measurement_basis):
        for point_source in range(number_of_point_sources):
            s = coordinates[:, point_source]
            t = coordinates[:, number_of_point_sources + point_source]
            p = coordinates[:, 2 * number_of_point_sources + point_source]

            derivatives_s[:, mode_index, point_source] = (
                gaussian_overlaps.derivate_s_inner_product_measurement_mode_point_source_overlap(
                    s=s,
                    t=t,
                    p=p,
                    k=wave_vector,
                    zr=rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            )
            derivatives_t[:, mode_index, point_source] = (
                gaussian_overlaps.derivate_t_inner_product_measurement_mode_point_source_overlap(
                    s=s,
                    t=t,
                    p=p,
                    k=wave_vector,
                    zr=rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            )
            derivatives_p[:, mode_index, point_source] = (
                gaussian_overlaps.derivate_p_inner_product_measurement_mode_point_source_overlap(
                    s=s,
                    t=t,
                    p=p,
                    k=wave_vector,
                    zr=rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            )

    return derivatives_s, derivatives_t, derivatives_p


def apply_mode_mixing(
    amplitudes: npt.NDArray[np.complex128],
    mode_mixing_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Apply a fixed mode-mixing matrix to the mode amplitudes."""
    _validate_mode_mixing_matrix(mode_mixing_matrix, amplitudes.shape[1])
    return np.einsum("ab,mbs->mas", mode_mixing_matrix, amplitudes)


def _probabilities_from_amplitudes(
    amplitudes: npt.NDArray[np.complex128],
    number_of_point_sources: int,
) -> npt.NDArray[np.float64]:
    return np.real(np.sum(np.abs(amplitudes) ** 2, axis=2) / number_of_point_sources)


def _probability_derivatives_from_amplitudes(
    amplitudes: npt.NDArray[np.complex128],
    derivatives_s: npt.NDArray[np.complex128],
    derivatives_t: npt.NDArray[np.complex128],
    derivatives_p: npt.NDArray[np.complex128],
    number_of_point_sources: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    scale = 2 / number_of_point_sources
    conjugated_amplitudes = np.conjugate(amplitudes)
    return (
        np.real(scale * conjugated_amplitudes * derivatives_s),
        np.real(scale * conjugated_amplitudes * derivatives_t),
        np.real(scale * conjugated_amplitudes * derivatives_p),
    )


def _projected_fisher_information_per_mode(
    probability: npt.NDArray[np.float64],
    derivatives_s: npt.NDArray[np.float64],
    derivatives_t: npt.NDArray[np.float64],
    derivatives_p: npt.NDArray[np.float64],
    coordinate_jacobian: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    derivatives = np.concatenate((derivatives_s, derivatives_t, derivatives_p), axis=2)
    fisher_information = np.einsum("mbi,mbj->mbij", derivatives, derivatives)

    mask = probability > 1e-16
    fisher_information[mask] /= probability[mask, np.newaxis, np.newaxis]
    fisher_information[~mask] = 0.0

    return np.einsum(
        "pi,mbij,qj->mbpq",
        coordinate_jacobian,
        fisher_information,
        coordinate_jacobian,
    )


def append_loss_channel(
    probability: npt.NDArray[np.float64],
    derivatives_s: npt.NDArray[np.float64],
    derivatives_t: npt.NDArray[np.float64],
    derivatives_p: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Append an explicit loss bucket and its derivatives."""
    loss_probability = 1 - np.sum(probability, axis=1, keepdims=True)
    if np.any(loss_probability < -1e-10):
        raise ValueError("Loss probability became negative beyond numerical tolerance.")

    loss_probability = np.clip(loss_probability, a_min=0.0, a_max=None)
    loss_derivative_s = -np.sum(derivatives_s, axis=1, keepdims=True)
    loss_derivative_t = -np.sum(derivatives_t, axis=1, keepdims=True)
    loss_derivative_p = -np.sum(derivatives_p, axis=1, keepdims=True)

    return (
        np.concatenate((probability, loss_probability), axis=1),
        np.concatenate((derivatives_s, loss_derivative_s), axis=1),
        np.concatenate((derivatives_t, loss_derivative_t), axis=1),
        np.concatenate((derivatives_p, loss_derivative_p), axis=1),
    )


@config.when(imperfect_mode_sorting=True)
def probability__1D(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    mode_mixing_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    include_loss_channel: bool,
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.float64]:
    """Probability of detecting a photon with imperfect mode sorting."""
    amplitudes = _mode_amplitudes(
        coordinates=coordinates,
        number_of_point_sources=number_of_point_sources,
        measurement_basis=measurement_basis,
        wave_vector=wave_vector,
        rayleigh_length=rayleigh_length,
    )
    mixed_amplitudes = apply_mode_mixing(
        amplitudes=amplitudes, mode_mixing_matrix=mode_mixing_matrix
    )
    probability = _probabilities_from_amplitudes(
        amplitudes=mixed_amplitudes,
        number_of_point_sources=number_of_point_sources,
    )

    if not include_loss_channel:
        return probability

    derivative_shape = (
        number_of_point_source_distributions,
        probability.shape[1],
        number_of_point_sources,
    )
    probability_with_loss, _, _, _ = append_loss_channel(
        probability=probability,
        derivatives_s=np.zeros(derivative_shape),
        derivatives_t=np.zeros(derivative_shape),
        derivatives_p=np.zeros(derivative_shape),
    )
    return probability_with_loss


@config.when(imperfect_mode_sorting=True)
def fisher_information_matrix_for_entire_distribution(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    probability: npt.NDArray[np.float64],
    wave_vector: float,
    rayleigh_length: float,
    coordinate_jacobian: npt.NDArray[np.float64],
    mode_mixing_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    include_loss_channel: bool,
) -> npt.NDArray[np.float64]:
    """Classical FI with amplitude-level mode mixing and optional truncation loss."""
    amplitudes = _mode_amplitudes(
        coordinates=coordinates,
        number_of_point_sources=number_of_point_sources,
        measurement_basis=measurement_basis,
        wave_vector=wave_vector,
        rayleigh_length=rayleigh_length,
    )
    derivatives_s, derivatives_t, derivatives_p = _mode_amplitude_derivatives(
        coordinates=coordinates,
        number_of_point_sources=number_of_point_sources,
        measurement_basis=measurement_basis,
        wave_vector=wave_vector,
        rayleigh_length=rayleigh_length,
    )

    mixed_amplitudes = apply_mode_mixing(
        amplitudes=amplitudes,
        mode_mixing_matrix=mode_mixing_matrix,
    )
    mixed_derivatives_s = apply_mode_mixing(
        amplitudes=derivatives_s,
        mode_mixing_matrix=mode_mixing_matrix,
    )
    mixed_derivatives_t = apply_mode_mixing(
        amplitudes=derivatives_t,
        mode_mixing_matrix=mode_mixing_matrix,
    )
    mixed_derivatives_p = apply_mode_mixing(
        amplitudes=derivatives_p,
        mode_mixing_matrix=mode_mixing_matrix,
    )

    (
        derivatives_s_probability,
        derivatives_t_probability,
        derivatives_p_probability,
    ) = _probability_derivatives_from_amplitudes(
        amplitudes=mixed_amplitudes,
        derivatives_s=mixed_derivatives_s,
        derivatives_t=mixed_derivatives_t,
        derivatives_p=mixed_derivatives_p,
        number_of_point_sources=number_of_point_sources,
    )

    if include_loss_channel:
        (
            probability,
            derivatives_s_probability,
            derivatives_t_probability,
            derivatives_p_probability,
        ) = append_loss_channel(
            probability=probability,
            derivatives_s=derivatives_s_probability,
            derivatives_t=derivatives_t_probability,
            derivatives_p=derivatives_p_probability,
        )

    return _projected_fisher_information_per_mode(
        probability=probability,
        derivatives_s=derivatives_s_probability,
        derivatives_t=derivatives_t_probability,
        derivatives_p=derivatives_p_probability,
        coordinate_jacobian=coordinate_jacobian,
    )

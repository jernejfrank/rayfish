"""Amplitude-level imperfect mode sorting for classical Fisher information."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from beartype.typing import Dict, List
from hamilton.function_modifiers import config

from rayfish.classical_fisher_information.incoherent import amplitude_measurement_model

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


def apply_mode_mixing(
    amplitudes: npt.NDArray[np.complex128],
    mode_mixing_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Apply a fixed mode-mixing matrix to the mode amplitudes."""
    _validate_mode_mixing_matrix(mode_mixing_matrix, amplitudes.shape[1])
    return np.einsum("ab,mbs->mas", mode_mixing_matrix, amplitudes)


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
    """Probability of detecting a photon with imperfect mode sorting.

    Args:
        coordinates: (M, 3N) point-source coordinates.
        number_of_point_sources: number of point sources in the distribution.
        number_of_point_source_distributions: number of coordinate rows being computed.
        measurement_basis: measured mode basis encoded as polynomial coefficients.
        mode_mixing_matrix: fixed amplitude-level mode-mixing matrix.
        include_loss_channel: append `1 - sum(p)` as an explicit outcome when `True`.
        wave_vector: phase parameter carried by the amplitude-level overlap model.
        rayleigh_length: length scale carried by the amplitude-level overlap model.

    Returns:
        A `(M, B)` array of measured mode probabilities, or `(M, B + 1)` when the
        loss channel is enabled.
    """
    amplitudes = amplitude_measurement_model.mode_amplitudes(
        coordinates=coordinates,
        number_of_point_sources=number_of_point_sources,
        measurement_basis=measurement_basis,
        wave_vector=wave_vector,
        rayleigh_length=rayleigh_length,
    )
    mixed_amplitudes = apply_mode_mixing(
        amplitudes=amplitudes, mode_mixing_matrix=mode_mixing_matrix
    )
    probability = amplitude_measurement_model.probabilities_from_amplitudes(
        amplitudes=mixed_amplitudes, number_of_point_sources=number_of_point_sources
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
    amplitudes = amplitude_measurement_model.mode_amplitudes(
        coordinates=coordinates,
        number_of_point_sources=number_of_point_sources,
        measurement_basis=measurement_basis,
        wave_vector=wave_vector,
        rayleigh_length=rayleigh_length,
    )
    derivatives_s, derivatives_t, derivatives_p = (
        amplitude_measurement_model.mode_amplitude_derivatives(
            coordinates=coordinates,
            number_of_point_sources=number_of_point_sources,
            measurement_basis=measurement_basis,
            wave_vector=wave_vector,
            rayleigh_length=rayleigh_length,
        )
    )

    mixed_amplitudes = apply_mode_mixing(
        amplitudes=amplitudes, mode_mixing_matrix=mode_mixing_matrix
    )
    mixed_derivatives_s = apply_mode_mixing(
        amplitudes=derivatives_s, mode_mixing_matrix=mode_mixing_matrix
    )
    mixed_derivatives_t = apply_mode_mixing(
        amplitudes=derivatives_t, mode_mixing_matrix=mode_mixing_matrix
    )
    mixed_derivatives_p = apply_mode_mixing(
        amplitudes=derivatives_p, mode_mixing_matrix=mode_mixing_matrix
    )

    probability_derivatives = (
        amplitude_measurement_model.probability_derivatives_from_amplitudes(
            amplitudes=mixed_amplitudes,
            derivatives_s=mixed_derivatives_s,
            derivatives_t=mixed_derivatives_t,
            derivatives_p=mixed_derivatives_p,
            number_of_point_sources=number_of_point_sources,
        )
    )

    if include_loss_channel:
        (
            probability,
            derivatives_s_probability,
            derivatives_t_probability,
            derivatives_p_probability,
        ) = append_loss_channel(
            probability=probability,
            derivatives_s=probability_derivatives[0],
            derivatives_t=probability_derivatives[1],
            derivatives_p=probability_derivatives[2],
        )
    else:
        (
            derivatives_s_probability,
            derivatives_t_probability,
            derivatives_p_probability,
        ) = probability_derivatives

    return amplitude_measurement_model.projected_fisher_information_per_mode(
        probability=probability,
        derivatives_s=derivatives_s_probability,
        derivatives_t=derivatives_t_probability,
        derivatives_p=derivatives_p_probability,
        coordinate_jacobian=coordinate_jacobian,
    )

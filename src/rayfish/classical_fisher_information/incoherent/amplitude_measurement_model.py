"""Shared amplitude-level helpers for classical mode measurements."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from beartype.typing import Dict, List

from rayfish import gaussian_overlaps

# Public API
__all__ = (
    "mode_amplitudes",
    "mode_amplitude_derivatives",
    "probabilities_from_amplitudes",
    "probability_derivatives_from_amplitudes",
    "projected_fisher_information_per_mode",
)


def mode_amplitudes(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.complex128]:
    """Compute ideal mode amplitudes for each distribution and source."""
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


def mode_amplitude_derivatives(
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
    """Compute analytic mode-amplitude derivatives for each source coordinate."""
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


def probabilities_from_amplitudes(
    amplitudes: npt.NDArray[np.complex128], number_of_point_sources: int
) -> npt.NDArray[np.float64]:
    """Compute mode probabilities from mode amplitudes."""
    return np.real(np.sum(np.abs(amplitudes) ** 2, axis=2) / number_of_point_sources)


def probability_derivatives_from_amplitudes(
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
    """Propagate amplitude derivatives to probability derivatives."""
    scale = 2 / number_of_point_sources
    conjugated_amplitudes = np.conjugate(amplitudes)

    return (
        np.real(scale * conjugated_amplitudes * derivatives_s),
        np.real(scale * conjugated_amplitudes * derivatives_t),
        np.real(scale * conjugated_amplitudes * derivatives_p),
    )


def projected_fisher_information_per_mode(
    probability: npt.NDArray[np.float64],
    derivatives_s: npt.NDArray[np.float64],
    derivatives_t: npt.NDArray[np.float64],
    derivatives_p: npt.NDArray[np.float64],
    coordinate_jacobian: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Project per-mode FI from source coordinates to the parameters of interest."""
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

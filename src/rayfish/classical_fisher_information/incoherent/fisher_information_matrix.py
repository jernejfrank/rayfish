"""Calculating the classical Fisher information matrix for incoherent point sources.

This module calculates the classical Fisher information matrix where parameters
correspond to the 3D coordinates of the point sources. This is a starting point from
which we then transform the matrix to the parameters of interest.

The length scale for the position of the particles is normalized w.r.t. the Rayleigh
length [zr] and wavevector [k] as follows:
x,y <-> sqrt[k/zr] x,y
z <-> z / zr

The model assumes incoherent point sources distributed in 3D space and a Gaussian beam
in the image plane as the point spread function. With the above relative length scale
this looks like:
ψ(x-dx, y-dy, z-dz) =
sqrt[1/pi] (i / (z-dz+i))
exp[-i ((x-dx)^2 + (y-dy)^2) / (2 (z-dz+i))] exp[-i k zr (z-dz)]

The calculates are for arbitrary measurement basis in x,y, and z dimensions with the
assumption that
"""

import numpy as np
import numpy.typing as npt
from beartype.typing import Dict, List
from hamilton.function_modifiers import config

from rayfish.classical_fisher_information.incoherent import amplitude_measurement_model

# Public API
__all__ = (
    "probability__1D",
    "fisher_information_matrix_for_entire_distribution",
    "covariance_matrix_for_entire_distribution",
)


@config.when_not(dimension=2, imperfect_mode_sorting=True)
def probability__1D(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.float64]:
    """Probability of detecting a photon in a particular measurement mode.

    We assume measurement is 1D and w.l.o.g. in x-dimension. We calculate the
    probability according to

    prob_i = 1/N ∑_j^N |<phi_i|psi(x+s_j, y+t_j,p_j)>|^2

    for each measurement mode <psi_i|.

    Args:
        coordinates: (M, 3N) array. The position of individual Gaussians. N is the
            number of point sources in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        number_of_point_source_distributions: how many different point source
            distributions we are computing for
        measurement_basis: the measurement modes encoded as polynomial coefficients.
        wave_vector: phase parameter carried by the amplitude-level overlap model.
        rayleigh_length: length scale carried by the amplitude-level overlap model.

    Returns:
        A (M,B) array of complex floats, where B is the number of measurement modes and
        M is the number of different distributions of the N point sources.
    """
    amplitudes = amplitude_measurement_model.mode_amplitudes(
        coordinates=coordinates,
        number_of_point_sources=number_of_point_sources,
        measurement_basis=measurement_basis,
        wave_vector=wave_vector,
        rayleigh_length=rayleigh_length,
    )
    return amplitude_measurement_model.probabilities_from_amplitudes(
        amplitudes=amplitudes, number_of_point_sources=number_of_point_sources
    )


def fisher_information_matrix_for_entire_distribution(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    probability: npt.NDArray[np.float64],
    wave_vector: float,
    rayleigh_length: float,
    coordinate_jacobian: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates covariance matrix for a measurement basis.

    prob_i = 1/N ∑_j^N |<phi_i|psi(x+s_j, y+t_j,p_j)>|^2

    for each measurement mode <psi_i|.

    Args:
        coordinates: (M, 3N) array. The position of individual Gaussians. N is the
            number of point sources in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        number_of_point_source_distributions: how many different point source
            distributions we are computing for
        measurement_basis: polynomial coefficients of the measurement basis.
        probability: the probability of measuring a photon in the specific mode.
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: It is the distance along the propagation
            direction of a beam from the waist to the place where the area of
            1the cross section is doubled
        coordinate_jacobian: transformation from 3N parameters to the parameters of
            interest.

    Returns:
        A (M,B,*,*) array that calculates the covariance matrix for each point source
        distribution M and measurement mode B. We apply the jacobian transformation at
        the end to get the covariance matrix for parameters of interest.
    """
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
    probability_derivatives = (
        amplitude_measurement_model.probability_derivatives_from_amplitudes(
            amplitudes=amplitudes,
            derivatives_s=derivatives_s,
            derivatives_t=derivatives_t,
            derivatives_p=derivatives_p,
            number_of_point_sources=number_of_point_sources,
        )
    )

    return amplitude_measurement_model.projected_fisher_information_per_mode(
        probability=probability,
        derivatives_s=probability_derivatives[0],
        derivatives_t=probability_derivatives[1],
        derivatives_p=probability_derivatives[2],
        coordinate_jacobian=coordinate_jacobian,
    )


def covariance_matrix_for_entire_distribution(
    fisher_information_matrix_for_entire_distribution: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The covariance matrix calculated as the inverse of the fisher information."""
    combined_fi = np.sum(fisher_information_matrix_for_entire_distribution, axis=1)
    cov_matrices = np.zeros_like(combined_fi, dtype=np.float64)

    # only invert non-singular matrices
    conds = np.linalg.cond(combined_fi)
    mask = conds < 1e16
    inv_subset = np.linalg.inv(combined_fi[mask])

    cov_matrices[mask] = inv_subset

    # set singular matrices to inf
    cov_matrices[~mask] = np.inf

    return cov_matrices

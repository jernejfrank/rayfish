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

from rayfish import gaussian_overlaps

# Public API
__all__ = (
    "probability__1D",
    "fisher_information_matrix_for_entire_distribution",
    "covariance_matrix_for_entire_distribution",
)


@config.when_not(dimension=2)
def probability__1D(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
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

    Returns:
        A (M,B) array of complex floats, where B is the number of measurement modes and
        M is the number of different distributions of the N point sources.
    """
    n_measurement_basis = len(measurement_basis)
    prob = np.zeros((number_of_point_source_distributions, n_measurement_basis))

    for i, mode in enumerate(measurement_basis):
        for point_source in range(number_of_point_sources):
            p_ = np.real(
                gaussian_overlaps.probability_measurement_mode_point_source_overlap(
                    coordinates[:, point_source],
                    coordinates[:, number_of_point_sources + point_source],
                    coordinates[:, 2 * number_of_point_sources + point_source],
                    mode,
                )
            )

            prob[:, i] += p_

    return prob / number_of_point_sources


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
    n_measurement_basis = len(measurement_basis)
    fisher_information = np.zeros((
        number_of_point_source_distributions,
        n_measurement_basis,
        3 * number_of_point_sources,
        3 * number_of_point_sources,
    ))

    for b, mode in enumerate(measurement_basis):
        prob = probability[:, b]
        # Here to avoid division by 0, we know if P=0 then dP=0
        mask = prob > 1e-16
        if not np.any(mask):
            continue
        for i in range(number_of_point_sources):
            first_derivative_s = (
                np.real(
                    gaussian_overlaps.derivate_s_probability_measurement_mode_point_source_overlap(
                        s=coordinates[:, i],
                        t=coordinates[:, number_of_point_sources + i],
                        p=coordinates[:, 2 * number_of_point_sources + i],
                        k=wave_vector,
                        zr=rayleigh_length,
                        mode_polynomial_coefficients=mode,
                    )
                )
                / number_of_point_sources
            )
            first_derivative_t = (
                np.real(
                    gaussian_overlaps.derivate_t_probability_measurement_mode_point_source_overlap(
                        s=coordinates[:, i],
                        t=coordinates[:, number_of_point_sources + i],
                        p=coordinates[:, 2 * number_of_point_sources + i],
                        k=wave_vector,
                        zr=rayleigh_length,
                        mode_polynomial_coefficients=mode,
                    )
                )
                / number_of_point_sources
            )
            first_derivative_p = (
                np.real(
                    gaussian_overlaps.derivate_p_probability_measurement_mode_point_source_overlap(
                        s=coordinates[:, i],
                        t=coordinates[:, number_of_point_sources + i],
                        p=coordinates[:, 2 * number_of_point_sources + i],
                        k=wave_vector,
                        zr=rayleigh_length,
                        mode_polynomial_coefficients=mode,
                    )
                )
                / number_of_point_sources
            )
            for j in range(number_of_point_sources):
                second_derivative_s = (
                    np.real(
                        gaussian_overlaps.derivate_s_probability_measurement_mode_point_source_overlap(
                            s=coordinates[:, j],
                            t=coordinates[:, number_of_point_sources + j],
                            p=coordinates[:, 2 * number_of_point_sources + j],
                            k=wave_vector,
                            zr=rayleigh_length,
                            mode_polynomial_coefficients=mode,
                        )
                    )
                    / number_of_point_sources
                )
                second_derivative_t = (
                    np.real(
                        gaussian_overlaps.derivate_t_probability_measurement_mode_point_source_overlap(
                            s=coordinates[:, j],
                            t=coordinates[:, number_of_point_sources + j],
                            p=coordinates[:, 2 * number_of_point_sources + j],
                            k=wave_vector,
                            zr=rayleigh_length,
                            mode_polynomial_coefficients=mode,
                        )
                    )
                    / number_of_point_sources
                )
                second_derivative_p = (
                    np.real(
                        gaussian_overlaps.derivate_p_probability_measurement_mode_point_source_overlap(
                            s=coordinates[:, j],
                            t=coordinates[:, number_of_point_sources + j],
                            p=coordinates[:, 2 * number_of_point_sources + j],
                            k=wave_vector,
                            zr=rayleigh_length,
                            mode_polynomial_coefficients=mode,
                        )
                    )
                    / number_of_point_sources
                )

                # ss
                fisher_information[:, b, i, j] = (
                    first_derivative_s[mask] * second_derivative_s[mask]
                ) / prob[mask]
                # tt
                fisher_information[
                    :, b, number_of_point_sources + i, number_of_point_sources + j
                ] = (first_derivative_t[mask] * second_derivative_t[mask]) / prob[mask]
                # pp
                fisher_information[
                    :,
                    b,
                    2 * number_of_point_sources + i,
                    2 * number_of_point_sources + j,
                ] = (first_derivative_p[mask] * second_derivative_p[mask]) / prob[mask]

                # ts
                fisher_information[:, b, number_of_point_sources + i, j] = (
                    first_derivative_t[mask] * second_derivative_s[mask]
                ) / prob[mask]
                # st
                fisher_information[:, b, i, number_of_point_sources + j] = (
                    first_derivative_s[mask] * second_derivative_t[mask]
                ) / prob[mask]

                # ps
                fisher_information[:, b, 2 * number_of_point_sources + i, j] = (
                    first_derivative_p[mask] * second_derivative_s[mask]
                ) / prob[mask]
                # sp
                fisher_information[:, b, i, 2 * number_of_point_sources + j] = (
                    first_derivative_s[mask] * second_derivative_p[mask]
                ) / prob[mask]

                # tp
                fisher_information[
                    :, b, number_of_point_sources + i, 2 * number_of_point_sources + j
                ] = (first_derivative_t[mask] * second_derivative_p[mask]) / prob[mask]
                # pt
                fisher_information[
                    :, b, 2 * number_of_point_sources + i, number_of_point_sources + j
                ] = (first_derivative_p[mask] * second_derivative_t[mask]) / prob[mask]

    fi_matrices = []
    for _ in range(number_of_point_source_distributions):
        cov_per_mode = []
        for __ in range(n_measurement_basis):
            temp = np.matmul(
                coordinate_jacobian,
                np.matmul(
                    fisher_information[_, __, :, :], np.transpose(coordinate_jacobian)
                ),
            )
            cov_per_mode.append(temp)
        fi_matrices.append(np.stack(cov_per_mode))

    return np.stack(fi_matrices)


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

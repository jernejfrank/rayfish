"""Classical Fisher information for incoherent point sources with a remainder channel.

This module calculates per-channel classical Fisher information where parameters
correspond to the 3D coordinates of the point sources. In addition to the explicitly
resolved measurement modes, it includes one extra channel that buckets the unresolved
probability mass, i.e. ``1 - sum_b P_b`` over the supplied mode bank.

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

The calculations are for an arbitrary measurement basis in the transverse dimensions.
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
    """Mode probabilities for an incoherent point-source distribution.

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
        measurement_basis: the resolved measurement modes encoded as polynomial
            coefficients.

    Returns:
        A ``(M, B)`` array of mode probabilities, where ``B`` is the number of resolved
        measurement modes and ``M`` is the number of point-source distributions.
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


@config.when_not(dimension=2)
def derivatives_probability__1D(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.float64]:
    """Mode-probability derivatives for an incoherent point-source distribution.

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
        measurement_basis: the resolved measurement modes encoded as polynomial
            coefficients.
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: distance along the propagation direction of a beam from the
            waist to the place where the area of the cross section is doubled

    Returns:
        A ``(M, B, 3N)`` array, where entry ``[:, b, a]`` is the derivative of the
        probability of resolved mode ``b`` with respect to source-coordinate parameter
        ``a``.
    """
    n_measurement_basis = len(measurement_basis)
    dprob = np.zeros((
        number_of_point_source_distributions,
        n_measurement_basis,
        3 * number_of_point_sources,
    ))

    for i, mode in enumerate(measurement_basis):
        for point_source in range(number_of_point_sources):
            dprob[:, i, point_source] = np.real(
                gaussian_overlaps.derivate_s_probability_measurement_mode_point_source_overlap(
                    s=coordinates[:, point_source],
                    t=coordinates[:, number_of_point_sources + point_source],
                    p=coordinates[:, 2 * number_of_point_sources + point_source],
                    k=wave_vector,
                    zr=rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            )
            dprob[:, i, number_of_point_sources + point_source] = np.real(
                gaussian_overlaps.derivate_t_probability_measurement_mode_point_source_overlap(
                    s=coordinates[:, point_source],
                    t=coordinates[:, number_of_point_sources + point_source],
                    p=coordinates[:, 2 * number_of_point_sources + point_source],
                    k=wave_vector,
                    zr=rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            )
            dprob[:, i, 2 * number_of_point_sources + point_source] = np.real(
                gaussian_overlaps.derivate_p_probability_measurement_mode_point_source_overlap(
                    s=coordinates[:, point_source],
                    t=coordinates[:, number_of_point_sources + point_source],
                    p=coordinates[:, 2 * number_of_point_sources + point_source],
                    k=wave_vector,
                    zr=rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            )

    return dprob / number_of_point_sources


def fisher_information_matrix_for_entire_distribution(
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    probability: npt.NDArray[np.float64],
    derivatives_probability: npt.NDArray[np.float64],
    coordinate_jacobian: npt.NDArray[np.float64],
    poisson_rate_tau: float = 1.0,
    background_noise: tuple[float, ...] = (0.0,),
) -> npt.NDArray[np.float64]:
    """Per-channel Fisher information including an unresolved remainder channel.

    prob_i = 1/N ∑_j^N |<phi_i|psi(x+s_j, y+t_j,p_j)>|^2

    for each measurement mode <psi_i|.

    Args:
        number_of_point_sources: how many point sources are used to model the object.
        number_of_point_source_distributions: how many different point source
            distributions are being computed.
        measurement_basis: polynomial coefficients of the measurement basis.
        probability: resolved-mode probabilities with shape ``(M, B)``.
        derivatives_probability: derivatives of each mode probability with respect to
            the ``3N`` source coordinates, with shape ``(M, B, 3N)``.
        coordinate_jacobian: transformation from 3N parameters to the parameters of
            interest.
        poisson_rate_tau: total photon budget ``tau`` multiplying the channel means
            ``lambda_j = tau P_j + b_j``. Scales the Fisher information overall.
        background_noise: theta-independent additive count floor ``b_j`` per channel
            (dark counts + background light), entering the denominator as
            ``tau P_j + b_j``. A single-element tuple is broadcast to all measurement
            channels plus the remainder bucket; otherwise its length must equal the
            number of measurement modes plus one.

    Returns:
        A ``(M, B + 1, K, K)`` array of per-channel Fisher information matrices after
        Jacobian transformation to the parameters of interest. The final channel
        ``B`` is the remainder bucket with probability ``1 - sum_b P_b``.
    """
    if len(background_noise) == 1:
        noise_factor = background_noise[0]
        background_noise = tuple(
            noise_factor for _ in range(len(measurement_basis) + 1)
        )

    if (
        len(background_noise)
        != len(measurement_basis) + 1  # accounting for remainder channel
    ):
        raise ValueError(
            "Mismatch, expecting background noise to be the same length as "
            "measurement basis."
        )

    n_measurement_basis = len(measurement_basis)
    fisher_information = np.zeros((
        number_of_point_source_distributions,
        n_measurement_basis + 1,  # Last channel counting as remainder
        3 * number_of_point_sources,
        3 * number_of_point_sources,
    ))

    for i in range(number_of_point_sources):
        for j in range(number_of_point_sources):
            der_s1 = np.zeros(number_of_point_source_distributions)
            der_s2 = np.zeros(number_of_point_source_distributions)
            der_t1 = np.zeros(number_of_point_source_distributions)
            der_t2 = np.zeros(number_of_point_source_distributions)
            der_p1 = np.zeros(number_of_point_source_distributions)
            der_p2 = np.zeros(number_of_point_source_distributions)
            prob_sum = np.zeros(number_of_point_source_distributions)
            for b in range(len(measurement_basis)):
                prob = probability[:, b]
                # Here to avoid division by 0, we know if P=0 then dP=0
                mask = prob > 1e-16
                if not np.any(mask):
                    continue

                der_s1[mask] += derivatives_probability[mask, b, i]
                der_s2[mask] += derivatives_probability[mask, b, j]
                der_t1[mask] += derivatives_probability[
                    mask, b, number_of_point_sources + i
                ]
                der_t2[mask] += derivatives_probability[
                    mask, b, number_of_point_sources + j
                ]
                der_p1[mask] += derivatives_probability[
                    mask, b, 2 * number_of_point_sources + i
                ]
                der_p2[mask] += derivatives_probability[
                    mask, b, 2 * number_of_point_sources + j
                ]
                prob_sum[mask] += prob[mask]

                background_beta = background_noise[b] / poisson_rate_tau

                # ss
                fisher_information[mask, b, i, j] = poisson_rate_tau * (
                    derivatives_probability[mask, b, i]
                    * derivatives_probability[mask, b, j]
                    / (prob[mask] + background_beta)
                )

                # tt
                fisher_information[
                    mask, b, number_of_point_sources + i, number_of_point_sources + j
                ] = poisson_rate_tau * (
                    derivatives_probability[mask, b, number_of_point_sources + i]
                    * derivatives_probability[mask, b, number_of_point_sources + j]
                    / (prob[mask] + background_beta)
                )

                # pp
                fisher_information[
                    mask,
                    b,
                    2 * number_of_point_sources + i,
                    2 * number_of_point_sources + j,
                ] = poisson_rate_tau * (
                    derivatives_probability[mask, b, 2 * number_of_point_sources + i]
                    * derivatives_probability[mask, b, 2 * number_of_point_sources + j]
                    / (prob[mask] + background_beta)
                )

                # ts
                fisher_information[mask, b, number_of_point_sources + i, j] = (
                    poisson_rate_tau
                    * (
                        derivatives_probability[mask, b, number_of_point_sources + i]
                        * derivatives_probability[mask, b, j]
                        / (prob[mask] + background_beta)
                    )
                )
                # st
                fisher_information[mask, b, i, j + number_of_point_sources] = (
                    poisson_rate_tau
                    * (
                        derivatives_probability[mask, b, i]
                        * derivatives_probability[mask, b, number_of_point_sources + j]
                        / (prob[mask] + background_beta)
                    )
                )
                # ps
                fisher_information[mask, b, 2 * number_of_point_sources + i, j] = (
                    poisson_rate_tau
                    * (
                        derivatives_probability[
                            mask, b, 2 * number_of_point_sources + i
                        ]
                        * derivatives_probability[mask, b, j]
                        / (prob[mask] + background_beta)
                    )
                )
                # sp
                fisher_information[mask, b, i, j + 2 * number_of_point_sources] = (
                    poisson_rate_tau
                    * (
                        derivatives_probability[mask, b, i]
                        * derivatives_probability[
                            mask, b, 2 * number_of_point_sources + j
                        ]
                        / (prob[mask] + background_beta)
                    )
                )
                # tp
                fisher_information[
                    mask,
                    b,
                    number_of_point_sources + i,
                    j + 2 * number_of_point_sources,
                ] = poisson_rate_tau * (
                    derivatives_probability[mask, b, number_of_point_sources + i]
                    * derivatives_probability[mask, b, 2 * number_of_point_sources + j]
                    / (prob[mask] + background_beta)
                )
                # pt
                fisher_information[
                    mask,
                    b,
                    2 * number_of_point_sources + i,
                    j + number_of_point_sources,
                ] = poisson_rate_tau * (
                    derivatives_probability[mask, b, 2 * number_of_point_sources + i]
                    * derivatives_probability[mask, b, number_of_point_sources + j]
                    / (prob[mask] + background_beta)
                )

            remainder_background_beta = (
                background_noise[len(measurement_basis)] / poisson_rate_tau
            )
            remainder_prob = 1 - prob_sum
            remainder_mask = remainder_prob > 1e-16
            if not np.any(remainder_mask):
                continue

            # ss
            fisher_information[remainder_mask, n_measurement_basis, i, j] = (
                poisson_rate_tau
                * (
                    der_s1[remainder_mask]
                    * der_s2[remainder_mask]
                    / (remainder_prob[remainder_mask] + remainder_background_beta)
                )
            )

            # tt
            fisher_information[
                remainder_mask,
                n_measurement_basis,
                number_of_point_sources + i,
                number_of_point_sources + j,
            ] = poisson_rate_tau * (
                der_t1[remainder_mask]
                * der_t2[remainder_mask]
                / (remainder_prob[remainder_mask] + remainder_background_beta)
            )

            # pp
            fisher_information[
                remainder_mask,
                n_measurement_basis,
                2 * number_of_point_sources + i,
                2 * number_of_point_sources + j,
            ] = poisson_rate_tau * (
                der_p1[remainder_mask]
                * der_p2[remainder_mask]
                / (remainder_prob[remainder_mask] + remainder_background_beta)
            )

            # ts
            fisher_information[
                remainder_mask, n_measurement_basis, number_of_point_sources + i, j
            ] = poisson_rate_tau * (
                der_t1[remainder_mask]
                * der_s2[remainder_mask]
                / (remainder_prob[remainder_mask] + remainder_background_beta)
            )
            # st
            fisher_information[
                remainder_mask, n_measurement_basis, i, j + number_of_point_sources
            ] = poisson_rate_tau * (
                der_s1[remainder_mask]
                * der_t2[remainder_mask]
                / (remainder_prob[remainder_mask] + remainder_background_beta)
            )
            # ps
            fisher_information[
                remainder_mask, n_measurement_basis, 2 * number_of_point_sources + i, j
            ] = poisson_rate_tau * (
                der_p1[remainder_mask]
                * der_s2[remainder_mask]
                / (remainder_prob[remainder_mask] + remainder_background_beta)
            )
            # sp
            fisher_information[
                remainder_mask, n_measurement_basis, i, j + 2 * number_of_point_sources
            ] = poisson_rate_tau * (
                der_s1[remainder_mask]
                * der_p2[remainder_mask]
                / (remainder_prob[remainder_mask] + remainder_background_beta)
            )
            # tp
            fisher_information[
                remainder_mask,
                n_measurement_basis,
                number_of_point_sources + i,
                j + 2 * number_of_point_sources,
            ] = poisson_rate_tau * (
                der_t1[remainder_mask]
                * der_p2[remainder_mask]
                / (remainder_prob[remainder_mask] + remainder_background_beta)
            )
            # pt
            fisher_information[
                remainder_mask,
                n_measurement_basis,
                2 * number_of_point_sources + i,
                j + number_of_point_sources,
            ] = poisson_rate_tau * (
                der_p1[remainder_mask]
                * der_t2[remainder_mask]
                / (remainder_prob[remainder_mask] + remainder_background_beta)
            )

    fi_matrices = []
    for _ in range(number_of_point_source_distributions):
        cov_per_mode = []
        for __ in range(n_measurement_basis + 1):
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

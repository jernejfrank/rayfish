"""Classical Fisher information for incoherent point sources with a remainder channel.

This module calculates per-channel classical Fisher information where parameters
correspond to the 3D coordinates of the point sources. In addition to the explicitly
resolved measurement modes, it includes one extra channel that buckets the unresolved
probability mass, the remainder channel P_rem(θ) = 1 - Σ_{(j,l)} P_{j,l}(θ), completing
the resolved modes to a valid POVM. The Fisher information is additive over outcomes, so
the remainder adds a single term to the mode-sorting FIM,

    J_{a,b} = τ [ Σ_{(j,l)} (∂_a P_{j,l} ∂_b P_{j,l}) / P_{j,l}
                  + (∂_a P_rem ∂_b P_rem) / P_rem ],

with ∂_a P_rem = -Σ_{(j,l)} ∂_a P_{j,l}. A detector noise floor b_j (dark counts,
background light) generalizes this to J_{a,b} = τ^2 Σ_j ∂_a P_j ∂_b P_j / (τ P_j + b_j).

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
from beartype.typing import Dict, List, Tuple

from rayfish import gaussian_overlaps

# Public API
__all__ = (
    "covariance_matrix_for_entire_distribution",
    "covariance_matrix_for_entire_distribution_nuisance_intensity",
    "covariance_matrix_for_entire_distribution_spatial",
    "derivatives_probability",
    "fisher_information_matrix_for_entire_distribution",
    "probability",
)


def derivatives_probability(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    point_source_intensities: list[float],
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.float64]:
    """Mode-probability derivatives for an incoherent point-source distribution.

    We assume measurement is 1D and w.l.o.g. in x-dimension. Returns the derivatives
    ∂_a P_i of the measurement-channel probability

        P_i(θ) = Σ_v I_v |<φ_i|ψ_v(x+s_v, y+t_v, p_v)>|^2,

    w.r.t. the 3N source coordinates, used to assemble the per-channel Fisher
    information.

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
        point_source_intensities: intensity distribution for point sources.
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: distance along the propagation direction of a beam from the
            waist to the place where the area of the cross section is doubled

    Returns:
        A ``(M, B, 4N)`` array, where entry ``[:, b, a]`` is the derivative of the
        probability of resolved mode ``b`` with respect to parameter ``a``. The 4N
        parameters are ordered as the 3N source coordinates [s, t, p] followed by the
        N intensity derivatives; the intensity block equals the raw single-source
        overlap probabilities (unweighted by intensity).
    """
    n_measurement_basis = len(measurement_basis)
    dprob = np.zeros((
        number_of_point_source_distributions,
        n_measurement_basis,
        4 * number_of_point_sources,
    ))

    for i, mode in enumerate(measurement_basis):
        for point_source in range(number_of_point_sources):
            dprob[:, i, point_source] = point_source_intensities[
                point_source
            ] * np.real(
                gaussian_overlaps.derivate_s_probability_measurement_mode_point_source_overlap(
                    s=coordinates[:, point_source],
                    t=coordinates[:, number_of_point_sources + point_source],
                    p=coordinates[:, 2 * number_of_point_sources + point_source],
                    k=wave_vector,
                    zr=rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            )
            dprob[:, i, number_of_point_sources + point_source] = (
                point_source_intensities[point_source]
                * np.real(
                    gaussian_overlaps.derivate_t_probability_measurement_mode_point_source_overlap(
                        s=coordinates[:, point_source],
                        t=coordinates[:, number_of_point_sources + point_source],
                        p=coordinates[:, 2 * number_of_point_sources + point_source],
                        k=wave_vector,
                        zr=rayleigh_length,
                        mode_polynomial_coefficients=mode,
                    )
                )
            )
            dprob[:, i, 2 * number_of_point_sources + point_source] = (
                point_source_intensities[point_source]
                * np.real(
                    gaussian_overlaps.derivate_p_probability_measurement_mode_point_source_overlap(
                        s=coordinates[:, point_source],
                        t=coordinates[:, number_of_point_sources + point_source],
                        p=coordinates[:, 2 * number_of_point_sources + point_source],
                        k=wave_vector,
                        zr=rayleigh_length,
                        mode_polynomial_coefficients=mode,
                    )
                )
            )

            dprob[:, i, 3 * number_of_point_sources + point_source] = np.real(
                gaussian_overlaps.probability_measurement_mode_point_source_overlap(
                    s=coordinates[:, point_source],
                    t=coordinates[:, number_of_point_sources + point_source],
                    p=coordinates[:, 2 * number_of_point_sources + point_source],
                    k=wave_vector,
                    zr=rayleigh_length,
                    mode_polynomial_coefficients=mode,
                )
            )

    return dprob


def probability(
    number_of_point_sources: int,
    derivatives_probability: npt.NDArray[np.float64],
    point_source_intensities: list[float],
) -> npt.NDArray[np.float64]:
    """Mode probabilities for an incoherent point-source distribution.

    We assume measurement is 1D and w.l.o.g. in x-dimension. Implements the
    measurement-channel probability

        P_i(θ) = Σ_v I_v |<φ_i|ψ_v(x+s_v, y+t_v, p_v)>|^2,

    the intensity-weighted incoherent sum of single-source mode overlaps for each
    measurement mode <φ_i| (equal sources I_v = 1/V recover the 1/V weighting).

    Args:
        number_of_point_sources: how many point sources we use to model the object
        derivatives_probability: raw probabilities without intensity weights, which
            double as intensity derivatives.
        point_source_intensities: intensity distribution for point sources.

    Returns:
        A ``(M, B)`` array of mode probabilities, where ``B`` is the number of resolved
        measurement modes and ``M`` is the number of point-source distributions.
    """
    probabilities = np.zeros((
        derivatives_probability.shape[0],
        derivatives_probability.shape[1],
    ))
    intensities = np.array(point_source_intensities)
    raw_overlap_probs = derivatives_probability[:, :, 3 * number_of_point_sources :]

    measurement_basis = derivatives_probability.shape[1]
    for i in range(measurement_basis):
        overlaps = raw_overlap_probs[:, i, :]
        weighted_prob = intensities * overlaps
        probabilities[:, i] = np.sum(weighted_prob, axis=-1)

    return probabilities


def fisher_information_matrix_for_entire_distribution(  # noqa: C901
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    measurement_basis: List[Dict[str, npt.NDArray[np.float64]]],
    probability: npt.NDArray[np.float64],
    derivatives_probability: npt.NDArray[np.float64],
    coordinate_jacobian: npt.NDArray[np.float64],
    poisson_rate_tau: float = 1.0,
    background_noise: Tuple = (0.0,),
) -> npt.NDArray[np.float64]:
    """Per-channel Fisher information including an unresolved remainder channel.

    Implements the noise-floor generalization of the mode-sorting FIM with the additive
    remainder term,

        J_{a,b} = τ^2 Σ_j (1/(τ P_j + b_j)) ∂_a P_j ∂_b P_j,

    where the channel index j runs over the resolved modes plus the remainder bucket
    P_rem = 1 - Σ_b P_b (with ∂_a P_rem = -Σ_b ∂_a P_b). Mode probabilities are the
    incoherent single-source sums P_b = (1/V) Σ_v |<φ_b|ψ_v>|^2.

    Args:
        number_of_point_sources: how many point sources are used to model the object.
        number_of_point_source_distributions: how many different point source
            distributions are being computed.
        measurement_basis: polynomial coefficients of the measurement basis.
        probability: resolved-mode probabilities with shape ``(M, B)``.
        derivatives_probability: raw probabilities without intensity weights, which
            double as intensity derivatives.
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
        4 * number_of_point_sources,
        4 * number_of_point_sources,
    ))

    for i in range(number_of_point_sources):
        for j in range(number_of_point_sources):
            der_remainder = [
                np.zeros(number_of_point_source_distributions) for _ in range(2 * 4)
            ]
            prob_sum = np.zeros(number_of_point_source_distributions)
            for b in range(len(measurement_basis)):
                prob = probability[:, b]
                # Here to avoid division by 0, we know if P=0 then dP=0
                mask = prob > 1e-16
                if not np.any(mask):
                    continue

                for param in range(4):
                    param_offset = param * number_of_point_sources
                    for second_param in range(2):
                        _idx = i if second_param == 0 else j
                        der_remainder[2 * param + second_param][mask] += (
                            derivatives_probability[mask, b, param_offset + _idx]
                        )

                prob_sum[mask] += prob[mask]

                background_beta = background_noise[b] / poisson_rate_tau

                for param1 in range(4):
                    for param2 in range(4):
                        param1_offset = param1 * number_of_point_sources
                        param2_offset = param2 * number_of_point_sources

                        fisher_information[
                            mask, b, param1_offset + i, param2_offset + j
                        ] = poisson_rate_tau * (
                            derivatives_probability[mask, b, param1_offset + i]
                            * derivatives_probability[mask, b, param2_offset + j]
                            / (prob[mask] + background_beta)
                        )

            remainder_background_beta = (
                background_noise[len(measurement_basis)] / poisson_rate_tau
            )
            remainder_prob = 1 - prob_sum
            remainder_mask = remainder_prob > 1e-16
            if not np.any(remainder_mask):
                continue

            for param1 in range(4):
                for param2 in range(4):
                    param1_offset = param1 * number_of_point_sources
                    param2_offset = param2 * number_of_point_sources

                    der1_idx = param1 * 2
                    der2_idx = param2 * 2 + 1

                    fisher_information[
                        remainder_mask,
                        n_measurement_basis,
                        param1_offset + i,
                        param2_offset + j,
                    ] = poisson_rate_tau * (
                        der_remainder[der1_idx][remainder_mask]
                        * der_remainder[der2_idx][remainder_mask]
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


def covariance_matrix_for_entire_distribution_nuisance_intensity(
    number_of_spatial_params: int,
    fisher_information_matrix_for_entire_distribution: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The covariance matrix calculated as the inverse of the fisher information."""
    combined_fi = np.sum(fisher_information_matrix_for_entire_distribution, axis=1)

    intensity_fi = combined_fi[:, number_of_spatial_params:, number_of_spatial_params:]
    # only invert non-singular matrices
    conds = np.linalg.cond(intensity_fi)
    mask = conds < 1e16
    inv_intensity_fi = np.zeros_like(intensity_fi)
    inv_subset = np.linalg.inv(intensity_fi[mask])
    inv_intensity_fi[mask] = inv_subset

    old_fi = combined_fi[:, :number_of_spatial_params, :number_of_spatial_params]

    left_complement = combined_fi[
        :, :number_of_spatial_params, number_of_spatial_params:
    ]

    right_complement = combined_fi[
        :, number_of_spatial_params:, :number_of_spatial_params
    ]

    fi_marginal = old_fi - np.matmul(
        left_complement, np.matmul(inv_intensity_fi, right_complement)
    )

    cov_matrices = np.zeros_like(fi_marginal, dtype=np.float64)
    # only invert non-singular matrices
    conds = np.linalg.cond(fi_marginal)
    mask = conds < 1e16
    inv_subset = np.linalg.inv(fi_marginal[mask])

    cov_matrices[mask] = inv_subset

    # set singular matrices to inf
    cov_matrices[~mask] = np.inf

    return cov_matrices


def covariance_matrix_for_entire_distribution_spatial(
    fisher_information_matrix_for_entire_distribution: npt.NDArray[np.float64],
    number_of_spatial_params: int,
) -> npt.NDArray[np.float64]:
    """The covariance matrix calculated as the inverse of the fisher information."""
    combined_fi = np.sum(fisher_information_matrix_for_entire_distribution, axis=1)
    fi = combined_fi[:, :number_of_spatial_params, :number_of_spatial_params]

    cov_matrices = np.zeros_like(fi, dtype=np.float64)

    fim_real = np.real(fi)

    # only invert non-singular matrices
    conds = np.linalg.cond(fim_real)
    mask = conds < 1e16
    inv_subset = np.linalg.inv(fim_real[mask])

    cov_matrices[mask] = inv_subset

    # set singular matrices to inf
    cov_matrices[~mask] = np.inf

    return cov_matrices

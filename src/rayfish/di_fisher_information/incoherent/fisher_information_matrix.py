"""Benchmarking comparison to direct imaging of incoherent point sources.

Calculating the classical Fisher information matrix for incoherent point sources using
direct imaging (DI) which is modeled by a gaussian positioned at the origin of the
coordinate system: ψ(x,y,0).

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
exp[-i ((x-dx)^2 + (y-dy)^2) / (2 (z-dz+i))] exp[-i k zr (z-dz)].
"""

import numpy as np
import numpy.typing as npt
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
) -> npt.NDArray[np.float64]:
    """Probability of detecting a photon via direct imaging.

    We assume measurement is 1D and w.l.o.g. in x-dimension. We calculate the
    probability according to

    prob_i = 1/N ∑_j^N |<psi(x, y, 0)|psi(x+s_j, y+t_j, p_j)>|^2


    Args:
        coordinates: (M, 3N) array. The position of individual Gaussians. N is the
            number of point sources in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        number_of_point_source_distributions: how many different point source
            distributions we are computing for

    Returns:
        A (M,) array of complex floats, where B is the number of measurement modes and
        M is the number of different distributions of the N point sources.
    """
    prob = np.zeros(number_of_point_source_distributions)

    for point_source in range(number_of_point_sources):
        p_ = np.real(
            gaussian_overlaps.probability_measurement_di_point_source_overlap(
                coordinates[:, point_source],
                coordinates[:, number_of_point_sources + point_source],
                coordinates[:, 2 * number_of_point_sources + point_source],
            )
        )

        prob[:] += p_

    return prob / number_of_point_sources


def fisher_information_matrix_for_entire_distribution(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    probability: npt.NDArray[np.float64],
    coordinate_jacobian: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates fisher information matrix for direct imaging.

    Args:
        coordinates: (M, 3N) array. The position of individual Gaussians. N is the
            number of point sources in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        number_of_point_source_distributions: how many different point source
            distributions we are computing for
        probability: the direct imaging probability.
        coordinate_jacobian: transformation from 3N parameters to the parameters of
            interest.

    Returns:
        A (M,B,*,*) array that calculates the covariance matrix for each point source
        distribution M and measurement mode B. We apply the jacobian transformation at
        the end to get the covariance matrix for parameters of interest.
    """
    fisher_information = np.zeros((
        number_of_point_source_distributions,
        3 * number_of_point_sources,
        3 * number_of_point_sources,
    ))

    for i in range(number_of_point_sources):
        first_derivative_s = (
            np.real(
                gaussian_overlaps.derivative_s_probability_measurement_di_point_source_overlap(
                    s=coordinates[:, i],
                    t=coordinates[:, number_of_point_sources + i],
                    p=coordinates[:, 2 * number_of_point_sources + i],
                )
            )
            / number_of_point_sources
        )
        first_derivative_t = (
            np.real(
                gaussian_overlaps.derivative_t_probability_measurement_di_point_source_overlap(
                    s=coordinates[:, i],
                    t=coordinates[:, number_of_point_sources + i],
                    p=coordinates[:, 2 * number_of_point_sources + i],
                )
            )
            / number_of_point_sources
        )
        first_derivative_p = (
            np.real(
                gaussian_overlaps.derivative_p_probability_measurement_di_point_source_overlap(
                    s=coordinates[:, i],
                    t=coordinates[:, number_of_point_sources + i],
                    p=coordinates[:, 2 * number_of_point_sources + i],
                )
            )
            / number_of_point_sources
        )
        for j in range(number_of_point_sources):
            second_derivative_s = (
                np.real(
                    gaussian_overlaps.derivative_s_probability_measurement_di_point_source_overlap(
                        s=coordinates[:, j],
                        t=coordinates[:, number_of_point_sources + j],
                        p=coordinates[:, 2 * number_of_point_sources + j],
                    )
                )
                / number_of_point_sources
            )
            second_derivative_t = (
                np.real(
                    gaussian_overlaps.derivative_t_probability_measurement_di_point_source_overlap(
                        s=coordinates[:, j],
                        t=coordinates[:, number_of_point_sources + j],
                        p=coordinates[:, 2 * number_of_point_sources + j],
                    )
                )
                / number_of_point_sources
            )
            second_derivative_p = (
                np.real(
                    gaussian_overlaps.derivative_p_probability_measurement_di_point_source_overlap(
                        s=coordinates[:, j],
                        t=coordinates[:, number_of_point_sources + j],
                        p=coordinates[:, 2 * number_of_point_sources + j],
                    )
                )
                / number_of_point_sources
            )

            # ss
            fisher_information[:, i, j] = (
                first_derivative_s * second_derivative_s
            ) / probability
            # tt
            fisher_information[
                :, number_of_point_sources + i, number_of_point_sources + j
            ] = (first_derivative_t * second_derivative_t) / probability
            # pp
            fisher_information[
                :,
                2 * number_of_point_sources + i,
                2 * number_of_point_sources + j,
            ] = (first_derivative_p * second_derivative_p) / probability

            # ts
            fisher_information[:, number_of_point_sources + i, j] = (
                first_derivative_t * second_derivative_s
            ) / probability
            # st
            fisher_information[:, i, number_of_point_sources + j] = (
                first_derivative_s * second_derivative_t
            ) / probability

            # ps
            fisher_information[:, 2 * number_of_point_sources + i, j] = (
                first_derivative_p * second_derivative_s
            ) / probability
            # sp
            fisher_information[:, i, 2 * number_of_point_sources + j] = (
                first_derivative_s * second_derivative_p
            ) / probability

            # tp
            fisher_information[
                :, number_of_point_sources + i, 2 * number_of_point_sources + j
            ] = (first_derivative_t * second_derivative_p) / probability
            # pt
            fisher_information[
                :, 2 * number_of_point_sources + i, number_of_point_sources + j
            ] = (first_derivative_p * second_derivative_t) / probability

    fi_matrices = []
    for _ in range(number_of_point_source_distributions):
        temp = np.matmul(
            coordinate_jacobian,
            np.matmul(fisher_information[_, :, :], np.transpose(coordinate_jacobian)),
        )

        fi_matrices.append(temp)

    return np.stack(fi_matrices)


def covariance_matrix_for_entire_distribution(
    fisher_information_matrix_for_entire_distribution: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The covariance matrix calculated as the inverse of the fisher information."""
    # cov_matrices = np.zeros_like(
    #     fisher_information_matrix_for_entire_distribution, dtype=np.float64
    # )

    # # only invert non-singular matrices
    # conds = np.linalg.cond(fisher_information_matrix_for_entire_distribution)
    # mask = conds < 1e16
    # inv_subset = np.linalg.pinv(
    #     fisher_information_matrix_for_entire_distribution[mask])

    # cov_matrices[mask] = inv_subset

    # # set singular matrices to inf
    # cov_matrices[~mask] = np.inf

    # return cov_matrices
    return np.linalg.pinv(fisher_information_matrix_for_entire_distribution)

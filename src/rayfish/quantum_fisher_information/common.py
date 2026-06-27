"""Shared functions necessary to calculate quantum fisher information."""

import numpy as np
import numpy.typing as npt

from rayfish import gaussian_overlaps

# Public API
__all__ = (
    "delta_submat",
    "gamma_submat",
    "tau_submat",
)


def delta_submat(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.complex128]:
    """Upper left submatrix for non-orthogonal coordinate transformations.

    Creates the submatrix consisting of overlaps between Gaussians based on the function
    gaussian_overlaps.delta(...).

    Args:
        coordinates: (M, 3N) array. The position of individual Gaussians. N is the
            number of point sources in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        number_of_point_source_distributions: how many different point source
            distributions we are computing for
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: It is the distance along the propagation
            direction of a beam from the waist to the place where the area of
            1the cross section is doubled

    Returns:
        A (M,N,N) array of complex floats, where N is the number of point sources and M
        is the number of different distributions of the N point sources. The diagonal
        for each (N,N) matrix is 1 since we assume normalized Gaussians.
    """
    delta_mat = (
        np.zeros((
            number_of_point_source_distributions,
            number_of_point_sources,
            number_of_point_sources,
        ))
        + 0j
    )
    for i in range(number_of_point_sources):
        for j in range(number_of_point_sources):
            delta_mat[:, i, j] = gaussian_overlaps.delta(
                s1=coordinates[:, i],
                s2=coordinates[:, j],
                t1=coordinates[:, number_of_point_sources + i],
                t2=coordinates[:, number_of_point_sources + j],
                p1=coordinates[:, 2 * number_of_point_sources + i],
                p2=coordinates[:, 2 * number_of_point_sources + j],
                k=wave_vector,
                zr=rayleigh_length,
            )
    return delta_mat


def gamma_submat(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.complex128]:
    """Matrix of Gaussians and derivatives for non-orthogonal coordinate transformation.

    Creates the two submatrices consisting of overlaps between Gaussians and
    derivatives of Gaussian based on the functions:
    - gaussian_overlaps.gamma_s(...) for x-dimension,
    - gaussian_overlaps.gamma_t(...) for y-dimension,
    - gaussian_overlaps.gamma_p(...) for z-dimension.

    The two submatrices are embedded into a (4N,4N) nill matrix that has the same
    dimension as the quantum Fisher information matrix.
    The diagonal (N,N) and (3N,3N) blocks are 0.

    Args:
        coordinates: (M, 3N) array. The position of individual Gaussians. N is the
            number of point sources in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        number_of_point_source_distributions: how many different point source
            distributions we are computing for
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: It is the distance along the propagation
            direction of a beam from the waist to the place where the area of
            1the cross section is doubled

    Returns:
        A (M,4N,4N) array of complex floats, where N is the number of point sources
        and M is the number of different distributions of the N point sources.
    """
    gamma_mat = (
        np.zeros((
            number_of_point_source_distributions,
            4 * number_of_point_sources,
            4 * number_of_point_sources,
        ))
        + 0j
    )
    for i in range(number_of_point_sources):
        for j in range(number_of_point_sources):
            gamma_mat[:, i, number_of_point_sources + j] = gaussian_overlaps.gamma_s(
                s1=coordinates[:, i],
                s2=coordinates[:, j],
                t1=coordinates[:, number_of_point_sources + i],
                t2=coordinates[:, number_of_point_sources + j],
                p1=coordinates[:, 2 * number_of_point_sources + i],
                p2=coordinates[:, 2 * number_of_point_sources + j],
                k=wave_vector,
                zr=rayleigh_length,
            )
            gamma_mat[:, i, 2 * number_of_point_sources + j] = (
                gaussian_overlaps.gamma_t(
                    s1=coordinates[:, i],
                    s2=coordinates[:, j],
                    t1=coordinates[:, number_of_point_sources + i],
                    t2=coordinates[:, number_of_point_sources + j],
                    p1=coordinates[:, 2 * number_of_point_sources + i],
                    p2=coordinates[:, 2 * number_of_point_sources + j],
                    k=wave_vector,
                    zr=rayleigh_length,
                )
            )
            gamma_mat[:, i, 3 * number_of_point_sources + j] = (
                gaussian_overlaps.gamma_p(
                    s1=coordinates[:, i],
                    s2=coordinates[:, j],
                    t1=coordinates[:, number_of_point_sources + i],
                    t2=coordinates[:, number_of_point_sources + j],
                    p1=coordinates[:, 2 * number_of_point_sources + i],
                    p2=coordinates[:, 2 * number_of_point_sources + j],
                    k=wave_vector,
                    zr=rayleigh_length,
                )
            )

    gamma_mat += np.conj(np.transpose(gamma_mat, axes=(0, 2, 1)))

    return gamma_mat


def tau_submat(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.complex128]:
    """Lower right submatrix for non-orthogonal coordinate transformations.

    Creates the submatrix consisting of overlaps between Gaussians derivatives based
    on the functions:
    - gaussian_overlaps.tau_ss(...) for x-dimension,
    - gaussian_overlaps.tau_st(...) for x- and y-dimension,
    - gaussian_overlaps.tau_sp(...) for x- and z-dimension,
    - gaussian_overlaps.tau_tt(...) for y-dimension,
    - gaussian_overlaps.tau_tp(...) for y- and z-dimension,
    - gaussian_overlaps.tau_pp(...) for z-dimension.

    Args:
        coordinates: (M, 3N) array. The position of individual Gaussians. N is the
            number of point sources in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        number_of_point_source_distributions: how many different point source
            distributions we are computing for
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: It is the distance along the propagation
            direction of a beam from the waist to the place where the area of
            1the cross section is doubled

    Returns:
       A (M,3N,3N) array of complex floats, where N is the number of point sources
       and M is the number of different distributions of the N point sources.
    """
    tau_mat = (
        np.zeros((
            number_of_point_source_distributions,
            3 * number_of_point_sources,
            3 * number_of_point_sources,
        ))
        + 0j
    )

    for i in range(number_of_point_sources):
        for j in range(number_of_point_sources):
            tau_mat[:, i, j] = gaussian_overlaps.tau_ss(
                s1=coordinates[:, i],
                s2=coordinates[:, j],
                t1=coordinates[:, number_of_point_sources + i],
                t2=coordinates[:, number_of_point_sources + j],
                p1=coordinates[:, 2 * number_of_point_sources + i],
                p2=coordinates[:, 2 * number_of_point_sources + j],
                k=wave_vector,
                zr=rayleigh_length,
            )
            tau_mat[:, i, number_of_point_sources + j] = gaussian_overlaps.tau_st(
                s1=coordinates[:, i],
                s2=coordinates[:, j],
                t1=coordinates[:, number_of_point_sources + i],
                t2=coordinates[:, number_of_point_sources + j],
                p1=coordinates[:, 2 * number_of_point_sources + i],
                p2=coordinates[:, 2 * number_of_point_sources + j],
                k=wave_vector,
                zr=rayleigh_length,
            )
            tau_mat[:, i, 2 * number_of_point_sources + j] = gaussian_overlaps.tau_sp(
                s1=coordinates[:, i],
                s2=coordinates[:, j],
                t1=coordinates[:, number_of_point_sources + i],
                t2=coordinates[:, number_of_point_sources + j],
                p1=coordinates[:, 2 * number_of_point_sources + i],
                p2=coordinates[:, 2 * number_of_point_sources + j],
                k=wave_vector,
                zr=rayleigh_length,
            )
            tau_mat[:, number_of_point_sources + i, j] = gaussian_overlaps.tau_st(
                s1=coordinates[:, i],
                s2=coordinates[:, j],
                t1=coordinates[:, number_of_point_sources + i],
                t2=coordinates[:, number_of_point_sources + j],
                p1=coordinates[:, 2 * number_of_point_sources + i],
                p2=coordinates[:, 2 * number_of_point_sources + j],
                k=wave_vector,
                zr=rayleigh_length,
            )
            tau_mat[:, number_of_point_sources + i, number_of_point_sources + j] = (
                gaussian_overlaps.tau_tt(
                    s1=coordinates[:, i],
                    s2=coordinates[:, j],
                    t1=coordinates[:, number_of_point_sources + i],
                    t2=coordinates[:, number_of_point_sources + j],
                    p1=coordinates[:, 2 * number_of_point_sources + i],
                    p2=coordinates[:, 2 * number_of_point_sources + j],
                    k=wave_vector,
                    zr=rayleigh_length,
                )
            )
            tau_mat[:, number_of_point_sources + i, 2 * number_of_point_sources + j] = (
                gaussian_overlaps.tau_tp(
                    s1=coordinates[:, i],
                    s2=coordinates[:, j],
                    t1=coordinates[:, number_of_point_sources + i],
                    t2=coordinates[:, number_of_point_sources + j],
                    p1=coordinates[:, 2 * number_of_point_sources + i],
                    p2=coordinates[:, 2 * number_of_point_sources + j],
                    k=wave_vector,
                    zr=rayleigh_length,
                )
            )
            tau_mat[:, 2 * number_of_point_sources + i, j] = gaussian_overlaps.tau_sp(
                s1=coordinates[:, i],
                s2=coordinates[:, j],
                t1=coordinates[:, number_of_point_sources + i],
                t2=coordinates[:, number_of_point_sources + j],
                p1=coordinates[:, 2 * number_of_point_sources + i],
                p2=coordinates[:, 2 * number_of_point_sources + j],
                k=wave_vector,
                zr=rayleigh_length,
            )
            tau_mat[:, 2 * number_of_point_sources + i, number_of_point_sources + j] = (
                gaussian_overlaps.tau_tp(
                    s1=coordinates[:, i],
                    s2=coordinates[:, j],
                    t1=coordinates[:, number_of_point_sources + i],
                    t2=coordinates[:, number_of_point_sources + j],
                    p1=coordinates[:, 2 * number_of_point_sources + i],
                    p2=coordinates[:, 2 * number_of_point_sources + j],
                    k=wave_vector,
                    zr=rayleigh_length,
                )
            )
            tau_mat[
                :, 2 * number_of_point_sources + i, 2 * number_of_point_sources + j
            ] = gaussian_overlaps.tau_pp(
                s1=coordinates[:, i],
                s2=coordinates[:, j],
                t1=coordinates[:, number_of_point_sources + i],
                t2=coordinates[:, number_of_point_sources + j],
                p1=coordinates[:, 2 * number_of_point_sources + i],
                p2=coordinates[:, 2 * number_of_point_sources + j],
                k=wave_vector,
                zr=rayleigh_length,
            )
    return tau_mat

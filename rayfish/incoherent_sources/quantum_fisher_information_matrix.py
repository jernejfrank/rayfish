"""Calculating the quantum Fisher information matrix for incoherent point sources.

This module calculates the general quantum Fisher information matrix where
parameters correspond to the 3D coordinates of the point sources. This is a
starting point from which we then transform the matrix to the parameters of
interest.

The length scale for the position of the particles is normalized w.r.t.
the Rayleigh length [zr] and wavevector [k] as follows:
x,y <-> sqrt[k/zr] x,y
z <-> z / zr

The model assumes incoherent point sources distributed in 3D space and
a Gaussian beam in the image plane as the point spread function.
With the above relative length scale this looks like:
ψ(x-dx, y-dy, z-dz) =
sqrt[1/pi] (i / (z-dz+i))
exp[-i ((x-dx)^2 + (y-dy)^2) / (2 (z-dz+i))] exp[-i k zr (z-dz)]

The module implements the method proposed by Tufarelli, et.al.:
[1] Marco G Genoni and Tommaso Tufarelli, “Non-orthogonal bases for quantum metrology,”
Journal of Physics A: Mathematical and Theoretical 52, 434002 (2019).
[2] Lukas J Fiderer, Tommaso Tufarelli, Samanta Piano, and Gerardo Adesso,
“General expressions for the quantum fisher information matrix with applications
to discrete quantum imaging,” PRX Quantum 2, 020308 (2021).
"""

import numpy as np
import numpy.typing as npt

from rayfish import gaussian_overlaps

# Public API
__all__ = (
    "delta_submat",
    "gamma_submat",
    "tau_submat",
    "gramian",
    "density",
    "density_derivatives",
)


def delta_submat(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.complex128]:
    """Upper left submatrix for non-orthogonal coordinate transformations.

    Creates the submatrix consisting of overlaps between Gaussians based on the function
    gaussian_overlaps.delta(...).

    Args:
        coordinates: the position of individual Gaussians. Input shape is (M, 3N),
            where the N point sources should be provided in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: It is the distance along the propagation
            direction of a beam from the waist to the place where the area of
            1the cross section is doubled

    Returns:
        A (M,N,N)-array of complex floats, where N is the number of point sources
        and M is the number of different distributions of the N point sources.
        The diagonal for each (N,N) matraix is 1 since we assume normalized Gaussians.
    """
    delta_mat = (
        np.zeros((
            coordinates.shape[0],
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
        coordinates: the position of individual Gaussians. Input shape is (M, 3N),
            where the N point sources should be provided in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: It is the distance along the propagation
            direction of a beam from the waist to the place where the area of
            1the cross section is doubled

    Returns:
        A (M,4N,4N)-array of complex floats, where N is the number of point sources
        and M is the number of different distributions of the N point sources.
    """
    gamma_mat = (
        np.zeros((
            coordinates.shape[0],
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
        coordinates: the position of individual Gaussians. Input shape is (M, 3N),
            where the N point sources should be provided in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: It is the distance along the propagation
            direction of a beam from the waist to the place where the area of
            1the cross section is doubled

    Returns:
       A (M,3N,3N)-array of complex floats, where N is the number of point sources
       and M is the number of different distributions of the N point sources.
    """
    tau_mat = (
        np.zeros((
            coordinates.shape[0],
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


def gramian(
    delta_submat: npt.NDArray[np.complex128],
    gamma_submat: npt.NDArray[np.complex128],
    tau_submat: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Gramian matrix for non-orthogonal basis.

    Creates the Gramian matrix that is necessary for linear algebra in
    vector spaces spanned by non-orthogonal basis.

    The matrix takes the form:

    |  delta_submat  |  gamma_submat |
    |  gamma_submat  |   tau_submat  |

    where
    - dim(delta_submat) = (N,N)
    - dim(gamma_submat) = (N,3N) and (3N, N)
    - dim(tau_submat) = (3N,3N)

    Args:
        delta_submat: submatrix of overlaps between Gaussians
        gamma_submat: submatrix of overlaps between Gaussians and derivatives
        tau_submat: submatrox of overlaps between Gaussian derivatives

    Returns:
       A (M,4N,4N)-array of complex floats, where N is the number of point sources
       and M is the number of different distributions of the N point sources.
    """
    g = gamma_submat
    g[:, : delta_submat.shape[1], : delta_submat.shape[2]] += delta_submat
    g[:, delta_submat.shape[1] :, delta_submat.shape[2] :] += tau_submat
    return g


def density(number_of_point_sources: int) -> npt.NDArray[np.float64]:
    """Density matrix.

    For now we assume all sources have equal probability of emitting
    a photon (they are equally bright). The trace sums to 1.

    Args:
        number_of_point_sources: how many point sources we
            have to model the object

    Returns:
        A (4N,4N) matrix where the upper left (N,N) submatrix corresponds to the
        density matrix for N point sources.
    """
    r_full = np.zeros((
        4 * number_of_point_sources,
        4 * number_of_point_sources,
    ))
    r_sub = np.eye(number_of_point_sources) / number_of_point_sources
    r_full[:number_of_point_sources, :number_of_point_sources] = r_sub
    return r_full


def density_derivatives(number_of_point_sources: int) -> npt.NDArray[np.float64]:
    """Derivatives of the density matrix w.r.t. the parameters.

    For now we assume all sources have equal probability of emitting
    a photon (they are equally bright) so we can directly populate
    the individual derivative matrices.

    Args:
        number_of_point_sources: how many point sources we
            have to model the object

    Returns:
        A (3N,4N,4N) matrix where the (4N,4N) is sparse to encode
        the 3N parameter derivatives. The sequence matches the
        coordinates sequence (s1,s2,...,t1,t2,...,p1,p2,...).
    """
    d_tensor = np.zeros((
        3 * number_of_point_sources,
        4 * number_of_point_sources,
        4 * number_of_point_sources,
    ))

    for i in range(number_of_point_sources):
        d_s = np.zeros((4 * number_of_point_sources, 4 * number_of_point_sources))
        d_s[i, number_of_point_sources + i] = 1 / number_of_point_sources
        d_tensor[i] = d_s + np.transpose(d_s)

        d_t = np.zeros((4 * number_of_point_sources, 4 * number_of_point_sources))
        d_t[i, 2 * number_of_point_sources + i] = 1 / number_of_point_sources
        d_tensor[number_of_point_sources + i] = d_t + np.transpose(d_t)

        d_p = np.zeros((4 * number_of_point_sources, 4 * number_of_point_sources))
        d_p[i, 3 * number_of_point_sources + i] = 1 / number_of_point_sources
        d_tensor[2 * number_of_point_sources + i] = d_p + np.transpose(d_p)

    return d_tensor

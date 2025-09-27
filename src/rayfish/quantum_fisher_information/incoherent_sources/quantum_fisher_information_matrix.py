"""Calculating the quantum Fisher information matrix for incoherent point sources.

This module calculates the general quantum Fisher information matrix where parameters
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

The module implements the method proposed by Tufarelli, et.al.:
[1] Marco G Genoni and Tommaso Tufarelli, “Non-orthogonal bases for quantum metrology,”
Journal of Physics A: Mathematical and Theoretical 52, 434002 (2019).
[2] Lukas J Fiderer, Tommaso Tufarelli, Samanta Piano, and Gerardo Adesso,
“General expressions for the quantum fisher information matrix with applications
to discrete quantum imaging,” PRX Quantum 2, 020308 (2021).
"""

import numpy as np
import numpy.typing as npt
from hamilton.htypes import Collect, Parallelizable
from scipy import linalg

from rayfish import gaussian_overlaps

# Public API
__all__ = (
    "delta_submat",
    "gamma_submat",
    "gramian",
    "gramian_of_single_distribution",
    "density",
    "all_density_derivatives",
    "lyapunov_operators",
    "fisher_information_matrix",
    "fisher_information_matrix_for_entire_distribution",
    "covariance_matrix_for_entire_distribution",
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


def gramian(
    delta_submat: npt.NDArray[np.complex128],
    gamma_submat: npt.NDArray[np.complex128],
    tau_submat: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Gramian matrix for non-orthogonal basis.

    Creates the Gramian matrix that is necessary for linear algebra in vector spaces
    spanned by non-orthogonal basis.

    The matrix takes the form:

    |  delta_submat  |  gamma_submat |
    |  gamma_submat  |   tau_submat  |

    Args:
        delta_submat: (M,N,N) array of overlaps between Gaussians
        gamma_submat: (M,4N,4N) array. Only the upper right (N,3N) and
            lower left (3N, N) submatrices are populated with overlaps between
            Gaussians and derivatives
        tau_submat: (M,3N,3N) array of overlaps between Gaussian derivatives

    Returns:
       A (M,4N,4N) array of complex floats, where N is the number of point sources
       and M is the number of different distributions of the N point sources.
    """
    g = gamma_submat
    g[:, : delta_submat.shape[1], : delta_submat.shape[2]] += delta_submat
    g[:, delta_submat.shape[1] :, delta_submat.shape[2] :] += tau_submat
    return g


def gramian_of_single_distribution(
    gramian: npt.NDArray[np.complex128],
) -> Parallelizable[npt.NDArray[np.complex128]]:
    """Fans out the gramian to process the individual distributions.

    We can decide if this will occur in parallel or serial at execution.

    Args:
        gramian: (M,4N,4N) array. Computed matrix necessary for non-orthogonal algebra

    Returns:
        A (4N,4N) array that corresponds to the gramian of a single point source
        distribution.
    """
    for i in range(gramian.shape[0]):
        yield gramian[i]


def density(number_of_point_sources: int) -> npt.NDArray[np.float64]:
    """Density matrix.

    For now we assume all sources have equal probability of emitting
    a photon (they are equally bright). The trace sums to 1.

    Args:
        number_of_point_sources: how many point sources we
            have to model the object

    Returns:
        A (4N,4N) array where the upper left (N,N) submatrix corresponds to the
        density matrix for N point sources.
    """
    r_full = np.zeros((
        4 * number_of_point_sources,
        4 * number_of_point_sources,
    ))
    r_sub = np.eye(number_of_point_sources) / number_of_point_sources
    r_full[:number_of_point_sources, :number_of_point_sources] = r_sub
    return r_full


def all_density_derivatives(number_of_point_sources: int) -> npt.NDArray[np.complex128]:
    """Derivatives of the density matrix w.r.t. all parameters.

    For now we assume all sources have equal probability of emitting a photon (they are
    equally bright) so we can directly populate the individual derivative matrices.

    Args:
        number_of_point_sources: how many point sources we
            have to model the object

    Returns:
        A (3N,4N,4N) array where the (4N,4N) is sparse to encode the 3N parameter
        derivatives. The sequence matches the coordinates sequence
        (s1,s2,...,t1,t2,...,p1,p2,...).
    """
    d_tensor = np.zeros(
        (
            3 * number_of_point_sources,
            4 * number_of_point_sources,
            4 * number_of_point_sources,
        ),
        dtype=np.complex128,
    )

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


def lyapunov_operators(
    density: npt.NDArray[np.float64],
    all_density_derivatives: npt.NDArray[np.complex128],
    gramian_of_single_distribution: npt.NDArray[np.complex128],
    number_of_point_sources: int,
) -> npt.NDArray[np.complex128]:
    """Calculates Lyapunov operators using Sylvester's equation.

    A Lyapunov operator associated to a particular derivative of the density matrix is
    the solution to the equation

    2 * Density Derivative =
    Lyapunov * Gramian * Density + Density * Gramian *  Lyapunov

    We use the package scipy.linalg.solve_silvester(...)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_sylvester.html
    that solves matrix equations of the form

    AX + XB = Q

    for unknown X using the Bartels- Stewart algorithm. In our case we have:
    - A = Density * Gramian
    - B = Gramian * Density
    - Q = 2 * Density Derivative

    Since the density derivatives are sparse we can reduce the dimensionality
    of each Sylvester equation to N+1 and pad with 0 rows and columns back to
    the original (4N,4N) shape.

    Args:
        density: (4N,4N) array encoding intensities of the point source distribution
        all_density_derivatives: (3N,4N,4N) array encoding the derivative of the
            density matrix for each point source coordinate
        gramian_of_single_distribution: (4N,4N) array used for linear algebra
            in a vector space spanned by a non-orthogonal basis
        number_of_point_sources: how many point sources we
            have to model the object

    Returns:
        A (3N,4N,4N) array where the (4N,4N) are the Lyapunov operators for each of the
        3N coordinates. The sequence matches the coordinates sequence
        (s1,s2,...,t1,t2,...,p1,p2,...).
    """
    lyapunov = np.zeros_like(all_density_derivatives) + 0j
    for i in range(3 * number_of_point_sources):
        _slice = np.arange(0, number_of_point_sources + 1)
        _slice[-1] = number_of_point_sources + i
        dg = np.dot(
            density[np.ix_(_slice, _slice)],
            gramian_of_single_distribution[np.ix_(_slice, _slice)],
        )
        gd = np.dot(
            gramian_of_single_distribution[np.ix_(_slice, _slice)],
            density[np.ix_(_slice, _slice)],
        )
        der = (2 * all_density_derivatives[i])[np.ix_(_slice, _slice)]
        sub = linalg.solve_sylvester(dg, gd, der)
        lyapunov[i] = _pad_operator(sub, number_of_point_sources, i)
    return lyapunov


def _pad_operator(
    matrix: npt.NDArray[np.complex128], n: int, ind: int
) -> npt.NDArray[np.complex128]:
    """Helper function to pad a matrix.

    Inplace padding of the matrix with 0 columns / rows back to the original (dim,dim)
    around a certain column / row.

    Args:
        matrix: the submatrix to pad
        n: number of point sources
        ind: index of row/column around which we pad

    Returns:
        A (4n,4n) matrix where we padded so that the row/column entry matches the
        correct coordinates basis.
    """
    dim = 3 * n - 1
    before = ind
    after = dim - ind
    for _ in range(before):
        matrix = np.insert(np.insert(matrix, [n], 0, axis=0), [n], 0, axis=1)
    for _ in range(after):
        matrix = np.insert(
            np.insert(matrix, [n + 1 + before], 0, axis=0), [n + 1 + before], 0, axis=1
        )
    return matrix


def fisher_information_matrix(
    all_density_derivatives: npt.NDArray[np.complex128],
    gramian_of_single_distribution: npt.NDArray[np.complex128],
    lyapunov_operators: npt.NDArray[np.complex128],
    coordinate_jacobian: npt.NDArray[np.float64],
    number_of_point_sources: int,
) -> npt.NDArray[np.complex128]:
    """Calculates covariance matrix between Lyapunov operators.

    The covariance matrix is the general entity encoding parameter estimation
    boundaries. The real part corresponds to the quantum Fisher information
    matrix and the imaginary part to the compatibility matrix for simultaneously
    estimating parameters.

    Are Gaussian overlaps are calculated for displacements ψ(x+dx, y+dy, z+dz),
    but the actual distributions might depend a on different parametrization,
    i.e. ψ(x+s/2, y+t/2, z+p/2) + ψ(x-s/2, y-t/2, z-p/2), which we can use to
    express our covariance matrix in the coordinate system (s,t,p). For that
    we use the Jacobian of the coordinate transformation

    B = ∂f(dx[s,t,p],dy[s,t,p],dz[s,t,p]) / ∂(s,t,p)

    to do a change of basis of the covariance matrix

    Cov(s,t,p) = B(s,t,p) Cov(dx[s,t,p],dy[s,t,p],dz[s,t,p]) T^B(s,t,p),

    with T^B being the transpose. This would transform the (3N,3N) covariance
    matrix to a (N,N) matrix.

    Args:
        all_density_derivatives: (3N,4N,4N) array encoding the derivative of the
            density matrix for each point source coordinate
        gramian_of_single_distribution: (4N,4N) array used for linear algebra
            in a vector space spanned by a non-orthogonal basis
        lyapunov_operators: (3N,4N,4N) array encoding the stability of each
            parameter.
        coordinate_jacobian: (*,3N) array correcting the re-parametrization
            of point source coordinates and collapsing the matrix for the parameters
            of interest. The first dimension is dependent on the user input.
        number_of_point_sources: how many point sources we
            have to model the object

    Returns:
        A (*,*) array of complex floats encoding precision boundaries for
        parameters of interest that have been encoded into the `coordinate_jacobian`.
    """
    fisher_information = (
        np.zeros((3 * number_of_point_sources, 3 * number_of_point_sources)) + 0j
    )
    for i in range(3 * number_of_point_sources):
        for j in range(3 * number_of_point_sources):
            fisher_information[i, j] = np.trace(
                np.dot(
                    lyapunov_operators[i],
                    np.dot(
                        gramian_of_single_distribution,
                        np.dot(
                            all_density_derivatives[j], gramian_of_single_distribution
                        ),
                    ),
                )
            )

    fisher_information = np.matmul(
        coordinate_jacobian,
        np.matmul(fisher_information, np.transpose(coordinate_jacobian)),
    )

    return fisher_information


def fisher_information_matrix_for_entire_distribution(
    fisher_information_matrix: Collect[npt.NDArray[np.complex128]],
) -> npt.NDArray[np.complex128]:
    """Concatenates individual fisher information matrices.

    Args:
        fisher_information_matrix: (*,*) array, corresponding to the fisher information
            matrix for a single point source distribution

    Returns:
        A (M,*,*) array that concatenates the individual fisher information matrices for
        the parameters of interest.
    """
    return np.stack(fisher_information_matrix)  # type: ignore


def covariance_matrix_for_entire_distribution(
    fisher_information_matrix_for_entire_distribution: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    """The covariance matrix calculated as the inverse of the fisher information."""
    cov_matrices = np.zeros_like(
        fisher_information_matrix_for_entire_distribution, dtype=np.float64
    )

    fim_real = np.real(fisher_information_matrix_for_entire_distribution)

    # only invert non-singular matrices
    conds = np.linalg.cond(fim_real)
    mask = conds < 1e12
    inv_subset = np.linalg.inv(fim_real[mask])

    cov_matrices[mask] = inv_subset

    # set singular matrices to inf
    cov_matrices[~mask] = np.inf

    return cov_matrices

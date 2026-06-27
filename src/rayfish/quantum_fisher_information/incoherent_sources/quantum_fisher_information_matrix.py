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

# Public API
__all__ = (
    "fisher_information_matrix_for_entire_distribution",
    "covariance_matrix_for_entire_distribution",
)


def fisher_information_matrix_for_entire_distribution(
    density_derivatives: npt.NDArray[np.complex128],
    gramian: npt.NDArray[np.complex128],
    lyapunov_operators: npt.NDArray[np.complex128],
    coordinate_jacobian: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
) -> npt.NDArray[np.complex128]:
    """Calculates the quantum Fisher information matrix from the Lyapunov operators.

    Implements the non-orthogonal-basis QFIM,

        H_{a,b} = Re Tr(L_a G D_b G),

    evaluated for every parameter pair and every point source distribution. The full
    complex covariance between Lyapunov operators is the general entity encoding the
    parameter estimation boundaries: its real part is the quantum Fisher information
    matrix and its imaginary part the compatibility matrix for simultaneously estimating
    parameters. The trace is evaluated as batched matrix products, sandwiching the
    precomputed G D_b G with L_a over the whole distribution at once.

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
        density_derivatives: (3N,4N,4N) array encoding the derivative of the
            density matrix for each point source coordinate
        gramian: (M,4N,4N) array used for linear algebra
            in a vector space spanned by a non-orthogonal basis, one per distribution.
        lyapunov_operators: (M,3N,4N,4N) array encoding the SLD operator of each
            parameter for each distribution.
        coordinate_jacobian: (*,3N) array correcting the re-parametrization
            of point source coordinates and collapsing the matrix for the parameters
            of interest. The first dimension is dependent on the user input.
        number_of_point_sources: how many point sources we
            have to model the object.
        number_of_point_source_distributions: number M of different distributions.

    Returns:
        A (M,*,*) array of complex floats encoding precision boundaries for
        parameters of interest that have been encoded into the `coordinate_jacobian`.
    """
    gd = gramian[:, None] @ density_derivatives[None]
    gdg = gd @ gramian[:, None]

    l_flat = lyapunov_operators.reshape(
        number_of_point_source_distributions,
        3 * number_of_point_sources,
        4 * number_of_point_sources * 4 * number_of_point_sources,
    )
    gdg_flat = np.transpose(gdg, (0, 1, 3, 2)).reshape(
        number_of_point_source_distributions,
        3 * number_of_point_sources,
        4 * number_of_point_sources * 4 * number_of_point_sources,
    )

    fi_full = np.matmul(l_flat, np.transpose(gdg_flat, (0, 2, 1)))

    fisher_information = np.matmul(
        coordinate_jacobian,
        np.matmul(fi_full, np.transpose(coordinate_jacobian)),
    )

    return fisher_information


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

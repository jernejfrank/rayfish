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
ψ(x+dx, y+dy, z+dz) =
sqrt[1/pi] (i / (z+dz+i))
exp[-i ((x+dx)^2 + (y+dy)^2) / (2 (z+dz+i))] exp[-i k zr (z+dz)]

The module implements the method proposed by Tufarelli, et.al.:
[1] Marco G Genoni and Tommaso Tufarelli, “Non-orthogonal bases for quantum metrology,”
Journal of Physics A: Mathematical and Theoretical 52, 434002 (2019).
[2] Lukas J Fiderer, Tommaso Tufarelli, Samanta Piano, and Gerardo Adesso,
“General expressions for the quantum fisher information matrix with applications
to discrete quantum imaging,” PRX Quantum 2, 020308 (2021).
"""

import numpy as np
import numpy.typing as npt
from hamilton.function_modifiers import extract_fields

# Public API
__all__ = (
    "covariance_matrix_for_entire_distribution",
    "covariance_matrix_for_entire_distribution_nuisance_intensity",
    "covariance_matrix_for_entire_distribution_spatial",
    "fisher_information_matrix_and_compatibility_matrix",
    "fisher_information_matrix_using_density_derivatives",
)


def fisher_information_matrix_using_density_derivatives(
    density_derivatives: npt.NDArray[np.complex128],
    gramian: npt.NDArray[np.complex128],
    lyapunov_operators: npt.NDArray[np.complex128],
    coordinate_jacobian: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
) -> npt.NDArray[np.float64]:
    """Calculates the quantum Fisher information matrix from the Lyapunov operators.

    Implements the non-orthogonal-basis QFIM,

        H_{a,b} = Re Tr(L_a G D_b G),

    evaluated for every parameter pair and every point source distribution. The full
    complex covariance between Lyapunov operators is the general entity encoding the
    parameter estimation boundaries: its real part is the quantum Fisher information
    matrix and its imaginary part the compatibility matrix for simultaneously estimating
    parameters. The trace is evaluated as batched matrix products, sandwiching the
    precomputed G D_b G with L_a over the whole distribution at once.

    The Gaussian overlaps are calculated for displacements ψ(x+dx, y+dy, z+dz),
    but the actual distributions might depend a on different parametrization,
    i.e. ψ(x+s/2, y+t/2, z+p/2) + ψ(x-s/2, y-t/2, z-p/2), which we can use to
    express our covariance matrix in the coordinate system (s,t,p). For that
    we use the Jacobian of the coordinate transformation

    B = ∂f(dx[s,t,p],dy[s,t,p],dz[s,t,p]) / ∂(s,t,p)

    to do a change of basis of the covariance matrix

    Cov(s,t,p) = B(s,t,p) Cov(dx[s,t,p],dy[s,t,p],dz[s,t,p]) T^B(s,t,p),

    with T^B being the transpose. This transforms the (P,P) covariance matrix (P = 3N
    for position only, or P = 4N with intensity) into the space of the parameters of
    interest whose dimension is set by the leading dimension of `coordinate_jacobian`.

    Args:
        density_derivatives: (P,4N,4N) array encoding the derivative of the
            density matrix for each parameter, with P = 3N (position only) or P = 4N
            (intensity included).
        gramian: (M,4N,4N) array used for linear algebra
            in a vector space spanned by a non-orthogonal basis, one per distribution.
        lyapunov_operators: (M,P,4N,4N) array encoding the SLD operator of each
            parameter for each distribution.
        coordinate_jacobian: (*,P) array correcting the re-parametrization
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
        density_derivatives.shape[0],
        4 * number_of_point_sources * 4 * number_of_point_sources,
    )
    gdg_flat = np.transpose(gdg, (0, 1, 3, 2)).reshape(
        number_of_point_source_distributions,
        density_derivatives.shape[0],
        4 * number_of_point_sources * 4 * number_of_point_sources,
    )

    fi_full = np.matmul(l_flat, np.transpose(gdg_flat, (0, 2, 1)))

    fisher_information = np.matmul(
        coordinate_jacobian,
        np.matmul(fi_full, np.transpose(coordinate_jacobian)),
    )

    return np.real(fisher_information)


@extract_fields({
    "fisher_information_matrix_for_entire_distribution": npt.NDArray[np.float64],
    "compatibility_matrix": npt.NDArray[np.float64],
})
def fisher_information_matrix_and_compatibility_matrix(
    density: npt.NDArray[np.float64],
    gramian: npt.NDArray[np.complex128],
    lyapunov_operators: npt.NDArray[np.complex128],
    coordinate_jacobian: npt.NDArray[np.float64],
    number_of_point_sources: int,
) -> dict[str, npt.NDArray[np.float64]]:
    """Calculates the quantum Fisher information matrix from the Lyapunov operators.

    Implements the non-orthogonal-basis QFIM,

        H_{a,b} = Re Tr(L_a G D_b G),

    evaluated for every parameter pair and every point source distribution. The full
    complex covariance between Lyapunov operators is the general entity encoding the
    parameter estimation boundaries: its real part is the quantum Fisher information
    matrix and its imaginary part the compatibility matrix for simultaneously estimating
    parameters. The trace is evaluated as batched matrix products, sandwiching the
    precomputed G D_b G with L_a over the whole distribution at once.

    The Gaussian overlaps are calculated for displacements ψ(x+dx, y+dy, z+dz),
    but the actual distributions might depend a on different parametrization,
    i.e. ψ(x+s/2, y+t/2, z+p/2) + ψ(x-s/2, y-t/2, z-p/2), which we can use to
    express our covariance matrix in the coordinate system (s,t,p). For that
    we use the Jacobian of the coordinate transformation

    B = ∂f(dx[s,t,p],dy[s,t,p],dz[s,t,p]) / ∂(s,t,p)

    to do a change of basis of the covariance matrix

    Cov(s,t,p) = B(s,t,p) Cov(dx[s,t,p],dy[s,t,p],dz[s,t,p]) T^B(s,t,p),

    with T^B being the transpose. This transforms the (P,P) covariance matrix (P = 3N
    for position only, or P = 4N with intensity) into the space of the parameters of
    interest whose dimension is set by the leading dimension of `coordinate_jacobian`.

    Args:
        density: (4N,4N) array encoding the the density matrix for each parameter.
        gramian: (M,4N,4N) array used for linear algebra
            in a vector space spanned by a non-orthogonal basis, one per distribution.
        lyapunov_operators: (M,P,4N,4N) array encoding the SLD operator of each
            parameter for each distribution.
        coordinate_jacobian: (*,P) array correcting the re-parametrization
            of point source coordinates and collapsing the matrix for the parameters
            of interest. The first dimension is dependent on the user input.
        number_of_point_sources: how many point sources we
            have to model the object.

    Returns:
        A (M,*,*) array of complex floats encoding precision boundaries for
        parameters of interest that have been encoded into the `coordinate_jacobian`.
    """
    rg = density[None] @ gramian
    grg = gramian @ rg
    grgl_a = grg[:, None] @ lyapunov_operators

    gl_b = gramian[:, None] @ lyapunov_operators

    grgl_a_flat = grgl_a.reshape(
        lyapunov_operators.shape[0],
        lyapunov_operators.shape[1],
        4 * number_of_point_sources * 4 * number_of_point_sources,
    )

    gl_b_flat = np.transpose(gl_b, axes=(0, 1, 3, 2)).reshape(
        lyapunov_operators.shape[0],
        lyapunov_operators.shape[1],
        4 * number_of_point_sources * 4 * number_of_point_sources,
    )

    fi_full = grgl_a_flat @ np.transpose(gl_b_flat, (0, 2, 1))

    fisher_information = coordinate_jacobian @ (
        fi_full @ np.transpose(coordinate_jacobian)
    )

    return {
        "fisher_information_matrix_for_entire_distribution": np.real(
            fisher_information
        ),
        "compatibility_matrix": np.imag(fisher_information),
    }


def covariance_matrix_for_entire_distribution(
    fisher_information_matrix_for_entire_distribution: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The covariance matrix calculated as the inverse of the fisher information."""
    cov_matrices = np.zeros_like(
        fisher_information_matrix_for_entire_distribution, dtype=np.float64
    )

    fim_real = np.real(fisher_information_matrix_for_entire_distribution)

    # only invert non-singular matrices
    conds = np.linalg.cond(fim_real)
    mask = conds < 1e16
    inv_subset = np.linalg.inv(fim_real[mask])

    cov_matrices[mask] = inv_subset

    # set singular matrices to inf
    cov_matrices[~mask] = np.inf

    return cov_matrices


def covariance_matrix_for_entire_distribution_nuisance_intensity(
    number_of_spatial_params: int,
    fisher_information_matrix_for_entire_distribution: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Spatial covariance with the intensity parameters marginalized out.

    Treats the intensity parameters as nuisance parameters and removes them via the
    Schur complement of the intensity block. Writing the (real part of the) Fisher
    information matrix in block form over spatial (S) and intensity (I) parameters,

        H = | H_SS   H_SI |,
            | H_IS   H_II |

    the marginal Fisher information for the spatial parameters is the Schur complement

        H_S = H_SS - H_SI H_II^{-1} H_IS,

    and the returned covariance is its inverse. Singular blocks are handled by masking
    on the condition number; distributions whose marginal Fisher information is singular
    get an infinite covariance.

    Args:
        number_of_spatial_params: number of leading (spatial) parameters; the remaining
            trailing parameters are the intensity nuisance parameters that get
            marginalized out.
        fisher_information_matrix_for_entire_distribution: (M,P,P) array, the full
            Fisher information matrix per distribution including both spatial and
            intensity parameters.

    Returns:
        A (M, number_of_spatial_params, number_of_spatial_params) real array of spatial
        covariance matrices, one per distribution, with singular cases set to inf.
    """
    # The QFIM is the real part of the complex covariance; take it before the Schur
    # complement so the marginal stays real instead of being silently cast later.
    fim_real = np.real(fisher_information_matrix_for_entire_distribution)

    intensity_fi = fim_real[:, number_of_spatial_params:, number_of_spatial_params:]
    # only invert non-singular matrices
    conds = np.linalg.cond(intensity_fi)
    mask = conds < 1e16
    inv_intensity_fi = np.zeros_like(intensity_fi)
    inv_subset = np.linalg.inv(intensity_fi[mask])
    inv_intensity_fi[mask] = inv_subset

    old_fi = fim_real[:, :number_of_spatial_params, :number_of_spatial_params]

    left_complement = fim_real[:, :number_of_spatial_params, number_of_spatial_params:]

    right_complement = fim_real[:, number_of_spatial_params:, :number_of_spatial_params]

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
    """Spatial covariance restricted to the spatial Fisher information block.

    Restricts the (real part of the) Fisher information matrix to its leading spatial
    block H_SS and inverts it, ignoring any intensity parameters entirely (as opposed to
    the nuisance-intensity variant, which marginalizes them out via a Schur complement).
    Singular blocks are handled by masking on the condition number; distributions whose
    spatial block is singular get an infinite covariance.

    Args:
        fisher_information_matrix_for_entire_distribution: (M,P,P) array, the full
            Fisher information matrix per distribution.
        number_of_spatial_params: number of leading (spatial) parameters to keep; the
            (M, number_of_spatial_params, number_of_spatial_params) block is inverted.

    Returns:
        A (M, number_of_spatial_params, number_of_spatial_params) real array of spatial
        covariance matrices, one per distribution, with singular cases set to inf.
    """
    fi = fisher_information_matrix_for_entire_distribution[
        :, :number_of_spatial_params, :number_of_spatial_params
    ]
    cov_matrices = np.zeros_like(fi, dtype=np.float64)

    fim_real = np.real(fi)

    # only invert non-singular matrices
    conds = np.linalg.cond(fim_real)
    mask = conds < 1e12
    inv_subset = np.linalg.inv(fim_real[mask])

    cov_matrices[mask] = inv_subset

    # set singular matrices to inf
    cov_matrices[~mask] = np.inf

    return cov_matrices

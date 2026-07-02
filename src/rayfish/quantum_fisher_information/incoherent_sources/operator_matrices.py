"""Construct the Gramian, density and derivative matrices in the non-orthogonal basis.

Following the non-orthogonal basis approach (Genoni-Tufarelli / Fiderer et al.), every
operator is expanded in the basis {|ψ_v>, |∂_{δx_v}ψ_v>, |∂_{δy_v}ψ_v>, |∂_{δz_v}ψ_v>}
and written in the block form

    X = [[X_Δ,   X_γ ],
         [X_γ†,  X_τ ]],

where the Δ block (V×V) is the PSF-manifold geometry, the τ block (3V×3V) the overlaps
between tangent vectors, and the γ / γ† blocks (V×3V, 3V×V) the PSF-tangent couplings.
This module assembles the Gramian G_{u,v} = <ψ_u|ψ_v>, the density matrix
R = Σ_v I_v |ψ_v><ψ_v| and its parameter derivatives D_a from the precomputed overlap
submatrices.
"""

import numpy as np
import numpy.typing as npt

__all__ = (
    "gramian",
    "density",
    "density_derivatives",
)


def gramian(
    delta_submat: npt.NDArray[np.complex128],
    gamma_submat: npt.NDArray[np.complex128],
    tau_submat: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Gramian matrix for the non-orthogonal basis.

    Assembles the Gram matrix G_{u,v} = <ψ_u|ψ_v> used for linear algebra in the vector
    space spanned by the non-orthogonal basis. In block form it reads

        G = |  G_Δ   G_γ  |   =   |  delta_submat   gamma_submat |
            |  G_γ†  G_τ  |       |  gamma_submat    tau_submat  |,

    with G_Δ the PSF overlaps, G_γ / G_γ† the PSF-derivative overlaps and G_τ the
    derivative-derivative overlaps.

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
    g = np.zeros_like(gamma_submat)
    g += gamma_submat
    g[:, : delta_submat.shape[1], : delta_submat.shape[2]] = delta_submat
    g[:, delta_submat.shape[1] :, delta_submat.shape[2] :] = tau_submat
    return g


def density(
    number_of_point_sources: int, point_source_intensities: list[float]
) -> npt.NDArray[np.float64]:
    """Density matrix of the incoherent point-source ensemble.

    Implements R = Σ_v I_v |ψ_v><ψ_v|, which in the
    non-orthogonal basis has support only on the Δ block,

        R_Δ = diag(I_1, ..., I_V),

    with the relative intensities normalized so that Σ_v I_v = Tr R = 1. Equal sources
    I_v = 1/V recover the uniform density R_Δ = (1/V) 1_V used in the main text.

    Args:
        number_of_point_sources: how many point sources we
            have to model the object
        point_source_intensities: relative intensities I_v of the point sources,
            summing to 1.

    Returns:
        A (4N,4N) array where the upper left (N,N) submatrix corresponds to the
        density matrix for N point sources.
    """
    r_full = np.zeros((
        4 * number_of_point_sources,
        4 * number_of_point_sources,
    ))
    r_sub = np.diag(point_source_intensities)
    r_full[:number_of_point_sources, :number_of_point_sources] = r_sub
    return r_full


def density_derivatives(
    number_of_point_sources: int,
    point_source_intensities: list[float],
) -> npt.NDArray[np.complex128]:
    """Derivatives of the density matrix w.r.t. all parameters.

    Each derivative matrix D_a is sparse with only two non-zero entries that couple a
    source to its tangent vector,

        D_a^{∂u,u} = D_a^{u,∂u} = I_u,

    where ∂u indexes the basis vector |∂_{θ_u^a}ψ_u>. Equal sources I_u = 1/V recover
    the uniform-intensity expression of the main text.

    Args:
        number_of_point_sources: how many point sources we
            have to model the object
        point_source_intensities: relative intensities I_v of the point sources,
            summing to 1.

    Returns:
        A (3N,4N,4N) array where the (4N,4N) is sparse to encode the 3N parameter
        derivatives. The sequence matches the coordinates sequence
        (s1,s2,...,t1,t2,...,p1,p2,...).
    """
    d_tensor = np.zeros(
        (
            4 * number_of_point_sources,
            4 * number_of_point_sources,
            4 * number_of_point_sources,
        ),
        dtype=np.complex128,
    )

    for i in range(number_of_point_sources):
        d_s = np.zeros((4 * number_of_point_sources, 4 * number_of_point_sources))
        d_s[i, number_of_point_sources + i] = point_source_intensities[i]
        d_tensor[i] = d_s + np.transpose(d_s)

        d_t = np.zeros((4 * number_of_point_sources, 4 * number_of_point_sources))
        d_t[i, 2 * number_of_point_sources + i] = point_source_intensities[i]
        d_tensor[number_of_point_sources + i] = d_t + np.transpose(d_t)

        d_p = np.zeros((4 * number_of_point_sources, 4 * number_of_point_sources))
        d_p[i, 3 * number_of_point_sources + i] = point_source_intensities[i]
        d_tensor[2 * number_of_point_sources + i] = d_p + np.transpose(d_p)

        d_intensity = np.zeros((
            4 * number_of_point_sources,
            4 * number_of_point_sources,
        ))
        d_intensity[i, i] = 1
        d_tensor[3 * number_of_point_sources + i] = d_intensity

    return d_tensor

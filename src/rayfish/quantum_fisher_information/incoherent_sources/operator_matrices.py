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
    "include_intensity_params",
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


def include_intensity_params(
    coordinate_jacobian: npt.NDArray[np.float64],
    number_of_point_sources: int,
) -> bool:
    """Infer whether intensity is treated as a (nuisance) parameter.

    The coordinate Jacobian maps the raw per-source parameters onto the parameters of
    interest, so its width equals the number of raw parameters. Position-only estimation
    uses 3N parameters (s, t, p per source); promoting intensity to a parameter adds one
    per source, giving 4N. We read this off the Jacobian rather than passing a separate
    flag so the two can never disagree.

    Args:
        coordinate_jacobian: (*, 3N) or (*, 4N) array re-parametrizing the raw
            per-source coordinates onto the parameters of interest.
        number_of_point_sources: how many point sources model the object.

    Returns:
        True if the Jacobian carries the N extra intensity parameters (width 4N),
        False for the position-only case (width 3N).
    """
    return coordinate_jacobian.shape[-1] == 4 * number_of_point_sources


def density_derivatives(
    number_of_point_sources: int,
    point_source_intensities: list[float],
    include_intensity_params: bool,
) -> npt.NDArray[np.complex128]:
    """Derivatives of the density matrix w.r.t. all parameters.

    Each spatial derivative matrix D_a is sparse with only two non-zero entries that
    couple a source to its tangent vector,

        D_a^{∂u,u} = D_a^{u,∂u} = I_u,

    where ∂u indexes the basis vector |∂_{θ_u^a}ψ_u>. Equal sources I_u = 1/V recover
    the uniform-intensity expression of the main text. When intensity is treated as a
    (nuisance) parameter, we additionally emit the N diagonal intensity derivatives
    D_a^{u,u} = 1, one per source.

    Args:
        number_of_point_sources: how many point sources we
            have to model the object
        point_source_intensities: relative intensities I_v of the point sources,
            summing to 1.
        include_intensity_params: if True, append the N intensity derivatives so the
            first axis has length 4N; otherwise emit only the 3N spatial derivatives.
            Use `include_intensity_params()` to infer this from the coordinate Jacobian.

    Returns:
        A (P,4N,4N) array where the (4N,4N) blocks are sparse and P = 3N (position only)
        or P = 4N (with intensity). The parameter sequence matches the coordinate
        sequence (s1,s2,...,t1,t2,...,p1,p2,...) followed by (I1,I2,...) when included.
    """
    number_of_parameters = (
        4 * number_of_point_sources
        if include_intensity_params
        else 3 * number_of_point_sources
    )
    d_tensor = np.zeros(
        (
            number_of_parameters,
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

        if include_intensity_params:
            d_intensity = np.zeros((
                4 * number_of_point_sources,
                4 * number_of_point_sources,
            ))
            d_intensity[i, i] = 1
            d_tensor[3 * number_of_point_sources + i] = d_intensity

    return d_tensor

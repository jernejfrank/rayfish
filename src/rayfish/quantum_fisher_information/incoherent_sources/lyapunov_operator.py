"""Solving for Lyapunov (SLD) operators via a vectorized, batched Sylvester solve."""

import numpy as np
import numpy.typing as npt


def lyapunov_operators(
    density: npt.NDArray[np.float64],
    density_derivatives: npt.NDArray[np.complex128],
    gramian: npt.NDArray[np.complex128],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
) -> npt.NDArray[np.complex128]:
    """Calculates the Lyapunov (SLD) operators from the reduced Sylvester equation.

    The symmetric logarithmic derivative L_a for parameter θ_a satisfies the SLD
    relation in the non-orthogonal basis,

        2 D_a = L_a G R + R G L_a,

    which is a Sylvester equation A X + X B = Q with A = R G, B = G R and
    Q = 2 D_a. Because each derivative matrix D_a is sparse, the equation reduces to
    the (N+1)-dimensional subsystem spanned by the N source populations and the single
    tangent index N+a; the solution is padded with zero rows
    and columns back to the full (4N,4N) shape.

    The reduced density block diag(I_1, ..., I_V, 0) carries a zero eigenvalue, so A and
    -B share the eigenvalue 0 and the Sylvester equation is singular (non-unique X).
    Rather than scipy's Bartels-Stewart (which is also strictly 2-D), we vectorize the
    equation with the Kronecker identity

        (I ⊗ A + B^T ⊗ I) vec(X) = vec(Q),

    and take the minimum-norm solution via a batched ``np.linalg.pinv`` over all M
    distributions at once. The QFIM is invariant to the kernel ambiguity in X, so the
    minimum-norm choice is immaterial to the final result.

    Args:
        density: (4N,4N) array encoding the intensities of the point source
            distribution; constant across distributions.
        density_derivatives: (3N,4N,4N) array encoding the derivative of the
            density matrix for each point source coordinate; constant across
            distributions.
        gramian: (M,4N,4N) array used for linear algebra in the non-orthogonal basis,
            one Gramian per point source distribution.
        number_of_point_sources: how many point sources we
            have to model the object
        number_of_point_source_distributions: number M of different distributions.

    Returns:
        A (M,3N,4N,4N) array where the (4N,4N) blocks are the Lyapunov operators for
        each of the 3N coordinates and each of the M distributions. The coordinate
        sequence matches (s1,s2,...,t1,t2,...,p1,p2,...).
    """
    lyapunov = (
        np.zeros((
            number_of_point_source_distributions,
            3 * number_of_point_sources,
            4 * number_of_point_sources,
            4 * number_of_point_sources,
        ))
        + 0j
    )

    n = number_of_point_sources + 1
    for i in range(3 * number_of_point_sources):
        _slice = np.arange(0, n)
        _slice[-1] = number_of_point_sources + i

        gsub = gramian[:, _slice[:, None], _slice[None, :]]
        dsub = density[np.ix_(_slice, _slice)]

        dg = dsub @ gsub
        gd = gsub @ dsub
        der = (2 * density_derivatives[i])[np.ix_(_slice, _slice)]

        identity = np.eye(n)

        k = (
            identity[None, :, None, :, None] * dg[:, None, :, None, :]
            + np.transpose(gd, (0, 2, 1))[:, :, None, :, None]
            * identity[None, None, :, None, :]
        )

        k = k.reshape(number_of_point_source_distributions, n * n, n * n)

        vec_der = np.transpose(der).reshape(-1)
        lyapunov_vec = np.linalg.pinv(k) @ vec_der

        z = (
            np.zeros((
                number_of_point_source_distributions,
                4 * number_of_point_sources,
                4 * number_of_point_sources,
            ))
            + 0j
        )
        z[:, _slice[:, None], _slice[None, :]] = lyapunov_vec.reshape(
            number_of_point_source_distributions, n, n
        ).transpose(0, 2, 1)

        lyapunov[:, i] = z

    return lyapunov

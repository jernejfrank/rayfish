"""Solving for Lyapunov operators via pinv on a vectorized Sylvester equation."""

import numpy as np
import numpy.typing as npt


def lyapunov_operators(
    density: npt.NDArray[np.float64],
    density_derivatives: npt.NDArray[np.complex128],
    gramian: npt.NDArray[np.complex128],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
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
        density_derivatives: (3N,4N,4N) array encoding the derivative of the
            density matrix for each point source coordinate
        gramian: (4N,4N) array used for linear algebra
            in a vector space spanned by a non-orthogonal basis
        number_of_point_sources: how many point sources we
            have to model the object
        number_of_point_source_distributions: number of different distributions.

    Returns:
        A (3N,4N,4N) array where the (4N,4N) are the Lyapunov operators for each of the
        3N coordinates. The sequence matches the coordinates sequence
        (s1,s2,...,t1,t2,...,p1,p2,...).
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

"""Module to collect Grammian , Density and Derivative matrices."""

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
    g = np.zeros_like(gamma_submat)
    g += gamma_submat
    g[:, : delta_submat.shape[1], : delta_submat.shape[2]] = delta_submat
    g[:, delta_submat.shape[1] :, delta_submat.shape[2] :] = tau_submat
    return g


def density(
    number_of_point_sources: int, point_source_intensities: list[float]
) -> npt.NDArray[np.float64]:
    """Density matrix.

    For now we assume all sources have equal probability of emitting
    a photon (they are equally bright). The trace sums to 1.

    Args:
        number_of_point_sources: how many point sources we
            have to model the object
        point_source_intensities: intensity distribution for point sources.

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
    number_of_point_sources: int, point_source_intensities: list[float]
) -> npt.NDArray[np.complex128]:
    """Derivatives of the density matrix w.r.t. all parameters.

    For now we assume all sources have equal probability of emitting a photon (they are
    equally bright) so we can directly populate the individual derivative matrices.

    Args:
        number_of_point_sources: how many point sources we
            have to model the object
        point_source_intensities: intensity distribution for point sources.

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
        d_s[i, number_of_point_sources + i] = point_source_intensities[i]
        d_tensor[i] = d_s + np.transpose(d_s)

        d_t = np.zeros((4 * number_of_point_sources, 4 * number_of_point_sources))
        d_t[i, 2 * number_of_point_sources + i] = point_source_intensities[i]
        d_tensor[number_of_point_sources + i] = d_t + np.transpose(d_t)

        d_p = np.zeros((4 * number_of_point_sources, 4 * number_of_point_sources))
        d_p[i, 3 * number_of_point_sources + i] = point_source_intensities[i]
        d_tensor[2 * number_of_point_sources + i] = d_p + np.transpose(d_p)

    return d_tensor

"""Benchmarking comparison to direct imaging of incoherent point sources.

Calculating the classical Fisher information matrix for incoherent point sources using
direct imaging (DI) which is modeled by a gaussian positioned at the origin of the
coordinate system: ψ(x,y,0).

This module calculates the classical Fisher information matrix where parameters
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
exp[-i ((x-dx)^2 + (y-dy)^2) / (2 (z-dz+i))] exp[-i k zr (z-dz)].
"""

import numpy as np
import numpy.typing as npt

from rayfish import gaussian_overlaps

# Public API
__all__ = (
    "fisher_information_matrix_for_entire_distribution",
    "covariance_matrix_for_entire_distribution",
    "numerical_grid",
)


def numerical_grid(
    min_z_displacement: float,
    max_z_displacement: float,
    maximal_length: float,
    numerical_integration_resolution: int,
    numerical_integration_support: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Creates the underlying x-y plane for numerical integration."""
    sigma_min = np.sqrt((1.0 + min_z_displacement**2) / 2.0)
    sigma_max = np.sqrt((1.0 + max_z_displacement**2) / 2.0)

    measure_x = sigma_min / numerical_integration_resolution
    measure_y = measure_x

    pad = sigma_max * numerical_integration_support

    xmin = -maximal_length - pad
    xmax = maximal_length + pad
    ymin = -maximal_length - pad
    ymax = maximal_length + pad

    x = np.arange(xmin, xmax + measure_x / 2.0, measure_x)
    y = np.arange(ymin, ymax + measure_y / 2.0, measure_y)

    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    return x_grid, y_grid


def _total_probability(
    x_: npt.NDArray[np.float64],
    y_: npt.NDArray[np.float64],
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    point_source_intensities: list[float],
) -> npt.NDArray[np.float64]:
    """Image-plane intensity P(x,y|θ) = Σ_v I_v |<x,y|ψ_v>|^2 over all sources."""
    ny, nx = x_.shape
    prob = np.zeros((coordinates.shape[0], ny, nx))
    for point_source in range(number_of_point_sources):
        p_ = point_source_intensities[point_source] * np.real(
            gaussian_overlaps.probability_measurement_di_point_source_overlap(
                x_,
                y_,
                coordinates[:, point_source],
                coordinates[:, number_of_point_sources + point_source],
                coordinates[:, 2 * number_of_point_sources + point_source],
            )
        )

        prob += p_
    return prob


def _probability_derivatives(
    x_: npt.NDArray[np.float64],
    y_: npt.NDArray[np.float64],
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    point_source_intensities: list[float],
) -> npt.NDArray[np.float64]:
    """Derivatives ∂_a P of the image-plane intensity w.r.t. the 3N coordinates."""
    ny, nx = x_.shape
    dprob = np.zeros((
        coordinates.shape[0],
        3 * number_of_point_sources,
        ny,
        nx,
    ))
    for i in range(number_of_point_sources):
        prob = np.real(
            gaussian_overlaps.probability_measurement_di_point_source_overlap(
                x_,
                y_,
                coordinates[:, i],
                coordinates[:, number_of_point_sources + i],
                coordinates[:, 2 * number_of_point_sources + i],
            )
        )
        ds = point_source_intensities[i] * np.real(
            gaussian_overlaps.derivative_s_probability_measurement_di_point_source_overlap(
                x_,
                y_,
                s=coordinates[:, i],
                t=coordinates[:, number_of_point_sources + i],
                p=coordinates[:, 2 * number_of_point_sources + i],
                probability=prob,
            )
        )
        dt = point_source_intensities[i] * np.real(
            gaussian_overlaps.derivative_t_probability_measurement_di_point_source_overlap(
                x_,
                y_,
                s=coordinates[:, i],
                t=coordinates[:, number_of_point_sources + i],
                p=coordinates[:, 2 * number_of_point_sources + i],
                probability=prob,
            )
        )
        dp = point_source_intensities[i] * np.real(
            gaussian_overlaps.derivative_p_probability_measurement_di_point_source_overlap(
                x_,
                y_,
                s=coordinates[:, i],
                t=coordinates[:, number_of_point_sources + i],
                p=coordinates[:, 2 * number_of_point_sources + i],
                probability=prob,
            )
        )

        dprob[:, i] = ds
        dprob[:, number_of_point_sources + i] = dt
        dprob[:, 2 * number_of_point_sources + i] = dp

    return dprob


def fisher_information_matrix_for_entire_distribution(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    coordinate_jacobian: npt.NDArray[np.float64],
    numerical_grid: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    point_source_intensities: list[float],
    batch_size: int,
) -> npt.NDArray[np.float64]:
    """Direct-imaging classical Fisher information matrix.

    For direct imaging the image-plane distribution is the intensity sum over sources,

        P(x,y|θ) = Σ_v I_v |<x,y|ψ_v(x+δx_v, y+δy_v, δz_v)>|^2,

    and the classical Fisher information follows the Poissonian form,

        J_{a,b} = ∫ dx dy (1/P) (∂_a P)(∂_b P),

    here evaluated by numerical quadrature over the image plane and then transformed to
    the parameters of interest via the Jacobian, J̃ = B J B^T. Equal intensities
    I_v = 1/V recover the uniform 1/V weighting.


    Args:
        coordinates: (M, 3N) array. The position of individual Gaussians. N is the
            number of point sources in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        number_of_point_source_distributions: how many different point source
            distributions we are computing for.
        coordinate_jacobian: transformation from 3N parameters to the parameters of
            interest.
        numerical_grid: the underliying x-y discretized plane for numerical integration.
        point_source_intensities: intensity distribution for point sources.
        batch_size: size of a single batch to avoid memory issues during computation.


    Returns:
        A (M,) array of complex floats, where B is the number of measurement modes and
        M is the number of different distributions of the N point sources.
    """
    x_, y_ = numerical_grid
    dx = float(x_[0, 1] - x_[0, 0])
    dy = float(y_[1, 0] - y_[0, 0])
    pixel_area = dx * dy
    ny, nx = x_.shape
    n_params_interests = coordinate_jacobian.shape[0]

    fisher_information = np.zeros((
        number_of_point_source_distributions,
        n_params_interests,
        n_params_interests,
    ))
    for start in range(0, number_of_point_source_distributions, batch_size):
        stop = min(start + batch_size, number_of_point_source_distributions)
        actual_batch_size = stop - start
        # Compute probability
        prob = _total_probability(
            x_,
            y_,
            coordinates[start:stop],
            number_of_point_sources,
            point_source_intensities,
        )

        # Compute derivatives
        dprob = _probability_derivatives(
            x_,
            y_,
            coordinates[start:stop],
            number_of_point_sources,
            point_source_intensities,
        )

        fi = np.zeros((actual_batch_size, n_params_interests, n_params_interests))
        p_flat = prob.reshape(actual_batch_size, ny * nx)
        dp_flat = dprob.reshape(actual_batch_size, 3 * number_of_point_sources, ny * nx)
        for bi in range(actual_batch_size):
            mask = p_flat[bi] > 1e-16
            if not np.any(mask):
                continue

            dp_flat_masked = dp_flat[bi, :, mask]
            p_flat_masked = (p_flat[bi])[mask]
            w = dp_flat_masked / np.sqrt(p_flat_masked)[:, None]

            # The actual integral
            wrong_fi = np.matmul(np.transpose(w), w)
            wrong_fi *= pixel_area

            # Symmetrize for numerical accuracy
            wrong_fi = 0.5 * (wrong_fi + wrong_fi.T)

            # Coordinate transformation to account for minus signs and collapse to
            # parameters of interest
            fi[bi] = np.matmul(
                coordinate_jacobian, np.matmul(wrong_fi, coordinate_jacobian.T)
            )

        fisher_information[start:stop] = fi

    return fisher_information


def covariance_matrix_for_entire_distribution(
    fisher_information_matrix_for_entire_distribution: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The covariance matrix calculated as the inverse of the fisher information."""
    cov_matrices = np.zeros_like(
        fisher_information_matrix_for_entire_distribution, dtype=np.float64
    )

    # # only invert non-singular matrices
    # for i in range(number_of_point_source_distributions):
    #     fisher_information = fisher_information_matrix_for_entire_distribution[i]
    #     try:
    #         cov = np.linalg.inv(fisher_information)
    #     except np.linalg.LinAlgError:
    #         cov = np.linalg.pinv(fisher_information)

    #     cov_matrices[i] = cov

    conds = np.linalg.cond(fisher_information_matrix_for_entire_distribution)
    mask = conds < 1e16
    inv_subset = np.linalg.inv(fisher_information_matrix_for_entire_distribution[mask])

    cov_matrices[mask] = inv_subset

    # set singular matrices to inf
    cov_matrices[~mask] = np.inf

    return cov_matrices

"""Calculating the quantum Fisher information matrix for coherent point sources.

This module calculates the general quantum Fisher information matrix where parameters
correspond to the 3D coordinates of the point sources. This is a starting point from
which we then transform the matrix to the parameters of interest.

The length scale for the position of the particles is normalized w.r.t. the Rayleigh
length [zr] and wavevector [k] as follows:
x,y <-> sqrt[k/zr] x,y
z <-> z / zr

The model assumes coherent point sources distributed in 3D space and a Gaussian beam
in the image plane as the point spread function. With the above relative length scale
this looks like:
ψ(x-dx, y-dy, z-dz) =
sqrt[1/pi] (i / (z-dz+i))
exp[-i ((x-dx)^2 + (y-dy)^2) / (2 (z-dz+i))] exp[-i k zr (z-dz)]

In the case of coherent point sources the quantum Fisher information matrix is a
scaling multiple of the tau-submatrix of the quantum Fisher information matrix for
incoherent point sources.
"""

import numpy as np
import numpy.typing as npt

# Public API
__all__ = ("covariance_matrix_for_entire_distribution",)


def covariance_matrix_for_entire_distribution(
    number_of_point_sources: int,
    tau_submat: npt.NDArray[np.complex128],
    coordinate_jacobian: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Quantum Fisher information matrix for coherent point sources.

    For coherent sources the QFIM reduces to a scaled τ block of the Gram matrix
    (overlaps between PSF tangent vectors),

        H = (4 / V^2) G_τ,

    which is then transformed to the parameters of interest via the Jacobian,
    H̃ = B H B^T.

    Args:
        number_of_point_sources: how many point sources we use to model the object
        tau_submat: (M,3N,3N) array of overlaps between Gaussian derivatives, i.e. the
            G_τ block of the Gram matrix
        coordinate_jacobian: (*,3N) array correcting the re-parametrization
            of point source coordinates and collapsing the matrix for the parameters
            of interest. The first dimension is dependent on the user input.

    Returns:
        A (M,*,*) array of complex floats encoding precision boundaries for
        parameters of interest that have been encoded into the `coordinate_jacobian`.
    """
    covariance = 4 * tau_submat / number_of_point_sources**2
    covariance = np.matmul(
        coordinate_jacobian, np.matmul(covariance, np.transpose(coordinate_jacobian))
    )
    return covariance

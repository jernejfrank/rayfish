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

from rayfish.incoherent_sources.quantum_fisher_information_matrix import tau_submat

# Public API
__all__ = ("covariance_matrix_for_entire_distribution",)


def covariance_matrix_for_entire_distribution(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    number_of_point_source_distributions: int,
    wave_vector: float,
    rayleigh_length: float,
    coordinate_jacobian: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Creates covariance matrix by calculating tau-submatrix.

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
        coordinate_jacobian: (*,3N) array correcting the re-parametrization
            of point source coordinates and collapsing the matrix for the parameters
            of interest. The first dimension is dependent on the user input.

    Returns:
        A (M,*,*) array of complex floats encoding precision boundaries for
        parameters of interest that have been encoded into the `coordinate_jacobian`.
    """
    scaling_factor = 4 / number_of_point_sources**2
    covariance = scaling_factor * tau_submat(
        coordinates=coordinates,
        number_of_point_sources=number_of_point_sources,
        number_of_point_source_distributions=number_of_point_source_distributions,
        wave_vector=wave_vector,
        rayleigh_length=rayleigh_length,
    )

    covariance = np.matmul(
        coordinate_jacobian, np.matmul(covariance, np.transpose(coordinate_jacobian))
    )
    return covariance

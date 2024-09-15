"""Calculating the quantum Fisher information matrix for incoherent point sources.

This module calculates the general quantum Fisher information matrix where
parameters correspond to the 3D coordinates of the point sources. This is a
starting point from which we then transform the matrix to the parameters of
interest.

The length scale for the position of the particles is normalized w.r.t.
the Rayleigh length [zr] and wavevector [k] as follows:
x,y <-> sqrt[k/zr] x,y
z <-> z / zr

The model assumes incoherent point sources distributed in 3D space and
a Gaussian beam in the image plane as the point spread function.
With the above relative length scale this looks like:
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

from rayfish import gaussian_overlaps

# Public API
__all__ = ("delta_submat",)


def delta_submat(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.float64]:
    """Upper left submatrix for non-orthogonal coordinate transformations.

    Creates the submatrix consisting of overlaps between Gaussians based on the function
    gaussian_overlaps.delta(...).

    Args:
        coordinates: the position of individual Gaussians. Input shape is (M, 3N),
            where the N point sources should be provided in the form
            coordinates = [s1,s2,...,sN,t1,t2,...,tN,,p1,p2,...,pN] for each of the
            M >= 1 distributions of particles.
        number_of_point_sources: how many point sources we use to model the object
        wave_vector: wave vector = 2 pi / wavelength the point source emits in
        rayleigh_length: It is the distance along the propagation
            direction of a beam from the waist to the place where the area of
            1the cross section is doubled

    Returns:
        A (M,N,N)-array of complex floats, where N is the number of point sources
        and M is the number of different distributions of the N point sources.
        The diagonal for each (N,N) matraix is 1 since we assume normalized Gaussians.
    """
    delta_mat = (
        np.zeros((
            coordinates.shape[0],
            number_of_point_sources,
            number_of_point_sources,
        ))
        + 0j
    )
    for i in range(number_of_point_sources):
        for j in range(number_of_point_sources):
            delta_mat[:, i, j] = gaussian_overlaps.delta(
                s1=coordinates[:, i],
                s2=coordinates[:, j],
                t1=coordinates[:, number_of_point_sources + i],
                t2=coordinates[:, number_of_point_sources + j],
                p1=coordinates[:, 2 * number_of_point_sources + i],
                p2=coordinates[:, 2 * number_of_point_sources + j],
                k=wave_vector,
                zr=rayleigh_length,
            )
    return delta_mat

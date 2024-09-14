"""Calculating the quantum Fisher information matrix for incoherent point sources.

This module calculates the quantum Fisher information matrix for a
finite set of parameters.
The length scale for the position of the particles is normalized w.r.t.
the Rayleigh length [z_R] and wavevector [k] as follows:
x,y <-> sqrt[k/z_R] x,y
z <-> z / z_R

The model assumes incoherent point sources distributed in 3D space and
a Gaussian point spread function. With the above normalization this looks like:
ψ(x-dx, y-dy, z-dz) =
sqrt[1/pi] (i / (z-dz+i))
exp[-i ((x-dx)^2 + (y-dy)^2) / (2 (z-dz+i))] exp[-i k z_R (z-dz)]

The module implements the method proposed by Tufarelli, et.al.:
[1] Marco G Genoni and Tommaso Tufarelli, “Non-orthogonal bases for quantum metrology,”
Journal of Physics A: Mathematical and Theoretical 52, 434002 (2019).
[2] Lukas J Fiderer, Tommaso Tufarelli, Samanta Piano, and Gerardo Adesso,
“General expressions for the quantum fisher information matrix with applications
to discrete quantum imaging,” PRX Quantum 2, 020308 (2021).
"""

from __future__ import annotations

import numpy as np

# Public API
# __all__ = ("foo",)


def delta(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Calculating the overlap of two displaced Gaussian Beams."""
    prefactor = 2j / (2j - p1 + p2)
    exp_factor = (1j / (2 * (-2j + p1 - p2))) * (
        (s1 - s2) ** 2
        + (t1 - t2) ** 2
        + 2 * k * zr * (p1 - p2) ** 2
        - 1j * 4 * k * zr * (p1 - p2)
    )
    return prefactor * np.exp(exp_factor)

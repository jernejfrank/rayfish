"""Calculating inner products of displaced Gaussians.

The length scale for the position of the particles is normalized w.r.t.
the Rayleigh length [zr] and wavevector [k] as follows:
x,y <-> sqrt[k/zr] x,y
z <-> z / zr

We model the point spread function in the image plane as a
Gaussian beam. With the above relative length scale this looks like:
ψ(x-dx, y-dy, z-dz) =
sqrt[1/pi] (i / (z-dz+i))
exp[-i ((x-dx)^2 + (y-dy)^2) / (2 (z-dz+i))] exp[-i k zr (z-dz)]

We calculate the following overlaps:
delta: <ψ(x-dx_1, y-dy_1, z-dz_1)|ψ(x-dx_2, y-dy_2, z-dz_2)>
gamma_µ: <∂_µ ψ(x-dx_1, y-dy_1, z-dz_1)|ψ(x-dx_2, y-dy_2, z-dz_2)> ; µ=dx_1, dy_1, dz_1
tau_µn: <∂_µ ψ(x-dx_1, y-dy_1, z-dz_1)|∂_n ψ(x-dx_2, y-dy_2, z-dz_2)> ;
    µ=dx_1, dy_1, dz_1 ; n=dx_2, dy_2, dz_2
"""

import numpy as np

# Public API
__all__ = (
    "delta",
    "gamma_s",
    "gamma_t",
    "gamma_p",
    "tau_ss",
    "tau_st",
    "tau_sp",
    "tau_tt",
    "tau_tp",
    "tau_pp",
)


def _exp_common_factor(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    return np.exp(
        (1j / (2 * (-2j + p1 - p2)))
        * (
            (s1 - s2) ** 2
            + (t1 - t2) ** 2
            + 2 * k * zr * (p1 - p2) ** 2
            - 1j * 4 * k * zr * (p1 - p2)
        )
    )


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
    """Calculating the overlap of two displaced Gaussian beams.

    Args:
        s1: displacing first Gaussian on x-axis by x+s1
        s2: displacing second Gaussian on x-axis by x+s2
        t1: displacing first Gaussian on y-axis by y+t1
        t2: displacing second Gaussian on y-axis by y+t2
        p1: displacing first Gaussian on z-axis by z+p1
        p2: displacing second Gaussian on z-axis by z+p2
        k: wave vector = 2 pi / wavelength the point source emits in
        zr: Rayleigh length. It is the distance along the propagation
            direction of a beam from the waist to the place where the area of
            1the cross section is doubled.

    Returns:
       In general a complex number. The integral has a closed form expression
       that we hard-code here for improved performance and accuracy.
    """
    prefactor = 2j / (2j - p1 + p2)
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def gamma_s(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Overlap of a Gaussian and derivative of another Gaussian in the x-dimension.

    Calculating the overlap of a Gaussian beam and derivative of the displacement
    of the Gaussian beam in the x-dimension.

    Args:
       s1: displacing first Gaussian on x-axis by x+s1
       s2: displacing derivative of the Gaussian on x-axis by x+s2
       t1: displacing first Gaussian on y-axis by y+t1
       t2: displacing derivative of the Gaussian on y-axis by y+t2
       p1: displacing first Gaussian on z-axis by z+p1
       p2: displacing derivative of the Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           1the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = -2 * (s1 - s2) / (2j - p1 + p2) ** 2
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def gamma_t(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Overlap of a Gaussian and derivative of another Gaussian in the y-dimension.

    Calculating the overlap of a Gaussian beam and derivative of the displacement
    of the Gaussian beam in the y-dimension.

    Args:
       s1: displacing first Gaussian on x-axis by x+s1
       s2: displacing derivative of the Gaussian on x-axis by x+s2
       t1: displacing first Gaussian on y-axis by y+t1
       t2: displacing derivative of the Gaussian on y-axis by y+t2
       p1: displacing first Gaussian on z-axis by z+p1
       p2: displacing derivative of the Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           1the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = -2 * (t1 - t2) / (2j - p1 + p2) ** 2
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def gamma_p(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Overlap of a Gaussian and derivative of another Gaussian in the z-dimension.

    Calculating the overlap of a Gaussian beam and derivative of the displacement
    of the Gaussian beam in the z-dimension.

    Args:
       s1: displacing first Gaussian on x-axis by x+s1
       s2: displacing derivative of the Gaussian on x-axis by x+s2
       t1: displacing first Gaussian on y-axis by y+t1
       t2: displacing derivative of the Gaussian on y-axis by y+t2
       p1: displacing first Gaussian on z-axis by z+p1
       p2: displacing derivative of the Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           1the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = (
        -4
        + (s1 - s2) ** 2
        + (t1 - t2) ** 2
        - 2 * k * zr * (-4 + (p1 - p2) ** 2)
        + 1j * 2 * (p1 - p2) * (-1 + 4 * k * zr)
    ) / (-2j + p1 - p2) ** 3
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def tau_ss(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Overlap between Gaussian derivatives in x-dimension.

    Calculating the overlap between two Gaussian beam derivatives of the
    displacement x-dimension.

    Args:
       s1: displacing derivative of the first Gaussian on x-axis by x+s1
       s2: displacing derivative of the second Gaussian on x-axis by x+s2
       t1: displacing derivative of the first Gaussian on y-axis by y+t1
       t2: displacing derivative of the second Gaussian on y-axis by y+t2
       p1: displacing derivative of the first Gaussian on z-axis by z+p1
       p2: displacing derivative of the second Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           1the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = 2j * (-2 + (s1 - s2) ** 2 + 1j * (p2 - p1)) / (2j - p1 + p2) ** 3
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def tau_st(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Overlap between Gaussian derivatives in x- and y-dimension.

    Calculating the overlap between two derivatives of two Gaussian beams.
    The first derivative is w.r.t. to the displacement along the x-dimension,
    the second derivative is w.r.t. to the displacement along the y-dimension.

    Args:
       s1: displacing derivative of the first Gaussian on x-axis by x+s1
       s2: displacing derivative of the second Gaussian on x-axis by x+s2
       t1: displacing derivative of the first Gaussian on y-axis by y+t1
       t2: displacing derivative of the second Gaussian on y-axis by y+t2
       p1: displacing derivative of the first Gaussian on z-axis by z+p1
       p2: displacing derivative of the second Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           1the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = 2j * (s1 - s2) * (t1 - t2) / (2j - p1 + p2) ** 3
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def tau_sp(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Overlap between Gaussian derivatives in x- and z-dimension.

    Calculating the overlap between two derivatives of two Gaussian beams.
    The first derivative is w.r.t. to the displacement along the x-dimension,
    the second derivative is w.r.t. to the displacement along the z-dimension.

    Args:
       s1: displacing derivative of the first Gaussian on x-axis by x+s1
       s2: displacing derivative of the second Gaussian on x-axis by x+s2
       t1: displacing derivative of the first Gaussian on y-axis by y+t1
       t2: displacing derivative of the second Gaussian on y-axis by y+t2
       p1: displacing derivative of the first Gaussian on z-axis by z+p1
       p2: displacing derivative of the second Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           1the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = (
        -((-1 + 1j * p2) ** 3)
        * (s1 - s2)
        * (
            -8
            + (s1 - s2) ** 2
            + (t1 - t2) ** 2
            - 2 * k * zr * (-4 + (p1 - p2) ** 2)
            + 4j * (p1 - p2) * (-1 + 2 * k * zr)
        )
        / ((1j + p2) ** 3 * (2j - p1 + p2) ** 4)
    )
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def tau_tt(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Overlap between Gaussian derivatives in y-dimension.

    Calculating the overlap between two Gaussian beam derivatives of the
    displacement y-dimension.

    Args:
       s1: displacing derivative of the first Gaussian on x-axis by x+s1
       s2: displacing derivative of the second Gaussian on x-axis by x+s2
       t1: displacing derivative of the first Gaussian on y-axis by y+t1
       t2: displacing derivative of the second Gaussian on y-axis by y+t2
       p1: displacing derivative of the first Gaussian on z-axis by z+p1
       p2: displacing derivative of the second Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           1the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = 2j * (-2 + (t1 - t2) ** 2 + 1j * (p2 - p1)) / (2j - p1 + p2) ** 3
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def tau_tp(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Overlap between Gaussian derivatives in y- and z-dimension.

    Calculating the overlap between two derivatives of two Gaussian beams.
    The first derivative is w.r.t. to the displacement along the y-dimension,
    the second derivative is w.r.t. to the displacement along the z-dimension.

    Args:
       s1: displacing derivative of the first Gaussian on x-axis by x+s1
       s2: displacing derivative of the second Gaussian on x-axis by x+s2
       t1: displacing derivative of the first Gaussian on y-axis by y+t1
       t2: displacing derivative of the second Gaussian on y-axis by y+t2
       p1: displacing derivative of the first Gaussian on z-axis by z+p1
       p2: displacing derivative of the second Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           1the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = (
        -((-1 + 1j * p2) ** 3)
        * (t1 - t2)
        * (
            -8
            + (s1 - s2) ** 2
            + (t1 - t2) ** 2
            - 2 * k * zr * (-4 + (p1 - p2) ** 2)
            + 4j * (p1 - p2) * (-1 + 2 * k * zr)
        )
        / ((1j + p2) ** 3 * (2j - p1 + p2) ** 4)
    )
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def tau_pp(
    s1: float,
    s2: float,
    t1: float,
    t2: float,
    p1: float,
    p2: float,
    k: float,
    zr: float,
) -> float:
    """Overlap between Gaussian derivatives in z-dimension.

    Calculating the overlap between two Gaussian beam derivatives of the
    displacement z-dimension.

    Args:
       s1: displacing derivative of the first Gaussian on x-axis by x+s1
       s2: displacing derivative of the second Gaussian on x-axis by x+s2
       t1: displacing derivative of the first Gaussian on y-axis by y+t1
       t2: displacing derivative of the second Gaussian on y-axis by y+t2
       p1: displacing derivative of the first Gaussian on z-axis by z+p1
       p2: displacing derivative of the second Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           1the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = (1j / (2 * (-2j + p1 - p2) ** 5)) * (
        -32
        + 8 * (p1 - p2) ** 2
        - (s1 - s2) ** 2 * ((s1 - s2) ** 2 + 2 * (-8 + (t1 - t2) ** 2))
        - (t1 - t2) ** 2 * (-16 + (t1 - t2) ** 2)
        + 64 * k * zr
        + 4
        * k
        * zr
        * (
            (p1 - p2) ** 2 * (-12 + (s1 - s2) ** 2 + (t1 - t2) ** 2)
            - 4 * ((s1 - s2) ** 2 + (t1 - t2) ** 2)
        )
        - (2 * k * zr) ** 2
        * (
            (-4 + 4 * p1 + p1**2 - 2 * (2 + p1) * p2 + p2**2)
            * (-4 + p1**2 - 2 * p1 * (2 + p2) + p2 * (4 + p2))
        )
        + 1j
        * (
            8
            * (p1 - p2)
            * (
                -4
                + (s1 - s2) ** 2
                + (t1 - t2) ** 2
                - k * zr * (-12 + (p1 - p2) ** 2)
                - 2 * k * zr * ((s1 - s2) ** 2 + (t1 - t2) ** 2)
                + (2 * k * zr) ** 2 * (-4 + (p1 - p2) ** 2)
            )
        )
    )
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )

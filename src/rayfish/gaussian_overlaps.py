"""Calculating inner products of displaced Gaussians.

The length scale for the position of the particles is normalized w.r.t.
the Rayleigh length [zr] and wavevector [k] as follows:
x,y <-> sqrt[k/zr] x,y
z <-> z / zr

We model the point spread function in the image plane as a
Gaussian beam. With the above relative length scale this looks like:
ψ(x+dx, y+dy, z+dz) =
sqrt[1/pi] (i / (z+dz+i))
exp[-i ((x+dx)^2 + (y+dy)^2) / (2 (z+dz+i))] exp[-i k zr (z+dz)]

We calculate the following overlaps, which populate the Gram matrix blocks G_Δ, G_γ
and G_τ:
delta: <ψ(x+dx_1, y+dy_1, z+dz_1)|ψ(x+dx_2, y+dy_2, z+dz_2)>
gamma_µ: <∂_µ ψ(x+dx_1, y+dy_1, z+dz_1)|ψ(x+dx_2, y+dy_2, z+dz_2)> ; µ=dx_1, dy_1, dz_1
tau_µn: <∂_µ ψ(x+dx_1, y+dy_1, z+dz_1)|∂_n ψ(x+dx_2, y+dy_2, z+dz_2)> ;
    µ=dx_1, dy_1, dz_1 ; n=dx_2, dy_2, dz_2

The measurement-mode overlaps further down implement the PSF-adapted Hermite expansion
via the recursive Gaussian moments.
"""

import numpy as np
import numpy.typing as npt
from beartype.typing import Dict
from scipy.special import comb, factorial

# Public API
__all__ = (
    "delta",
    "derivate_p_inner_product_measurement_mode_point_source_overlap",
    "derivate_p_probability_measurement_mode_point_source_overlap",
    "derivate_s_inner_product_measurement_mode_point_source_overlap",
    "derivate_s_probability_measurement_mode_point_source_overlap",
    "derivate_t_inner_product_measurement_mode_point_source_overlap",
    "derivate_t_probability_measurement_mode_point_source_overlap",
    "derivative_p_probability_measurement_di_point_source_overlap",
    "derivative_s_probability_measurement_di_point_source_overlap",
    "derivative_t_probability_measurement_di_point_source_overlap",
    "gamma_p",
    "gamma_s",
    "gamma_t",
    "inner_product_measurement_mode_point_source_overlap",
    "probability_measurement_di_point_source_overlap",
    "probability_measurement_mode_point_source_overlap",
    "tau_pp",
    "tau_sp",
    "tau_ss",
    "tau_st",
    "tau_tp",
    "tau_tt",
)


def _exp_common_factor(
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
    return np.exp(
        (1j * ((s1 - s2) ** 2 + (t1 - t2) ** 2)) / (2 * (-2j + p1 - p2))
        + 1j * k * zr * (p1 - p2)
    )


def delta(
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
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
            the cross section is doubled.

    Returns:
       In general a complex number. The integral has a closed form expression
       that we hard-code here for improved performance and accuracy.
    """
    prefactor = 2j / (2j - p1 + p2)
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def gamma_s(
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
    """Overlap of a Gaussian and derivative of another Gaussian in the x-dimension.

    Calculating the overlap of a Gaussian beam and derivative of the displacement
    of the Gaussian beam in the x-dimension. The derivative acts on the first
    Gaussian.

    Args:
       s1: displacing derivative of the first Gaussian on x-axis by x+s1
       s2: displacing second Gaussian on x-axis by x+s2
       t1: displacing derivative of the first Gaussian on y-axis by y+t1
       t2: displacing second Gaussian on y-axis by y+t2
       p1: displacing derivative of the first Gaussian on z-axis by z+p1
       p2: displacing second Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = -2 * (s1 - s2) / (2j - p1 + p2) ** 2
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def gamma_t(
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
    """Overlap of a Gaussian and derivative of another Gaussian in the y-dimension.

    Calculating the overlap of a Gaussian beam and derivative of the displacement
    of the Gaussian beam in the y-dimension. The derivative acts on the first
    Gaussian.

    Args:
       s1: displacing derivative of the first Gaussian on x-axis by x+s1
       s2: displacing second Gaussian on x-axis by x+s2
       t1: displacing derivative of the first Gaussian on y-axis by y+t1
       t2: displacing second Gaussian on y-axis by y+t2
       p1: displacing derivative of the first Gaussian on z-axis by z+p1
       p2: displacing second Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = -2 * (t1 - t2) / (2j - p1 + p2) ** 2
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def gamma_p(
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
    """Overlap of a Gaussian and derivative of another Gaussian in the z-dimension.

    Calculating the overlap of a Gaussian beam and derivative of the displacement
    of the Gaussian beam in the z-dimension. The derivative acts on the first
    Gaussian.

    Args:
       s1: displacing derivative of the first Gaussian on x-axis by x+s1
       s2: displacing second Gaussian on x-axis by x+s2
       t1: displacing derivative of the first Gaussian on y-axis by y+t1
       t2: displacing second Gaussian on y-axis by y+t2
       p1: displacing derivative of the first Gaussian on z-axis by z+p1
       p2: displacing second Gaussian on z-axis by z+p2
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.

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
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
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
           the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = 2j * (-2 + (s1 - s2) ** 2 + 1j * (p2 - p1)) / (2j - p1 + p2) ** 3
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def tau_st(
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
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
           the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = 2j * (s1 - s2) * (t1 - t2) / (2j - p1 + p2) ** 3
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def tau_sp(
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
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
           the cross section is doubled.

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
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
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
           the cross section is doubled.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = 2j * (-2 + (t1 - t2) ** 2 + 1j * (p2 - p1)) / (2j - p1 + p2) ** 3
    return prefactor * _exp_common_factor(
        s1=s1, s2=s2, t1=t1, t2=t2, p1=p1, p2=p2, k=k, zr=zr
    )


def tau_tp(
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
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
           the cross section is doubled.

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
    s1: npt.NDArray[np.float64],
    s2: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    k: float,
    zr: float,
) -> npt.NDArray[np.complex128]:
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
           the cross section is doubled.

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


def _calculate_moments(
    dxy: npt.NDArray[np.float64], dz: npt.NDArray[np.float64], mode_order: int
) -> npt.NDArray[np.complex128]:
    """Helper function to recursively calculate Gaussian moments."""
    mean = -1j * dxy / (2 * 1j + dz)
    variance = (1j + dz) / (2 * 1j + dz)

    moments = np.zeros((*np.shape(dxy), mode_order), dtype=np.complex128)
    moments[..., 0] += 1

    # Catching edge case for 0-mode
    if mode_order < 2:
        return moments

    moments[..., 1] += mean
    for i in range(2, mode_order):
        moments[..., i] = (
            mean * moments[..., i - 1] + (i - 1) * variance * moments[..., i - 2]
        )

    return moments


def _calculate_moments_derivative_dxy(
    dxy: npt.NDArray[np.float64], dz: npt.NDArray[np.float64], mode_order: int
) -> npt.NDArray[np.complex128]:
    """Helper function to calculate derivative w.r.t. dx or dy of Gaussian moments."""
    mean = -1j * dxy / (2j + dz)
    dmean = -1j / (2j + dz)
    variance = (1j + dz) / (2j + dz)

    result = np.zeros_like(dxy, dtype=np.complex128)
    for j in range(mode_order + 1):
        if (mode_order - j) % 2 == 0 and j > 0:
            result += (
                comb(mode_order, j)
                * j
                * mean ** (j - 1)
                * dmean
                * variance ** ((mode_order - j) // 2)
                * (
                    factorial(mode_order - j)
                    / (2 ** ((mode_order - j) // 2) * factorial((mode_order - j) // 2))
                )
            )

    return result


def _calculate_moments_derivative_dz(
    dxy: npt.NDArray[np.float64], dz: npt.NDArray[np.float64], mode_order: int
) -> npt.NDArray[np.complex128]:
    """Helper function to calculate derivative w.r.t. dz of Gaussian moments."""
    mean = -1j * dxy / (2j + dz)
    dmean = 1j * dxy / (2j + dz) ** 2
    variance = (1j + dz) / (2 * 1j + dz)
    dvariance = 1j / (2j + dz) ** 2

    result = np.zeros_like(dxy, dtype=np.complex128)
    for j in range(0, mode_order + 1):
        if (mode_order - j) % 2 == 0:
            first_part = comb(mode_order, j) * (
                factorial(mode_order - j)
                / (2 ** ((mode_order - j) // 2) * factorial((mode_order - j) // 2))
            )
            second_part = (
                mean**j
                * ((mode_order - j) // 2)
                * variance ** (((mode_order - j) // 2) - 1)
                * dvariance
            )
            if j > 0:
                second_part += (
                    j * mean ** (j - 1) * dmean * variance ** ((mode_order - j) // 2)
                )

            result += first_part * second_part

    return result


def inner_product_measurement_mode_point_source_overlap(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    k: float,
    zr: float,
    mode_polynomial_coefficients: Dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """Calculates overlap between measurement mode and point source.

    We assume the measurement mode is of the form phi(x,y) = g_m(x) f_n(y) psi(x,y,0),
    where g_m(x) and f_n(y) are polynomials in x (resp. y) of order m (resp. n) and psi
    is the Gaussian distribution matching the point-spread function of the system.

    Args:
       s: displacing the point source on the x-axis by x+s
       t: displacing the point source on the y-axis by y+t
       p: displacing the point source on the z-axis by z+p
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.
       mode_polynomial_coefficients: Array of coefficients that defines the modes
            in the x-dim and y-dim such dim1 equals g_m(x) and dim2 equals f_n(y)
            coefficients.

    Returns:
      In general a complex number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = (
        (2 * 1j)
        / (2 * 1j + p)
        * np.exp(-1j * k * zr * p)
        * np.exp(-1j * (s**2 + t**2) / (2 * (2 * 1j + p)))
    )

    mode_x = mode_polynomial_coefficients["x"]
    mode_y = mode_polynomial_coefficients["y"]

    normal_distribution_moments_x = _calculate_moments(s, p, len(mode_x))
    normal_distribution_moments_y = _calculate_moments(t, p, len(mode_y))
    return (
        prefactor
        * np.sum(np.conj(mode_x) * normal_distribution_moments_x, axis=-1)
        * np.sum(np.conj(mode_y) * normal_distribution_moments_y, axis=-1)
    )


def probability_measurement_mode_point_source_overlap(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    k: float,
    zr: float,
    mode_polynomial_coefficients: Dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """Calculates absolute value of overlap between measurement mode and point source.

    We assume the measurement mode is of the form phi(x,y) = g_m(x) f_n(y) psi(x,y,0),
    where g_m(x) and f_n(y) are polynomials in x (resp. y) of order m (resp. n) and psi
    is the Gaussian distribution matching the point-spread function of the system.

    Args:
       s: displacing the point source on the x-axis by x+s
       t: displacing the point source on the y-axis by y+t
       p: displacing the point source on the z-axis by z+p
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.
       mode_polynomial_coefficients: Array of coefficients that defines the modes
            in the x-dim and y-dim such dim1 equals g_m(x) and dim2 equals f_n(y)
            coefficients.

    Returns:
      In general a real number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    prefactor = 4 / (4 + p**2) * np.exp(-2 * (s**2 + t**2) / (4 + p**2))

    mode_x = mode_polynomial_coefficients["x"]
    mode_y = mode_polynomial_coefficients["y"]

    normal_distribution_moments_x = _calculate_moments(s, p, len(mode_x))
    normal_distribution_moments_y = _calculate_moments(t, p, len(mode_y))
    return np.real(
        prefactor
        * np.abs(np.sum(np.conj(mode_x) * normal_distribution_moments_x, axis=-1)) ** 2
        * np.abs(np.sum(np.conj(mode_y) * normal_distribution_moments_y, axis=-1)) ** 2
    )


def derivate_s_inner_product_measurement_mode_point_source_overlap(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    k: float,
    zr: float,
    mode_polynomial_coefficients: Dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """Calculates derivative w.r.t. s of the measurement mode and point source overlap.

    We assume the measurement mode is of the form phi(x,y) = g_m(x) f_n(y) psi(x,y,0),
    where g_m(x) and f_n(y) are polynomials in x (resp. y) of order m (resp. n) and psi
    is the Gaussian distribution matching the point-spread function of the system.

    Args:
       s: displacing the point source on the x-axis by x+s
       t: displacing the point source on the y-axis by y+t
       p: displacing the point source on the z-axis by z+p
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.
       mode_polynomial_coefficients: Array of coefficients that defines the modes
            in the x-dim and y-dim such dim1 equals g_m(x) and dim2 equals f_n(y)
            coefficients.

    Returns:
      In general a complex number.
    """
    prefactor = (
        (2j / (2j + p))
        * np.exp(-1j * k * zr * p)
        * np.exp(-1j * (s**2 + t**2) / (2 * (2j + p)))
    )

    mode_x = mode_polynomial_coefficients["x"]
    mode_y = mode_polynomial_coefficients["y"]

    normal_distribution_moments_x = _calculate_moments(s, p, len(mode_x))
    normal_distribution_moments_y = _calculate_moments(t, p, len(mode_y))

    term_1_ds = (-1j * s / (2j + p)) * np.sum(
        np.conj(mode_x) * normal_distribution_moments_x, axis=-1
    )

    term_2_ds = 0
    for mode_order, ci in enumerate(mode_x):
        term_2_ds += np.conj(ci) * _calculate_moments_derivative_dxy(s, p, mode_order)

    return (
        prefactor
        * np.sum(mode_y * normal_distribution_moments_y, axis=-1)
        * (term_1_ds + term_2_ds)
    )


def derivate_t_inner_product_measurement_mode_point_source_overlap(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    k: float,
    zr: float,
    mode_polynomial_coefficients: Dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """Calculates derivative w.r.t. t of the measurement mode and point source overlap.

    We assume the measurement mode is of the form phi(x,y) = g_m(x) f_n(y) psi(x,y,0),
    where g_m(x) and f_n(y) are polynomials in x (resp. y) of order m (resp. n) and psi
    is the Gaussian distribution matching the point-spread function of the system.

    Args:
       s: displacing the point source on the x-axis by x+s
       t: displacing the point source on the y-axis by y+t
       p: displacing the point source on the z-axis by z+p
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.
       mode_polynomial_coefficients: Array of coefficients that defines the modes
            in the x-dim and y-dim such dim1 equals g_m(x) and dim2 equals f_n(y)
            coefficients.

    Returns:
      In general a complex number.
    """
    prefactor = (
        (2j / (2j + p))
        * np.exp(-1j * k * zr * p)
        * np.exp(-1j * (s**2 + t**2) / (2 * (2 * 1j + p)))
    )

    mode_x = mode_polynomial_coefficients["x"]
    mode_y = mode_polynomial_coefficients["y"]

    normal_distribution_moments_x = _calculate_moments(s, p, len(mode_x))
    normal_distribution_moments_y = _calculate_moments(t, p, len(mode_y))

    term_1_dt = (-1j * t / (2j + p)) * np.sum(
        mode_y * normal_distribution_moments_y, axis=-1
    )

    term_2_dt = 0
    for mode_order, di in enumerate(mode_y):
        term_2_dt += np.conj(di) * _calculate_moments_derivative_dxy(t, p, mode_order)

    return (
        prefactor
        * np.sum(mode_x * normal_distribution_moments_x, axis=-1)
        * (term_1_dt + term_2_dt)
    )


def derivate_p_inner_product_measurement_mode_point_source_overlap(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    k: float,
    zr: float,
    mode_polynomial_coefficients: Dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """Calculates derivative w.r.t. p of the measurement mode and point source overlap.

    We assume the measurement mode is of the form phi(x,y) = g_m(x) f_n(y) psi(x,y,0),
    where g_m(x) and f_n(y) are polynomials in x (resp. y) of order m (resp. n) and psi
    is the Gaussian distribution matching the point-spread function of the system.

    Args:
       s: displacing the point source on the x-axis by x+s
       t: displacing the point source on the y-axis by y+t
       p: displacing the point source on the z-axis by z+p
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.
       mode_polynomial_coefficients: Array of coefficients that defines the modes
            in the x-dim and y-dim such dim1 equals g_m(x) and dim2 equals f_n(y)
            coefficients.

    Returns:
      In general a complex number.
    """
    prefactor = (
        2j
        / (2j + p)
        * np.exp(-1j * k * zr * p)
        * np.exp(-1j * (s**2 + t**2) / (2 * (2j + p)))
    )

    mode_x = mode_polynomial_coefficients["x"]
    mode_y = mode_polynomial_coefficients["y"]

    normal_distribution_moments_x = _calculate_moments(s, p, len(mode_x))
    normal_distribution_moments_y = _calculate_moments(t, p, len(mode_y))

    sum_mode_x = np.sum(mode_x * normal_distribution_moments_x, axis=-1)
    sum_mode_y = np.sum(mode_y * normal_distribution_moments_y, axis=-1)

    der_non_sum_factors = (
        np.exp(-1j * k * zr * p)
        * np.exp(-1j * (s**2 + t**2) / (2 * (2j + p)))
        * (4 - s**2 - t**2 - 2j * p + 2 * k * zr * (2j + p) ** 2)
        / (2j + p) ** 3
    )

    der_sum_mode_x = 0
    for mode_order, ci in enumerate(mode_x):
        der_sum_mode_x += np.conj(ci) * _calculate_moments_derivative_dz(
            s, p, mode_order
        )

    der_sum_mode_y = 0
    for mode_order, di in enumerate(mode_y):
        der_sum_mode_y += np.conj(di) * _calculate_moments_derivative_dz(
            t, p, mode_order
        )

    return (der_non_sum_factors * sum_mode_x * sum_mode_y) + prefactor * (
        der_sum_mode_x * sum_mode_y + sum_mode_x * der_sum_mode_y
    )


def derivate_s_probability_measurement_mode_point_source_overlap(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    k: float,
    zr: float,
    mode_polynomial_coefficients: Dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """Calculates derivative w.r.t. s of the probability mode and point source overlap.

    We decompose the probability into the product of the inner product overlap and its
    complex conjugate and then take the derivative. We use the fact that derivatives
    and complex conjugation can be interchanged.

    Args:
       s: displacing the point source on the x-axis by x+s
       t: displacing the point source on the y-axis by y+t
       p: displacing the point source on the z-axis by z+p
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.
       mode_polynomial_coefficients: Array of coefficients that defines the modes
            in the x-dim and y-dim such dim1 equals g_m(x) and dim2 equals f_n(y)
            coefficients.

    Returns:
      In general a real number.
    """
    overlap_inner_product = inner_product_measurement_mode_point_source_overlap(
        s=s,
        t=t,
        p=p,
        k=k,
        zr=zr,
        mode_polynomial_coefficients=mode_polynomial_coefficients,
    )

    der_s_overlap_inner_product = (
        derivate_s_inner_product_measurement_mode_point_source_overlap(
            s=s,
            t=t,
            p=p,
            k=k,
            zr=zr,
            mode_polynomial_coefficients=mode_polynomial_coefficients,
        )
    )

    return np.real(
        (der_s_overlap_inner_product * np.conjugate(overlap_inner_product))
        + (overlap_inner_product * np.conjugate(der_s_overlap_inner_product))
    )


def derivate_t_probability_measurement_mode_point_source_overlap(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    k: float,
    zr: float,
    mode_polynomial_coefficients: Dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """Calculates derivative w.r.t. t of the probability mode and point source overlap.

    We decompose the probability into the product of the inner product overlap and its
    complex conjugate and then take the derivative. We use the fact that derivatives
    and complex conjugation can be interchanged.

    Args:
       s: displacing the point source on the x-axis by x+s
       t: displacing the point source on the y-axis by y+t
       p: displacing the point source on the z-axis by z+p
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.
       mode_polynomial_coefficients: Array of coefficients that defines the modes
            in the x-dim and y-dim such dim1 equals g_m(x) and dim2 equals f_n(y)
            coefficients.

    Returns:
      In general a real number.
    """
    overlap_inner_product = inner_product_measurement_mode_point_source_overlap(
        s=s,
        t=t,
        p=p,
        k=k,
        zr=zr,
        mode_polynomial_coefficients=mode_polynomial_coefficients,
    )

    der_t_overlap_inner_product = (
        derivate_t_inner_product_measurement_mode_point_source_overlap(
            s=s,
            t=t,
            p=p,
            k=k,
            zr=zr,
            mode_polynomial_coefficients=mode_polynomial_coefficients,
        )
    )

    return np.real(
        (der_t_overlap_inner_product * np.conjugate(overlap_inner_product))
        + (overlap_inner_product * np.conjugate(der_t_overlap_inner_product))
    )


def derivate_p_probability_measurement_mode_point_source_overlap(
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    k: float,
    zr: float,
    mode_polynomial_coefficients: Dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """Calculates derivative w.r.t. p of the probability mode and point source overlap.

    We decompose the probability into the product of the inner product overlap and its
    complex conjugate and then take the derivative. We use the fact that derivatives
    and complex conjugation can be interchanged.

    Args:
       s: displacing the point source on the x-axis by x+s
       t: displacing the point source on the y-axis by y+t
       p: displacing the point source on the z-axis by z+p
       k: wave vector = 2 pi / wavelength the point source emits in
       zr: Rayleigh length. It is the distance along the propagation
           direction of a beam from the waist to the place where the area of
           the cross section is doubled.
       mode_polynomial_coefficients: Array of coefficients that defines the modes
            in the x-dim and y-dim such dim1 equals g_m(x) and dim2 equals f_n(y)
            coefficients.

    Returns:
      In general a real number.
    """
    overlap_inner_product = inner_product_measurement_mode_point_source_overlap(
        s=s,
        t=t,
        p=p,
        k=k,
        zr=zr,
        mode_polynomial_coefficients=mode_polynomial_coefficients,
    )

    der_p_overlap_inner_product = (
        derivate_p_inner_product_measurement_mode_point_source_overlap(
            s=s,
            t=t,
            p=p,
            k=k,
            zr=zr,
            mode_polynomial_coefficients=mode_polynomial_coefficients,
        )
    )

    return np.real(
        (der_p_overlap_inner_product * np.conjugate(overlap_inner_product))
        + (overlap_inner_product * np.conjugate(der_p_overlap_inner_product))
    )


def probability_measurement_di_point_source_overlap(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates absolute value of directly measuring the point source.

    We assume the measurement mode is of the form phi(x,y) = psi(x,y,0), where psi
    is the Gaussian distribution matching the point-spread function of the system.

    This is a simplification of the generalised delta submat formula.

    Args:
       x: x grid of the object
       y: y grid of the object
       s: displacing point source on the x-axis by x+s
       t: displacing point source on the y-axis by y+t
       p: displacing point source on the z-axis by p (z=0)

    Returns:
      In general a real number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    s_, t_, p_ = (
        s[:, None, None],
        t[:, None, None],
        p[:, None, None],
    )

    prefactor = 1 / (np.pi * (1 + p_**2))
    exp = -((x + s_) ** 2 + (y + t_) ** 2) / (1 + p_**2)

    return np.real(prefactor * np.exp(exp))


def derivative_s_probability_measurement_di_point_source_overlap(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    probability: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates derivative w.r.t. s of directly measuring the point source.

    We assume the measurement mode is of the form phi(x,y) = psi(x,y,0), where psi
    is the Gaussian distribution matching the point-spread function of the system.

    Args:
       x: x grid of the object
       y: y grid of the object
       s: displacing point source on the x-axis by x+s
       t: displacing point source on the y-axis by y+t
       p: displacing point source on the z-axis by p (z=0)
       probability: the direct imaging probability.

    Returns:
      In general a real number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    s_, p_ = (
        s[:, None, None],
        p[:, None, None],
    )

    prefactor = -2 * (x + s_) / (1 + p_**2)

    return np.real(prefactor * probability)


def derivative_t_probability_measurement_di_point_source_overlap(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    probability: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates derivative w.r.t. t of directly measuring the point source..

    We assume the measurement mode is of the form phi(x,y) = psi(x,y,0), where psi
    is the Gaussian distribution matching the point-spread function of the system.

    Args:
       x: x grid of the object
       y: y grid of the object
       s: displacing point source on the x-axis by x+s
       t: displacing point source on the y-axis by y+t
       p: displacing point source on the z-axis by p (z=0)
       probability: the direct imaging probability.


    Returns:
      In general a real number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    t_, p_ = (
        t[:, None, None],
        p[:, None, None],
    )
    prefactor = -2 * (y + t_) / (1 + p_**2)

    return np.real(prefactor * probability)


def derivative_p_probability_measurement_di_point_source_overlap(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    s: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    probability: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates derivative w.r.t. p of directly measuring the point source..

    We assume the measurement mode is of the form phi(x,y) = psi(x,y,0), where psi
    is the Gaussian distribution matching the point-spread function of the system.

    Args:
       x: x grid of the object
       y: y grid of the object
       s: displacing point source on the x-axis by x+s
       t: displacing point source on the y-axis by y+t
       p: displacing point source on the z-axis by p (z=0)
       probability: the direct imaging probability.


    Returns:
      In general a real number. The integral has a closed form expression
      that we hard-code here for improved performance and accuracy.
    """
    s_, t_, p_ = (
        s[:, None, None],
        t[:, None, None],
        p[:, None, None],
    )

    prefactor = (
        -2
        * p_
        * ((1 / (1 + p_**2)) - (((x + s_) ** 2 + (y + t_) ** 2) / (1 + p_**2) ** 2))
    )

    return np.real(prefactor * probability)

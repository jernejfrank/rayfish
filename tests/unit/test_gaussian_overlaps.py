"""Testing closed-form expressions for Gaussian overlaps.

We can calculate closed-form expressions for Gaussian integrals and their derivatives
with arbitrary 3D displacement. This is beneficial since it speeds up the computation
(no integration necessary) and has higher accuracy
(analytical vs numerical calculations).

Here we make sure the closed-form expression are correct by comparing the results
to values calculated in a separate mathematica notebook ./gaussian_overlaps.nb.
"""

import numpy as np
import numpy.typing as npt

from rayfish import gaussian_overlaps


def test_delta_diagonal(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
) -> None:
    """Testing Gaussian displacement in x-dimension."""
    expected = 1
    actual = gaussian_overlaps.delta(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 0],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 2],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 4],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert expected == actual


def test_delta_in_plane_is_real(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap of two Gaussian beams in plane is real."""
    expected = 0.900865
    actual = gaussian_overlaps.delta(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=0,
        p2=0,
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected) < tolerance
    assert np.imag(actual) == 0.0


def test_delta_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap of two Gaussian beams for displacement in x-dimension."""
    expected_real = 0.869266
    expected_imag = 0.227171
    actual = gaussian_overlaps.delta(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_gamma_s_complete_overlap(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
) -> None:
    """Testing overlap of Gaussian and derivative in x is 0."""
    expected = 0
    actual = gaussian_overlaps.gamma_s(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 0],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert expected == actual


def test_gamma_s_in_plane_is_real(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap of Gaussian and derivative in x in plane is real."""
    expected = 0.270259
    actual = gaussian_overlaps.gamma_s(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=0,
        p2=0,
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected) < tolerance
    assert np.imag(actual) == 0.0


def test_gamma_s_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap of Gaussian and derivative in x."""
    expected_real = 0.264597
    expected_imag = 0.046375
    actual = gaussian_overlaps.gamma_s(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_gamma_s_conjugate(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Test derivative on first beam results in a complex conjugate."""
    expected_real = 0.264597
    expected_imag = -0.046375
    actual = np.conj(
        gaussian_overlaps.gamma_s(
            s1=mock_coordinates_scalar[0, 0],
            s2=mock_coordinates_scalar[0, 1],
            t1=mock_coordinates_scalar[0, 2],
            t2=mock_coordinates_scalar[0, 3],
            p1=mock_coordinates_scalar[0, 4],
            p2=mock_coordinates_scalar[0, 5],
            k=mock_wave_vector,
            zr=mock_rayleigh_length,
        )
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_gamma_t_complete_overlap(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
) -> None:
    """Testing overlap of Gaussian and derivative in y is 0."""
    expected = 0
    actual = gaussian_overlaps.gamma_t(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 2],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert expected == actual


def test_gamma_t_in_plane_is_real(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap of Gaussian and derivative in y in plane is real."""
    expected = -0.108104
    actual = gaussian_overlaps.gamma_t(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=0,
        p2=0,
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected) < tolerance
    assert np.imag(actual) == 0.0


def test_gamma_t_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap of Gaussian and derivative in y."""
    expected_real = -0.105839
    expected_imag = -0.01855
    actual = gaussian_overlaps.gamma_t(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_gamma_t_conjugate(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Test derivative on first beam results in a complex conjugate."""
    expected_real = -0.105839
    expected_imag = 0.01855
    actual = np.conj(
        gaussian_overlaps.gamma_t(
            s1=mock_coordinates_scalar[0, 0],
            s2=mock_coordinates_scalar[0, 1],
            t1=mock_coordinates_scalar[0, 2],
            t2=mock_coordinates_scalar[0, 3],
            p1=mock_coordinates_scalar[0, 4],
            p2=mock_coordinates_scalar[0, 5],
            k=mock_wave_vector,
            zr=mock_rayleigh_length,
        )
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_gamma_p_complete_overlap(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap of Gaussian and derivative in z is 0."""
    expected = -1.39832
    actual = gaussian_overlaps.gamma_p(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 4],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert np.real(actual) == 0.0
    assert abs(np.imag(actual) - expected) < tolerance


def test_gamma_p_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap of Gaussian and derivative in z."""
    expected_real = 0.381302
    expected_imag = -1.34393
    actual = gaussian_overlaps.gamma_p(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_gamma_p_conjugate(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Test derivative on first beam results in a complex conjugate."""
    expected_real = 0.381302
    expected_imag = 1.34393
    actual = np.conj(
        gaussian_overlaps.gamma_p(
            s1=mock_coordinates_scalar[0, 0],
            s2=mock_coordinates_scalar[0, 1],
            t1=mock_coordinates_scalar[0, 2],
            t2=mock_coordinates_scalar[0, 3],
            p1=mock_coordinates_scalar[0, 4],
            p2=mock_coordinates_scalar[0, 5],
            k=mock_wave_vector,
            zr=mock_rayleigh_length,
        )
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_ss_diagonal(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing Gaussian derivatives in x-dimension normalize to 1/2."""
    expected = 1 / 2
    actual = gaussian_overlaps.tau_ss(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 0],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 2],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 4],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected) < tolerance
    assert np.imag(actual) == 0


def test_tau_ss_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap Gaussian derivatives in x-dimension."""
    expected_real = 0.361012
    expected_imag = 0.0699617
    actual = gaussian_overlaps.tau_ss(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_ss_conjugate(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Test switching derivatives results in a complex conjugate."""
    expected_real = 0.361012
    expected_imag = -0.0699617
    actual = np.conj(
        gaussian_overlaps.tau_ss(
            s1=mock_coordinates_scalar[0, 0],
            s2=mock_coordinates_scalar[0, 1],
            t1=mock_coordinates_scalar[0, 2],
            t2=mock_coordinates_scalar[0, 3],
            p1=mock_coordinates_scalar[0, 4],
            p2=mock_coordinates_scalar[0, 5],
            k=mock_wave_vector,
            zr=mock_rayleigh_length,
        )
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_st_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing Gaussian derivatives in x- and y-dimension."""
    expected_real = 0.0319929
    expected_imag = 0.00293198
    actual = gaussian_overlaps.tau_st(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )

    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_st_conjugate(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Test switching derivatives results in a complex conjugate."""
    expected_real = 0.0319929
    expected_imag = -0.00293198
    actual = np.conj(
        gaussian_overlaps.tau_st(
            s1=mock_coordinates_scalar[0, 0],
            s2=mock_coordinates_scalar[0, 1],
            t1=mock_coordinates_scalar[0, 2],
            t2=mock_coordinates_scalar[0, 3],
            p1=mock_coordinates_scalar[0, 4],
            p2=mock_coordinates_scalar[0, 5],
            k=mock_wave_vector,
            zr=mock_rayleigh_length,
        )
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_sp_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing Gaussian derivatives in x- and z-dimension."""
    expected_real = -0.0684459
    expected_imag = 0.276513
    actual = gaussian_overlaps.tau_sp(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_sp_conjugate(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Test switching derivatives results in a complex conjugate."""
    expected_real = -0.0684459
    expected_imag = -0.276513
    actual = np.conj(
        gaussian_overlaps.tau_sp(
            s1=mock_coordinates_scalar[0, 0],
            s2=mock_coordinates_scalar[0, 1],
            t1=mock_coordinates_scalar[0, 2],
            t2=mock_coordinates_scalar[0, 3],
            p1=mock_coordinates_scalar[0, 4],
            p2=mock_coordinates_scalar[0, 5],
            k=mock_wave_vector,
            zr=mock_rayleigh_length,
        )
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_tt_diagonal(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing Gaussian derivatives in y-dimension normalize to 1/2."""
    expected = 1 / 2
    actual = gaussian_overlaps.tau_tt(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 0],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 2],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 4],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected) < tolerance
    assert np.imag(actual) == 0


def test_tau_tt_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap Gaussian derivatives in y-dimensio n."""
    expected_real = 0.428197
    expected_imag = 0.0761188
    actual = gaussian_overlaps.tau_tt(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_tt_conjugate(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Test switching derivatives results in a complex conjugate."""
    expected_real = 0.428197
    expected_imag = -0.0761188
    actual = np.conj(
        gaussian_overlaps.tau_tt(
            s1=mock_coordinates_scalar[0, 0],
            s2=mock_coordinates_scalar[0, 1],
            t1=mock_coordinates_scalar[0, 2],
            t2=mock_coordinates_scalar[0, 3],
            p1=mock_coordinates_scalar[0, 4],
            p2=mock_coordinates_scalar[0, 5],
            k=mock_wave_vector,
            zr=mock_rayleigh_length,
        )
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_tp_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing Gaussian derivatives in y- and z-dimension."""
    expected_real = 0.0273784
    expected_imag = -0.110605
    actual = gaussian_overlaps.tau_tp(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_tp_conjugate(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Test switching derivatives results in a complex conjugate."""
    expected_real = 0.0273784
    expected_imag = 0.110605
    actual = np.conj(
        gaussian_overlaps.tau_tp(
            s1=mock_coordinates_scalar[0, 0],
            s2=mock_coordinates_scalar[0, 1],
            t1=mock_coordinates_scalar[0, 2],
            t2=mock_coordinates_scalar[0, 3],
            p1=mock_coordinates_scalar[0, 4],
            p2=mock_coordinates_scalar[0, 5],
            k=mock_wave_vector,
            zr=mock_rayleigh_length,
        )
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_pp_general(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Testing overlap Gaussian derivatives in z-dimension."""
    expected_real = 2.25255
    expected_imag = 0.656208
    actual = gaussian_overlaps.tau_pp(
        s1=mock_coordinates_scalar[0, 0],
        s2=mock_coordinates_scalar[0, 1],
        t1=mock_coordinates_scalar[0, 2],
        t2=mock_coordinates_scalar[0, 3],
        p1=mock_coordinates_scalar[0, 4],
        p2=mock_coordinates_scalar[0, 5],
        k=mock_wave_vector,
        zr=mock_rayleigh_length,
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance


def test_tau_pp_conjugate(
    mock_coordinates_scalar: npt.NDArray[np.float64],
    mock_wave_vector: int,
    mock_rayleigh_length: int,
    tolerance: float,
) -> None:
    """Test switching derivatives results in a complex conjugate."""
    expected_real = 2.25255
    expected_imag = -0.656208
    actual = np.conj(
        gaussian_overlaps.tau_pp(
            s1=mock_coordinates_scalar[0, 0],
            s2=mock_coordinates_scalar[0, 1],
            t1=mock_coordinates_scalar[0, 2],
            t2=mock_coordinates_scalar[0, 3],
            p1=mock_coordinates_scalar[0, 4],
            p2=mock_coordinates_scalar[0, 5],
            k=mock_wave_vector,
            zr=mock_rayleigh_length,
        )
    )
    assert abs(np.real(actual) - expected_real) < tolerance
    assert abs(np.imag(actual) - expected_imag) < tolerance

"""Testing closed-form expressions for Gaussian overlaps.

We can calculate closed-form expressions for Gaussian integrals and their derivatives
with arbitrary 3D displacement. This is beneficial since it speeds up the computation
(no integration necessary) and has higher accuracy
(analytical vs numerical calculations).

Here we make sure the closed-form expression are correct by comparing the results
to values calculated in a separate mathematica notebook.
"""

import numpy as np

from rayfish import gaussian_overlaps

TESTSTING_CONSTANTS = {
    "s1": 1.3,
    "s2": 0.7,
    "t1": 0.93,
    "t2": 1.17,
    "p1": 0.23,
    "p2": 0.0654,
    "k": 2,
    "zr": 1,
}

TOLERANCE = 5e-6


def test_delta_diagonal() -> None:
    """Testing Gaussian displacement in x-dimension."""
    expected = 1
    actual = gaussian_overlaps.delta(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s1"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t1"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p1"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert expected == actual


def test_delta_in_plane_is_real() -> None:
    """Testing overlap of two Gaussian beams in plane is real."""
    expected = 0.900865
    actual = gaussian_overlaps.delta(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=0,
        p2=0,
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected) < TOLERANCE
    assert np.imag(actual) == 0.0


def test_delta_general() -> None:
    """Testing overlap of two Gaussian beams for displacement in x-dimension."""
    expected_real = 0.869266
    expected_imag = 0.227171
    actual = gaussian_overlaps.delta(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


def test_gamma_s_complete_overlap() -> None:
    """Testing overlap of Gaussian and derivative in x is 0."""
    expected = 0
    actual = gaussian_overlaps.gamma_s(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s1"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert expected == actual


def test_gamma_s_in_plane_is_real() -> None:
    """Testing overlap of Gaussian and derivative in x in plane is real."""
    expected = 0.270259
    actual = gaussian_overlaps.gamma_s(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=0,
        p2=0,
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected) < TOLERANCE
    assert np.imag(actual) == 0.0


def test_gamma_s_general() -> None:
    """Testing overlap of Gaussian and derivative in x."""
    expected_real = 0.264597
    expected_imag = 0.046375
    actual = gaussian_overlaps.gamma_s(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


def test_gamma_t_complete_overlap() -> None:
    """Testing overlap of Gaussian and derivative in y is 0."""
    expected = 0
    actual = gaussian_overlaps.gamma_t(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t1"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert expected == actual


def test_gamma_t_in_plane_is_real() -> None:
    """Testing overlap of Gaussian and derivative in y in plane is real."""
    expected = -0.108104
    actual = gaussian_overlaps.gamma_t(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=0,
        p2=0,
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected) < TOLERANCE
    assert np.imag(actual) == 0.0


def test_gamma_t_general() -> None:
    """Testing overlap of Gaussian and derivative in y."""
    expected_real = -0.105839
    expected_imag = -0.01855
    actual = gaussian_overlaps.gamma_t(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


def test_gamma_p_complete_overlap() -> None:
    """Testing overlap of Gaussian and derivative in z is 0."""
    expected = -1.39832
    actual = gaussian_overlaps.gamma_p(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p1"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert np.real(actual) == 0.0
    assert abs(np.imag(actual) - expected) < TOLERANCE


def test_gamma_p_general() -> None:
    """Testing overlap of Gaussian and derivative in z."""
    expected_real = 0.381302
    expected_imag = -1.34393
    actual = gaussian_overlaps.gamma_p(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


def test_tau_ss_diagonal() -> None:
    """Testing Gaussian derivatives in x-dimension normalize to 1/2."""
    expected = 1 / 2
    actual = gaussian_overlaps.tau_ss(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s1"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t1"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p1"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected) < TOLERANCE
    assert np.imag(actual) == 0


def test_tau_ss_general() -> None:
    """Testing overlap Gaussian derivatives in x-dimension."""
    expected_real = 0.361012
    expected_imag = 0.0699617
    actual = gaussian_overlaps.tau_ss(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


def test_tau_st_general() -> None:
    """Testing Gaussian derivatives in x- and y-dimension."""
    expected_real = 0.0319929
    expected_imag = 0.00293198
    actual = gaussian_overlaps.tau_st(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )

    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


def test_tau_sp_general() -> None:
    """Testing Gaussian derivatives in x- and z-dimension."""
    expected_real = -0.0684459
    expected_imag = 0.276513
    actual = gaussian_overlaps.tau_sp(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


def test_tau_tt_diagonal() -> None:
    """Testing Gaussian derivatives in y-dimension normalize to 1/2."""
    expected = 1 / 2
    actual = gaussian_overlaps.tau_tt(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s1"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t1"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p1"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected) < TOLERANCE
    assert np.imag(actual) == 0


def test_tau_tt_general() -> None:
    """Testing overlap Gaussian derivatives in y-dimension."""
    expected_real = 0.428197
    expected_imag = 0.0761188
    actual = gaussian_overlaps.tau_tt(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


def test_tau_tp_general() -> None:
    """Testing Gaussian derivatives in y- and z-dimension."""
    expected_real = 0.0273784
    expected_imag = -0.110605
    actual = gaussian_overlaps.tau_tp(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


def test_tau_pp_general() -> None:
    """Testing overlap Gaussian derivatives in z-dimension."""
    expected_real = 2.25255
    expected_imag = 0.656208
    actual = gaussian_overlaps.tau_pp(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )
    print(actual)
    assert abs(np.real(actual) - expected_real) < TOLERANCE
    assert abs(np.imag(actual) - expected_imag) < TOLERANCE


if __name__ == "__main__":
    test_delta_diagonal()
    test_delta_in_plane_is_real()
    test_delta_general()
    test_gamma_s_complete_overlap()
    test_gamma_s_in_plane_is_real()
    test_gamma_s_general()
    test_gamma_t_complete_overlap()
    test_gamma_t_in_plane_is_real()
    test_gamma_t_general()
    test_gamma_p_complete_overlap()
    test_gamma_p_general()
    test_tau_ss_diagonal()
    test_tau_ss_general()
    test_tau_st_general()
    test_tau_sp_general()
    test_tau_tt_diagonal()
    test_tau_tt_general()
    test_tau_tp_general()
    test_tau_pp_general()

"""Testing functions for incoherent sources.

We check that we are constructing the correct matrices for:
- non-orthogonal vector space linear algebra operations
- density operator
- derivative operator
- Lyapunov operator
- Quantum Fisher information

We test both the case of single point source distributions and the
vectorized version. We compare the results to values calculated in
a separate mathematica notebook ./non_orthogonal_transformation_matrix.nb.
"""

import numpy as np

import rayfish.incoherent_sources.quantum_fisher_information_matrix as qfi

WAVE_VECTOR = 2
RAYLEIGH_LENGTH = 1
NUMBER_OF_POINT_SOURCES = 2
TESTING_COORDINATES_SCALAR = np.array([[1.3, 0.7, 0.93, 1.17, 0.23, 0.0654]])
TESTING_COORDINATES_VECTOR = np.array([
    [1.3, 0.7, 0.93, 1.17, 0.23, 0.0654],
    [1.1, -1.1, 0.04, -0.04, 0.0054, -0.0054],
])

G_SCALAR = np.array([
    [
        1,
        0.869266 + 0.227171j,
        0,
        0.264597 + 0.046375j,
        0,
        -0.105839 - 0.01855j,
        -((3j) / 2),
        0.381302 - 1.34393j,
    ],
    [
        0.869266 - 0.227171j,
        1,
        -0.264597 + 0.046375j,
        0,
        0.105839 - 0.01855j,
        0,
        -0.381302 - 1.34393j,
        -((3j) / 2),
    ],
    [
        0,
        -0.264597 - 0.046375j,
        1 / 2,
        0.361012 + 0.0699617j,
        0,
        0.0319929 + 0.00293198j,
        0,
        -0.0684459 + 0.276513j,
    ],
    [
        0.264597 - 0.046375j,
        0,
        0.361012 - 0.0699617j,
        1 / 2,
        0.0319929 - 0.00293198j,
        0,
        -0.0684459 - 0.276513j,
        0,
    ],
    [
        0,
        0.105839 + 0.01855j,
        0,
        0.0319929 + 0.00293198j,
        1 / 2,
        0.428197 + 0.0761188j,
        0,
        0.0273784 - 0.110605j,
    ],
    [
        -0.105839 + 0.01855j,
        0,
        0.0319929 - 0.00293198j,
        0,
        0.428197 - 0.0761188j,
        1 / 2,
        0.0273784 + 0.110605j,
        0,
    ],
    [
        (3j) / 2,
        -0.381302 + 1.34393j,
        0,
        -0.0684459 + 0.276513j,
        0,
        0.0273784 - 0.110605j,
        5 / 2,
        2.25255 + 0.656208j,
    ],
    [
        0.381302 + 1.34393j,
        (3j) / 2,
        -0.0684459 - 0.276513j,
        0,
        0.0273784 + 0.110605j,
        0,
        2.25255 - 0.656208j,
        5 / 2,
    ],
])

G_VECTOR = np.array([
    [
        [
            1,
            0.869266 + 0.227171j,
            0,
            0.264597 + 0.046375j,
            0,
            -0.105839 - 0.01855j,
            -((3j) / 2),
            0.381302 - 1.34393j,
        ],
        [
            0.869266 - 0.227171j,
            1,
            -0.264597 + 0.046375j,
            0,
            0.105839 - 0.01855j,
            0,
            -0.381302 - 1.34393j,
            -((3j) / 2),
        ],
        [
            0,
            -0.264597 - 0.046375j,
            1 / 2,
            0.361012 + 0.0699617j,
            0,
            0.0319929 + 0.00293198j,
            0,
            -0.0684459 + 0.276513j,
        ],
        [
            0.264597 - 0.046375j,
            0,
            0.361012 - 0.0699617j,
            1 / 2,
            0.0319929 - 0.00293198j,
            0,
            -0.0684459 - 0.276513j,
            0,
        ],
        [
            0,
            0.105839 + 0.01855j,
            0,
            0.0319929 + 0.00293198j,
            1 / 2,
            0.428197 + 0.0761188j,
            0,
            0.0273784 - 0.110605j,
        ],
        [
            -0.105839 + 0.01855j,
            0,
            0.0319929 - 0.00293198j,
            0,
            0.428197 - 0.0761188j,
            1 / 2,
            0.0273784 + 0.110605j,
            0,
        ],
        [
            (3j) / 2,
            -0.381302 + 1.34393j,
            0,
            -0.0684459 + 0.276513j,
            0,
            0.0273784 - 0.110605j,
            5 / 2,
            2.25255 + 0.656208j,
        ],
        [
            0.381302 + 1.34393j,
            (3j) / 2,
            -0.0684459 - 0.276513j,
            0,
            0.0273784 + 0.110605j,
            0,
            2.25255 - 0.656208j,
            5 / 2,
        ],
    ],
    [
        [
            1,
            0.29765 + 0.00677047j,
            0,
            0.327445 + 0.00567931j,
            0,
            0.0119071 + 0.00020652j,
            -((3j) / 2),
            0.0131133 - 0.626805j,
        ],
        [
            0.29765 - 0.00677047j,
            1,
            -0.327445 + 0.00567931j,
            0,
            -0.0119071 + 0.00020652j,
            0,
            -0.0131133 - 0.626805j,
            -((3j) / 2),
        ],
        [
            0,
            -0.327445 - 0.00567931j,
            1 / 2,
            -0.211374 - 0.00172058j,
            0,
            -0.0130987 - 0.00015644j,
            0,
            -0.00874562 + 0.52581j,
        ],
        [
            0.327445 - 0.00567931j,
            0,
            -0.211374 + 0.00172058j,
            1 / 2,
            -0.0130987 + 0.00015644j,
            0,
            -0.00874562 - 0.52581j,
            0,
        ],
        [
            0,
            -0.0119071 - 0.00020652j,
            0,
            -0.0130987 - 0.00015644j,
            1 / 2,
            0.148362 + 0.00257582j,
            0,
            -0.000318023 + 0.0191204j,
        ],
        [
            0.0119071 - 0.00020652j,
            0,
            -0.0130987 + 0.00015644j,
            0,
            0.148362 - 0.00257582j,
            1 / 2,
            -0.000318023 - 0.0191204j,
            0,
        ],
        [
            (3j) / 2,
            -0.0131133 + 0.626805j,
            0,
            -0.00874562 + 0.52581j,
            0,
            -0.000318023 + 0.0191204j,
            5 / 2,
            1.21403 + 0.024914j,
        ],
        [
            0.0131133 + 0.626805j,
            (3j) / 2,
            -0.00874562 - 0.52581j,
            0,
            -0.000318023 - 0.0191204j,
            0,
            1.21403 - 0.024914j,
            5 / 2,
        ],
    ],
])

# Due to numerical precision we cannot get exact equality
TOLERANCE = 5e-6


def test_delta_submat_scalar() -> None:
    """Test constructing delta submatrix.

    Populating the delta submatrix with Gaussian overlaps
    obtained from gaussian_overlaps.delta(...). Calculation
    done for a scalar input for point source positions.
    """
    expected = G_SCALAR[:2, :2]
    actual = qfi.delta_submat(
        coordinates=TESTING_COORDINATES_SCALAR,
        number_of_point_sources=NUMBER_OF_POINT_SOURCES,
        wave_vector=WAVE_VECTOR,
        rayleigh_length=RAYLEIGH_LENGTH,
    )
    assert actual.shape == (1, 2, 2)
    assert np.allclose(actual, expected, atol=TOLERANCE)


def test_delta_submat_vector() -> None:
    """Test constructing delta submatrix.

    Populating the delta submatrix with Gaussian overlaps
    obtained from gaussian_overlaps.delta(...). Calculation
    done for a vector input for point source positions.
    """
    expected = G_VECTOR[:, :2, :2]
    actual = qfi.delta_submat(
        coordinates=TESTING_COORDINATES_VECTOR,
        number_of_point_sources=NUMBER_OF_POINT_SOURCES,
        wave_vector=WAVE_VECTOR,
        rayleigh_length=RAYLEIGH_LENGTH,
    )
    assert actual.shape == (2, 2, 2)
    assert np.allclose(actual, expected, atol=TOLERANCE)


def test_gamma_submat_scalar() -> None:
    """Test constructing gamma submatrix.

    Populating the gamma submatrix with overlaps of
    Gaussians and their derivatives obtained from
    - gaussian_overlaps.gamma_s(...) for derivative in x-dimension,
    - gaussian_overlaps.gamma_t(...) for derivative in y-dimension,
    - gaussian_overlaps.gamma_p(...) for derivative in z-dimension.

    Calculation done for a scalar input for point source positions.
    """
    expected = G_SCALAR
    expected[:2, :2] = 0.0
    expected[2:8, 2:8] = 0.0
    actual = qfi.gamma_submat(
        coordinates=TESTING_COORDINATES_SCALAR,
        number_of_point_sources=NUMBER_OF_POINT_SOURCES,
        wave_vector=WAVE_VECTOR,
        rayleigh_length=RAYLEIGH_LENGTH,
    )

    assert actual.shape == (1, 8, 8)
    assert np.allclose(actual, expected, atol=TOLERANCE)


def test_gamma_submat_vector() -> None:
    """Test constructing gamma submatrix.

    Populating the gamma submatrix with overlaps of
    Gaussians and their derivatives obtained from
    - gaussian_overlaps.gamma_s(...) for derivative in x-dimension,
    - gaussian_overlaps.gamma_t(...) for derivative in y-dimension,
    - gaussian_overlaps.gamma_p(...) for derivative in z-dimension.

    Calculation done for a vector input for point source positions.
    """
    expected = G_VECTOR
    expected[:, :2, :2] = 0.0
    expected[:, 2:8, 2:8] = 0.0
    actual = qfi.gamma_submat(
        coordinates=TESTING_COORDINATES_VECTOR,
        number_of_point_sources=NUMBER_OF_POINT_SOURCES,
        wave_vector=WAVE_VECTOR,
        rayleigh_length=RAYLEIGH_LENGTH,
    )
    assert actual.shape == (2, 8, 8)
    assert np.allclose(actual, expected, atol=TOLERANCE)


if __name__ == "__main__":
    # test_delta_submat_scalar()
    # test_delta_submat_vector()
    test_gamma_submat_scalar()
    test_gamma_submat_vector()

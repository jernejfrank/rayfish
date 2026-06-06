"""Regression tests for classical Fisher information invariants."""

import numpy as np
import numpy.typing as npt

from rayfish.classical_fisher_information.incoherent.fisher_information_matrix import (
    fisher_information_matrix_for_entire_distribution,
    probability__1D,
)
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis_x_only,
)
from rayfish.parameter_catalogue.common_transforms import (
    jacobian_1d,
    jacobian_to_collapse_xy,
    jacobian_to_correct_sign,
)
from rayfish.parameter_catalogue.distance_3_sources_xz import coordinate_jacobian


def _two_source_coordinate_jacobian(
    coordinates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    seed = np.array([[0.5, 0.5]])
    return (
        jacobian_1d(seed)
        @ jacobian_to_collapse_xy(number_of_point_sources=2)
        @ jacobian_to_correct_sign(grid=coordinates)
    )


def _total_fisher_information(
    coordinates: npt.NDArray[np.float64],
    number_of_point_sources: int,
    measurement_basis_modes: int,
    coordinate_jacobian: npt.NDArray[np.float64],
    wave_vector: float,
    rayleigh_length: float,
) -> npt.NDArray[np.float64]:
    basis = hermite_gaussian_basis_x_only.measurement_basis(
        number_of_modes=measurement_basis_modes
    )
    probability = probability__1D(
        coordinates=coordinates,
        number_of_point_sources=number_of_point_sources,
        number_of_point_source_distributions=1,
        measurement_basis=basis,
    )
    fisher_information = fisher_information_matrix_for_entire_distribution(
        coordinates=coordinates,
        number_of_point_sources=number_of_point_sources,
        number_of_point_source_distributions=1,
        measurement_basis=basis,
        probability=probability,
        wave_vector=wave_vector,
        rayleigh_length=rayleigh_length,
        coordinate_jacobian=coordinate_jacobian,
    )
    return np.sum(fisher_information, axis=1)[0]


def test_two_point_sources_x_fisher_information_is_constant() -> None:
    """The x-separation FI stays at the expected Tsang limit."""
    expected = 0.5
    matrices = []

    for displacement in (0.05, 0.1, 0.2, 0.4):
        coordinates = np.array([[-displacement, displacement, 0.0, 0.0, 0.0, 0.0]])
        matrices.append(
            _total_fisher_information(
                coordinates=coordinates,
                number_of_point_sources=2,
                measurement_basis_modes=10,
                coordinate_jacobian=_two_source_coordinate_jacobian(coordinates),
                wave_vector=2.0,
                rayleigh_length=1.0,
            )
        )

    actual = np.array([matrix[0, 0] for matrix in matrices])
    np.testing.assert_allclose(actual=actual, desired=expected, atol=1e-8)


def test_two_point_sources_z_fisher_information_is_constant() -> None:
    """The z-separation FI stays close to the small-separation limit."""
    expected = 0.125
    matrices = []

    for displacement in (0.05, 0.1):
        coordinates = np.array([[0.0, 0.0, 0.0, 0.0, displacement, -displacement]])
        matrices.append(
            _total_fisher_information(
                coordinates=coordinates,
                number_of_point_sources=2,
                measurement_basis_modes=10,
                coordinate_jacobian=_two_source_coordinate_jacobian(coordinates),
                wave_vector=2.0,
                rayleigh_length=1.0,
            )
        )

    actual = np.array([matrix[1, 1] for matrix in matrices])
    np.testing.assert_allclose(actual=actual, desired=expected, atol=3.5e-4)


def test_three_point_sources_z_information_stays_constant_when_only_x_varies() -> None:
    """The z FI is stable when only the in-plane pair moves near the origin."""
    expected = 0.1607563154810208
    matrices = []

    for displacement in (0.01, 0.02, 0.03):
        coordinates = np.array([
            [0.0, -displacement, displacement, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0]
        ])
        matrices.append(
            _total_fisher_information(
                coordinates=coordinates,
                number_of_point_sources=3,
                measurement_basis_modes=10,
                coordinate_jacobian=coordinate_jacobian(),
                wave_vector=2.0,
                rayleigh_length=1.0,
            )
        )

    actual = np.array([matrix[1, 1] for matrix in matrices])
    np.testing.assert_allclose(actual=actual, desired=expected, atol=5e-6)


def test_three_point_sources_x_information_stays_constant_when_only_z_varies() -> None:
    """The x FI stays stable for a small z-displacement sweep."""
    expected = 0.3282331528359046
    matrices = []

    for displacement in (0.01, 0.02):
        coordinates = np.array([
            [0.0, -0.3, 0.3, 0.0, 0.0, 0.0, -displacement, 0.0, 0.0]
        ])
        matrices.append(
            _total_fisher_information(
                coordinates=coordinates,
                number_of_point_sources=3,
                measurement_basis_modes=10,
                coordinate_jacobian=coordinate_jacobian(),
                wave_vector=2.0,
                rayleigh_length=1.0,
            )
        )

    actual = np.array([matrix[0, 0] for matrix in matrices])
    np.testing.assert_allclose(actual=actual, desired=expected, atol=6e-4)

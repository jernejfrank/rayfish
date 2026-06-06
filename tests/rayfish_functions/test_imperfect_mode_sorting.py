"""Tests for imperfect mode sorting in classical Fisher information."""

import numpy as np
import numpy.typing as npt

from rayfish.classical_fisher_information.incoherent import (
    fisher_information_matrix as ideal_fisher_information,
)
from rayfish.classical_fisher_information.incoherent import imperfect_mode_sorting
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis_x_only,
)
from rayfish.parameter_catalogue.common_transforms import (
    jacobian_1d,
    jacobian_to_collapse_xy,
    jacobian_to_correct_sign,
)


def _two_source_coordinate_jacobian(
    coordinates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    seed = np.array([[0.5, 0.5]])
    return (
        jacobian_1d(seed)
        @ jacobian_to_collapse_xy(number_of_point_sources=2)
        @ jacobian_to_correct_sign(grid=coordinates)
    )


def test_identity_mode_mixing_reproduces_ideal_probability_and_fi() -> None:
    """Identity mode mixing should match the ideal pipeline exactly."""
    coordinates = np.array([[-0.1, 0.1, 0.0, 0.0, 0.0, 0.0]])
    basis = hermite_gaussian_basis_x_only.measurement_basis(number_of_modes=10)
    mixing = np.eye(len(basis))

    ideal_probability = ideal_fisher_information.probability__1D(
        coordinates=coordinates,
        number_of_point_sources=2,
        number_of_point_source_distributions=1,
        measurement_basis=basis,
        wave_vector=2.0,
        rayleigh_length=1.0,
    )
    imperfect_probability = imperfect_mode_sorting.probability__1D(
        coordinates=coordinates,
        number_of_point_sources=2,
        number_of_point_source_distributions=1,
        measurement_basis=basis,
        mode_mixing_matrix=mixing,
        wave_vector=2.0,
        rayleigh_length=1.0,
    )

    np.testing.assert_allclose(actual=imperfect_probability, desired=ideal_probability)

    jacobian = _two_source_coordinate_jacobian(coordinates)
    ideal_fi = (
        ideal_fisher_information.fisher_information_matrix_for_entire_distribution(
            coordinates=coordinates,
            number_of_point_sources=2,
            number_of_point_source_distributions=1,
            measurement_basis=basis,
            probability=ideal_probability,
            wave_vector=2.0,
            rayleigh_length=1.0,
            coordinate_jacobian=jacobian,
        )
    )
    imperfect_fi = (
        imperfect_mode_sorting.fisher_information_matrix_for_entire_distribution(
            coordinates=coordinates,
            number_of_point_sources=2,
            number_of_point_source_distributions=1,
            measurement_basis=basis,
            probability=imperfect_probability,
            wave_vector=2.0,
            rayleigh_length=1.0,
            coordinate_jacobian=jacobian,
            mode_mixing_matrix=mixing,
        )
    )

    np.testing.assert_allclose(actual=imperfect_fi, desired=ideal_fi)


def test_unitary_mode_mixing_preserves_total_probability() -> None:
    """Orthogonal mode mixing should preserve total retained probability."""
    coordinates = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    basis = hermite_gaussian_basis_x_only.measurement_basis(number_of_modes=4)
    mixing = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    probability = imperfect_mode_sorting.probability__1D(
        coordinates=coordinates,
        number_of_point_sources=2,
        number_of_point_source_distributions=1,
        measurement_basis=basis,
        mode_mixing_matrix=mixing,
        wave_vector=2.0,
        rayleigh_length=1.0,
    )

    np.testing.assert_allclose(
        actual=np.sum(probability, axis=1), desired=np.array([1.0])
    )


def test_loss_channel_captures_truncation_and_derivative_conservation() -> None:
    """The loss bucket should close the probability budget and derivative sum."""
    probability = np.array([[0.7, 0.2]])
    derivatives_s = np.array([[[0.1], [0.05]]])
    derivatives_t = np.array([[[0.0], [0.02]]])
    derivatives_p = np.array([[[0.03], [0.04]]])

    (
        probability_with_loss,
        derivatives_s_with_loss,
        derivatives_t_with_loss,
        derivatives_p_with_loss,
    ) = imperfect_mode_sorting.append_loss_channel(
        probability=probability,
        derivatives_s=derivatives_s,
        derivatives_t=derivatives_t,
        derivatives_p=derivatives_p,
    )

    np.testing.assert_allclose(
        actual=np.sum(probability_with_loss, axis=1), desired=np.array([1.0])
    )
    np.testing.assert_allclose(
        actual=np.sum(derivatives_s_with_loss, axis=1), desired=np.zeros((1, 1))
    )
    np.testing.assert_allclose(
        actual=np.sum(derivatives_t_with_loss, axis=1), desired=np.zeros((1, 1))
    )
    np.testing.assert_allclose(
        actual=np.sum(derivatives_p_with_loss, axis=1), desired=np.zeros((1, 1))
    )

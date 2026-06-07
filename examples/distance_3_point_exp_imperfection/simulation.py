"""Spot-check experiment for imperfect mode sorting in the xy geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from rayfish.classical_fisher_information.incoherent import (
    fisher_information_matrix as ideal_fisher_information,
)
from rayfish.classical_fisher_information.incoherent import imperfect_mode_sorting
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis,
)
from rayfish.parameter_catalogue.distance_3_sources_xz import coordinate_jacobian

OUTPUT_DIRECTORY = Path(__file__).parent
BASELINE_DIRECTORY = OUTPUT_DIRECTORY.parent / "distance_3_point_sources_hg_xy"
WAVE_VECTOR = 2.0
RAYLEIGH_LENGTH = 1.0
NUMBER_OF_POINT_SOURCES = 3
MODE_DEGREE = 4
BATCH_SIZE = 250
N_GRIDS = 50
MINIMAL_SPACING = 0.1
MAXIMAL_LENGTH = 14.0
MIN_Z_DISPLACEMENT = 0.1
MAX_Z_DISPLACEMENT = 2.0
SAMPLE_COUNT = 6
MIXING_SEED = 20240606
EPSILONS = (0.0, 0.05, 0.1, 0.2)


def _sample_indices() -> npt.NDArray[np.int64]:
    return np.rint(np.linspace(0, N_GRIDS - 1, SAMPLE_COUNT)).astype(int)


def _sampled_spacing_values() -> np.ndarray:
    full_spacing = np.linspace(
        MINIMAL_SPACING,
        MAXIMAL_LENGTH / NUMBER_OF_POINT_SOURCES,
        N_GRIDS,
        endpoint=True,
    )
    return full_spacing[_sample_indices()]


def _sampled_z_values() -> np.ndarray:
    full_z = np.linspace(
        MIN_Z_DISPLACEMENT,
        MAX_Z_DISPLACEMENT,
        N_GRIDS,
        endpoint=True,
    )
    return full_z[_sample_indices()]


def _sampled_coordinates() -> np.ndarray:
    coordinates = []
    for spacing in _sampled_spacing_values():
        for z_displacement in _sampled_z_values():
            coordinates.append([
                0.0,
                -spacing,
                spacing,
                0.0,
                0.0,
                0.0,
                -z_displacement,
                0.0,
                0.0,
            ])
    return np.array(coordinates, dtype=np.float64)


def _fixed_skew_symmetric_generator(size: int) -> np.ndarray:
    rng = np.random.default_rng(MIXING_SEED)
    raw = rng.normal(size=(size, size))
    skew = raw - raw.T
    scale = np.linalg.norm(skew, ord=2)
    if scale == 0:
        raise ValueError("Degenerate skew-symmetric generator.")
    return skew / scale


def mode_mixing_matrix(number_of_modes: int, epsilon: float) -> np.ndarray:
    """Deterministic orthogonal mixing family with angle proportional to epsilon."""
    generator = _fixed_skew_symmetric_generator(number_of_modes)
    return expm(epsilon * generator)


def _batched_fi(
    coordinates: np.ndarray,
    mode_mixing: np.ndarray,
    include_loss_channel: bool,
) -> np.ndarray:
    basis = hermite_gaussian_basis.measurement_basis(mode_degree=MODE_DEGREE)
    fi_batches = []

    for start in range(0, coordinates.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, coordinates.shape[0])
        coordinate_batch = coordinates[start:stop]
        probability = imperfect_mode_sorting.probability__1D(
            coordinates=coordinate_batch,
            number_of_point_sources=NUMBER_OF_POINT_SOURCES,
            number_of_point_source_distributions=coordinate_batch.shape[0],
            measurement_basis=basis,
            mode_mixing_matrix=mode_mixing,
            include_loss_channel=include_loss_channel,
            wave_vector=WAVE_VECTOR,
            rayleigh_length=RAYLEIGH_LENGTH,
        )
        fi_batches.append(
            imperfect_mode_sorting.fisher_information_matrix_for_entire_distribution(
                coordinates=coordinate_batch,
                number_of_point_sources=NUMBER_OF_POINT_SOURCES,
                number_of_point_source_distributions=coordinate_batch.shape[0],
                measurement_basis=basis,
                probability=probability,
                wave_vector=WAVE_VECTOR,
                rayleigh_length=RAYLEIGH_LENGTH,
                coordinate_jacobian=coordinate_jacobian(),
                mode_mixing_matrix=mode_mixing,
                include_loss_channel=include_loss_channel,
            )
        )

    return np.concatenate(fi_batches, axis=0)


def _batched_ideal_fi(coordinates: np.ndarray) -> np.ndarray:
    basis = hermite_gaussian_basis.measurement_basis(mode_degree=MODE_DEGREE)
    fi_batches = []

    for start in range(0, coordinates.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, coordinates.shape[0])
        coordinate_batch = coordinates[start:stop]
        probability = ideal_fisher_information.probability__1D(
            coordinates=coordinate_batch,
            number_of_point_sources=NUMBER_OF_POINT_SOURCES,
            number_of_point_source_distributions=coordinate_batch.shape[0],
            measurement_basis=basis,
            wave_vector=WAVE_VECTOR,
            rayleigh_length=RAYLEIGH_LENGTH,
        )
        fi_batches.append(
            ideal_fisher_information.fisher_information_matrix_for_entire_distribution(
                coordinates=coordinate_batch,
                number_of_point_sources=NUMBER_OF_POINT_SOURCES,
                number_of_point_source_distributions=coordinate_batch.shape[0],
                measurement_basis=basis,
                probability=probability,
                wave_vector=WAVE_VECTOR,
                rayleigh_length=RAYLEIGH_LENGTH,
                coordinate_jacobian=coordinate_jacobian(),
            )
        )

    return np.concatenate(fi_batches, axis=0)


def _expected_classical_slices() -> tuple[np.ndarray, np.ndarray]:
    sample_indices = _sample_indices()
    expected_fi = np.load(BASELINE_DIRECTORY / "fi_ms.npy")[
        np.ix_(sample_indices, sample_indices)
    ].reshape(SAMPLE_COUNT * SAMPLE_COUNT, MODE_DEGREE * MODE_DEGREE, 2, 2)
    expected_cov = np.load(BASELINE_DIRECTORY / "fi_ms_inv.npy")[
        np.ix_(sample_indices, sample_indices)
    ].reshape(SAMPLE_COUNT * SAMPLE_COUNT, 2, 2)
    return expected_fi, expected_cov


def verify_no_mixing_reproduces_baseline() -> None:
    """Check the imperfect path matches the saved baseline when mixing is disabled."""
    coordinates = _sampled_coordinates()
    basis_size = MODE_DEGREE * MODE_DEGREE
    identity = np.eye(basis_size)

    imperfect_fi = _batched_fi(
        coordinates=coordinates,
        mode_mixing=identity,
        include_loss_channel=False,
    )
    ideal_fi = _batched_ideal_fi(coordinates=coordinates)
    imperfect_cov = ideal_fisher_information.covariance_matrix_for_entire_distribution(
        imperfect_fi
    )

    expected_fi, expected_cov = _expected_classical_slices()
    np.testing.assert_allclose(
        actual=imperfect_fi, desired=ideal_fi, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        actual=imperfect_fi, desired=expected_fi, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        actual=imperfect_cov, desired=expected_cov, rtol=1e-10, atol=1e-10
    )


def summarize_mixing_strengths() -> None:
    """Print a lightweight FI summary for deterministic mixing strengths."""
    coordinates = _sampled_coordinates()
    baseline_fi = np.sum(_batched_ideal_fi(coordinates=coordinates), axis=1)
    baseline_scalar = baseline_fi[:, 0, 0]

    for epsilon in EPSILONS:
        mixing = mode_mixing_matrix(MODE_DEGREE * MODE_DEGREE, epsilon)
        imperfect_fi = np.sum(
            _batched_fi(
                coordinates=coordinates,
                mode_mixing=mixing,
                include_loss_channel=False,
            ),
            axis=1,
        )
        ratio = imperfect_fi[:, 0, 0] / baseline_scalar
        print(
            f"epsilon={epsilon:.2f}: "
            f"mean(FI_xx / FI_xx_ideal)={np.mean(ratio):.12f}"
        )


def main() -> None:
    """Run the reduced no-mixing reproduction check and summarize fixed rotations."""
    verify_no_mixing_reproduces_baseline()
    summarize_mixing_strengths()
    print("Verified imperfect xy spot-check against the saved no-mixing baseline.")


if __name__ == "__main__":
    main()

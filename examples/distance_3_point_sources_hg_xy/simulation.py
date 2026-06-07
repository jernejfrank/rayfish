"""Baseline checker for the 3-point-source HG xy simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hamilton import driver
from hamilton.execution import executors

import rayfish
import rayfish.quantum_fisher_information as qfi
from rayfish.classical_fisher_information import (
    incoherent as classical_incoherent_sources,
)
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis,
)
from rayfish.di_fisher_information import incoherent as di_incoherent_sources
from rayfish.parameter_catalogue import distance_3_sources_xz
from rayfish.quantum_fisher_information import (
    incoherent_sources as qfi_incoherent_sources,
)

OUTPUT_DIRECTORY = Path(__file__).parent
CONFIG = {"simulation_dimension": 1}
N_GRIDS = 50
MODE_DEGREE = 4
INPUTS = {
    "wave_vector": 2.0,
    "rayleigh_length": 1.0,
    "number_of_grids": N_GRIDS,
    "maximal_length": 14.0,
    "minimal_spacing": 0.1,
    "min_z_displacement": 0.1,
    "max_z_displacement": 2.0,
    "mode_degree": MODE_DEGREE,
    "numerical_integration_resolution": 12,
    "numerical_integration_support": 5.0,
    "batch_size": 250,
}


def _build_quantum_driver() -> driver.Driver:
    return (
        driver.Builder()
        .with_modules(rayfish, qfi, qfi_incoherent_sources, distance_3_sources_xz)
        .with_config(config=CONFIG)
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_local_executor(executors.SynchronousLocalTaskExecutor())
    ).build()


def _build_classical_driver() -> driver.Driver:
    return (
        driver.Builder()
        .with_modules(
            rayfish,
            classical_incoherent_sources,
            distance_3_sources_xz,
            hermite_gaussian_basis,
        )
        .with_config(config=CONFIG)
    ).build()


def _build_di_driver() -> driver.Driver:
    return (
        driver.Builder()
        .with_modules(
            rayfish,
            di_incoherent_sources,
            distance_3_sources_xz,
        )
        .with_config(config=CONFIG)
    ).build()


def compute_reference_arrays() -> dict[str, np.ndarray]:
    """Compute the same arrays that are stored next to the original notebook."""
    print("Computing QFI reference arrays...")
    quantum_result = _build_quantum_driver().execute(
        inputs=INPUTS,
        final_vars=[
            "fisher_information_matrix_for_entire_distribution",
            "covariance_matrix_for_entire_distribution",
        ],
    )
    print("Computing classical HG reference arrays...")
    classical_result = _build_classical_driver().execute(
        inputs=INPUTS,
        final_vars=[
            "fisher_information_matrix_for_entire_distribution",
            "covariance_matrix_for_entire_distribution",
        ],
    )
    print("Computing direct-imaging reference arrays...")
    di_result = _build_di_driver().execute(
        inputs=INPUTS,
        final_vars=[
            "fisher_information_matrix_for_entire_distribution",
            "covariance_matrix_for_entire_distribution",
        ],
    )

    return {
        "qfi.npy": np.real(
            quantum_result["fisher_information_matrix_for_entire_distribution"].reshape(
                N_GRIDS, N_GRIDS, 2, 2
            )
        ),
        "qfi_inv.npy": np.real(
            quantum_result["covariance_matrix_for_entire_distribution"].reshape(
                N_GRIDS, N_GRIDS, 2, 2
            )
        ),
        "fi_ms.npy": classical_result[
            "fisher_information_matrix_for_entire_distribution"
        ].reshape(N_GRIDS, N_GRIDS, MODE_DEGREE * MODE_DEGREE, 2, 2),
        "fi_ms_inv.npy": classical_result[
            "covariance_matrix_for_entire_distribution"
        ].reshape(N_GRIDS, N_GRIDS, 2, 2),
        "fi_di.npy": di_result[
            "fisher_information_matrix_for_entire_distribution"
        ].reshape(N_GRIDS, N_GRIDS, 2, 2),
        "fi_di_inv.npy": di_result["covariance_matrix_for_entire_distribution"].reshape(
            N_GRIDS, N_GRIDS, 2, 2
        ),
    }


def verify_saved_arrays(reference_arrays: dict[str, np.ndarray]) -> None:
    """Assert the in-memory computation matches the tracked baseline arrays."""
    for filename, actual in reference_arrays.items():
        expected = np.load(OUTPUT_DIRECTORY / filename)
        np.testing.assert_allclose(
            actual=actual,
            desired=expected,
            rtol=1e-10,
            atol=1e-10,
        )
        print(f"Verified {filename}.")


def main() -> None:
    """Run the baseline check without overwriting the stored arrays."""
    reference_arrays = compute_reference_arrays()
    verify_saved_arrays(reference_arrays)
    print("Verified in-memory XY baseline against tracked .npy files.")


if __name__ == "__main__":
    main()

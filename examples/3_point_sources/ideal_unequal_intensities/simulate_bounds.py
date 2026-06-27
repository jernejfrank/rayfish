"""Keeping simulation concise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hamilton import driver
from hamilton.execution import executors
from loguru import logger

import rayfish
import rayfish.quantum_fisher_information as qfi
from rayfish.classical_fisher_information import fisher_information_remainder_channel
from rayfish.di_fisher_information import (
    incoherent as di_incoherent_sources,
)
from rayfish.parameter_catalogue import distance_3_sources_xz, intensity_distribution
from rayfish.quantum_fisher_information import (
    incoherent_sources as qfi_incoherent_sources,
)

ROOT_DIR = Path(__file__).parent

N_PT_SRC = 3
CONTRAST_EPS = [0.0, 0.1, 0.4, 0.7]
CONF = {"simulation_dimension": 1, "filter_modes": True, "intensity": "contrast"}
# inputs
N_GRIDS = 50
MODE_DEGREE = 4
IMPORTANT_MODES = ((0, 0), (1, 0), (2, 0), (0, 2))
MIDDLE_SOURCE_IDX = 0
INPUT_ = {
    "wave_vector": 2.0,
    "rayleigh_length": 1.0,
    "number_of_grids": N_GRIDS,  # 50,
    "maximal_length": 14.0,
    "minimal_spacing": 0.1,
    "min_z_displacement": 0.1,
    "max_z_displacement": 2.0,
    "mode_degree": MODE_DEGREE,
    "filter_mode_set": IMPORTANT_MODES,
    "numerical_integration_resolution": 12,
    "numerical_integration_support": 5.0,
    "batch_size": 250,
    "target_point_source_idx": MIDDLE_SOURCE_IDX,
}


def simulate_bounds() -> None:
    """Simulates FI for MS and saves to file."""
    for _eps in CONTRAST_EPS:
        for prefactor in [1, -1]:
            if _eps == 0.0 and prefactor == -1:
                continue

            contrast = 1 + prefactor * _eps
            logger.info(f"Simulation for contrast eps {contrast:.2f}")
            input_ = dict(INPUT_.items())
            input_["intensity_contrast"] = contrast

            logger.info("Simulating MS FI...")
            classical_result = classical_dr.execute(
                inputs=input_,
                final_vars=[
                    "fisher_information_matrix_for_entire_distribution",
                    "covariance_matrix_for_entire_distribution",
                    "probability",
                ],
            )

            fi_ms_ = classical_result[
                "fisher_information_matrix_for_entire_distribution"
            ].reshape(N_GRIDS, N_GRIDS, len(IMPORTANT_MODES) + 1, 2, 2)
            np.save(
                ROOT_DIR / f"fi_ms_{contrast:.2f}.npy",
                fi_ms_,
            )

            fi_ms_inv_ = classical_result[
                "covariance_matrix_for_entire_distribution"
            ].reshape(N_GRIDS, N_GRIDS, 2, 2)
            np.save(
                ROOT_DIR / f"fi_ms_inv_{contrast:.2f}.npy",
                fi_ms_inv_,
            )

            prob = classical_result["probability"].reshape(
                N_GRIDS, N_GRIDS, len(IMPORTANT_MODES)
            )
            np.save(
                ROOT_DIR / f"leaked_fraction_{contrast:.2f}.npy",
                np.clip(1 - prob.sum(axis=-1), 0, 1),
            )

            logger.info("Simulating QFI...")
            quantum_result = quantum_dr.execute(
                inputs=input_,
                final_vars=[
                    "fisher_information_matrix_for_entire_distribution",
                    "covariance_matrix_for_entire_distribution",
                ],
            )

            qfi_ = np.real(
                quantum_result[
                    "fisher_information_matrix_for_entire_distribution"
                ].reshape(N_GRIDS, N_GRIDS, 2, 2)
            )
            np.save(
                ROOT_DIR / f"qfi_{contrast:.2f}.npy",
                qfi_,
            )
            qfi_inv_ = np.real(
                quantum_result["covariance_matrix_for_entire_distribution"].reshape(
                    N_GRIDS, N_GRIDS, 2, 2
                )
            )
            np.save(
                ROOT_DIR / f"qfi_inv_{contrast:.2f}.npy",
                qfi_inv_,
            )

            logger.info("Simulating DI...")
            di_result = di_dr.execute(
                inputs=input_,
                final_vars=[
                    "fisher_information_matrix_for_entire_distribution",
                    "covariance_matrix_for_entire_distribution",
                ],
            )

            fi_di_ = di_result[
                "fisher_information_matrix_for_entire_distribution"
            ].reshape(N_GRIDS, N_GRIDS, 2, 2)
            np.save(
                ROOT_DIR / f"fi_di_{contrast:.2f}.npy",
                fi_di_,
            )

            fi_di_inv_ = di_result["covariance_matrix_for_entire_distribution"].reshape(
                N_GRIDS, N_GRIDS, 2, 2
            )
            np.save(
                ROOT_DIR / f"fi_di_inv_{contrast:.2f}.npy",
                fi_di_inv_,
            )


if __name__ == "__main__":
    classical_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            fisher_information_remainder_channel,
            distance_3_sources_xz,
            intensity_distribution,
        )
        .with_config(config=CONF)
    ).build()

    quantum_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            qfi,
            qfi_incoherent_sources,
            distance_3_sources_xz,
            intensity_distribution,
        )
        .with_config(config=CONF)
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_local_executor(executors.SynchronousLocalTaskExecutor())
    ).build()

    di_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            di_incoherent_sources,
            distance_3_sources_xz,
            intensity_distribution,
        )
        .with_config(config=CONF)
    ).build()

    logger.info("Simulating...")
    simulate_bounds()
    logger.info("Simulation finished.")

    coords_result = classical_dr.execute(
        inputs=INPUT_,
        final_vars=[
            "coordinates",
        ],
    )
    coords = coords_result["coordinates"].reshape(N_GRIDS, N_GRIDS, 9)
    np.save(
        ROOT_DIR / "coordinates.npy",
        coords,
    )
    logger.info("Coords saved.")

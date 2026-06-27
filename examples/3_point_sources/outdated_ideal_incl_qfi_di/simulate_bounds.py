"""Keeping simulation concise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hamilton import driver
from hamilton.execution import executors
from loguru import logger

import rayfish
import rayfish.quantum_fisher_information as qfi
from rayfish.parameter_catalogue import distance_3_sources_xz, intensity_distribution
from rayfish.quantum_fisher_information import (
    incoherent_sources as qfi_incoherent_sources,
)

ROOT_DIR = Path(__file__).parent

N_PT_SRC = 3
CONF = {"simulation_dimension": 1}
# inputs
N_GRIDS = 50
INPUT_ = {
    "wave_vector": 2.0,
    "rayleigh_length": 1.0,
    "number_of_grids": N_GRIDS,  # 50,
    "maximal_length": 14.0,
    "minimal_spacing": 0.1,
    "min_z_displacement": 0.1,
    "max_z_displacement": 2.0,
}


def simulate_bounds() -> None:
    """Simulates FI for MS and saves to file."""
    input_ = dict(INPUT_.items())
    logger.info("Simulating QFI...")
    quantum_result = quantum_dr.execute(
        inputs=input_,
        final_vars=[
            "fisher_information_matrix_for_entire_distribution",
            "covariance_matrix_for_entire_distribution",
        ],
    )

    qfi_ = np.real(
        quantum_result["fisher_information_matrix_for_entire_distribution"].reshape(
            N_GRIDS, N_GRIDS, 2, 2
        )
    )
    np.save(
        ROOT_DIR / "new_qfi.npy",
        qfi_,
    )
    qfi_inv_ = np.real(
        quantum_result["covariance_matrix_for_entire_distribution"].reshape(
            N_GRIDS, N_GRIDS, 2, 2
        )
    )
    np.save(
        ROOT_DIR / "new_qfi_inv.npy",
        qfi_inv_,
    )


if __name__ == "__main__":
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

    logger.info("Simulating...")
    simulate_bounds()
    logger.info("Simulation finished.")

    coords_result = quantum_dr.execute(
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

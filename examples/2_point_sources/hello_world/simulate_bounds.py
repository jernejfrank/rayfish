"""Hello-world: transverse separation of two incoherent point sources (Tsang).

Two incoherent point sources separated along the x-axis and kept in focus. For each
separation we compute the Fisher information for the separation with two measurements:

- QFI:   the quantum bound (optimal measurement).
- FI-MS: a Hermite-Gaussian mode-sorting measurement (SPADE).

Tsang's result [PRX 6, 031033 (2016)]: the QFI stays flat as the separation shrinks
(no Rayleigh's curse), and mode sorting *saturates* it -- FI-MS = QFI at every
separation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hamilton import driver
from hamilton.execution import executors
from loguru import logger

import rayfish
from rayfish.classical_fisher_information import incoherent_mode_sorting
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis,
)
from rayfish.parameter_catalogue import distance_2_sources_x
from rayfish.quantum_fisher_information import (
    common as qfi_common,
)
from rayfish.quantum_fisher_information import (
    lyapunov_operator,
    operator_matrices,
)
from rayfish.quantum_fisher_information import (
    quantum_fisher_information as qfi_matrix,
)

ROOT_DIR = Path(__file__).parent

N_PT_SRC = 2
N_GRIDS = 50
MODE_DEGREE = 6
CONF = {"simulation_dimension": 1, "filter_modes": False}
INPUT_ = {
    "wave_vector": 2.0,
    "rayleigh_length": 1.0,
    "number_of_grids": N_GRIDS,
    "maximal_length": 4.0,
    "minimal_spacing": 0.1,
    "point_source_intensities": [0.5, 0.5],
    "mode_degree": MODE_DEGREE,
    "mode_mixing_eps": 0.0,
}


def simulate_bounds() -> None:
    """Compute and save the QFI and mode-sorting FI for the separation."""
    logger.info("Simulating QFI...")
    quantum_result = quantum_dr.execute(
        inputs=INPUT_,
        final_vars=[
            "fisher_information_matrix_for_entire_distribution",
            "spacing",
        ],
    )
    # (M, 1, 1): Fisher information for the single separation parameter.
    qfi_ = np.real(
        quantum_result["fisher_information_matrix_for_entire_distribution"]
    ).reshape(N_GRIDS)
    np.save(ROOT_DIR / "qfi.npy", qfi_)

    logger.info("Simulating mode-sorting FI...")
    classical_result = classical_dr.execute(
        inputs=INPUT_,
        final_vars=["fisher_information_matrix_for_entire_distribution"],
    )
    # (M, B+1, 1, 1): per-channel FI incl. the remainder bucket. Sum over channels for
    # the total mode-sorting FI.
    fi_ms_ = (
        np.real(classical_result["fisher_information_matrix_for_entire_distribution"])
        .sum(axis=1)
        .reshape(N_GRIDS)
    )
    np.save(ROOT_DIR / "fi_ms.npy", fi_ms_)

    np.save(ROOT_DIR / "spacing.npy", quantum_result["spacing"])
    logger.info("Saved qfi.npy, fi_ms.npy, spacing.npy")


if __name__ == "__main__":
    quantum_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            qfi_common,
            operator_matrices,
            lyapunov_operator,
            qfi_matrix,
            distance_2_sources_x,
        )
        .with_config(config=CONF)
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_local_executor(executors.SynchronousLocalTaskExecutor())
    ).build()

    classical_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            incoherent_mode_sorting,
            distance_2_sources_x,
            hermite_gaussian_basis,
        )
        .with_config(config=CONF)
    ).build()

    logger.info("Simulating...")
    simulate_bounds()
    logger.info("Simulation finished.")

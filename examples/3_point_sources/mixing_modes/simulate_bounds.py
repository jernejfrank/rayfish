"""Keeping simulation concise."""

from pathlib import Path

import numpy as np
from hamilton import driver
from loguru import logger

import rayfish
from rayfish.classical_fisher_information import (
    incoherent_mode_sorting as fisher_information_remainder_channel,
)
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis,
)
from rayfish.parameter_catalogue import distance_3_sources_xz, intensity_distribution

ROOT_DIR = Path(__file__).parent

N_PT_SRC = 3
EPS_INTERVAL = [0.0, 0.04, 0.07, 0.1, 0.3, 0.6, 1.0]
CONF = {"simulation_dimension": 1, "filter_modes": True}
# inputs
N_GRIDS = 50
MODE_DEGREE = 4
IMPORTANT_MODES = ((0, 0), (1, 0), (2, 0), (0, 2))
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
}


def simulate_bounds() -> None:
    """Simulates FI for MS and saves to file."""
    for mixing_eps in EPS_INTERVAL:
        logger.info(f"Simulation for eps {mixing_eps:.2f}")
        input_ = dict(INPUT_.items())
        input_["mode_mixing_eps"] = mixing_eps
        classical_result = classical_dr.execute(
            inputs=input_,
            final_vars=[
                "fisher_information_matrix_for_entire_distribution",
                "covariance_matrix_for_entire_distribution",
                "number_of_point_sources",
                "spacing",
                "coordinates",
                "measurement_basis",
            ],
        )
        fi_ms_ = classical_result[
            "fisher_information_matrix_for_entire_distribution"
        ].reshape(N_GRIDS, N_GRIDS, len(IMPORTANT_MODES) + 1, 2, 2)
        np.save(
            ROOT_DIR / f"fi_ms_eps_{mixing_eps:.2f}.npy",
            fi_ms_,
        )

        fi_ms_inv_ = classical_result[
            "covariance_matrix_for_entire_distribution"
        ].reshape(N_GRIDS, N_GRIDS, 2, 2)
        np.save(
            ROOT_DIR / f"fi_ms_inv_eps_{mixing_eps:.2f}.npy",
            fi_ms_inv_,
        )


if __name__ == "__main__":
    classical_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            fisher_information_remainder_channel,
            distance_3_sources_xz,
            intensity_distribution,
            hermite_gaussian_basis,
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
        "coordinates.npy",
        coords,
    )

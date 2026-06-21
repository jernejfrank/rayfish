"""Keeping simulation concise."""

import numpy as np
from hamilton import driver
from loguru import logger

import rayfish
from rayfish.classical_fisher_information import (
    incoherent as classical_incoherent_sources,
)
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis,
)
from rayfish.parameter_catalogue import distance_3_sources_xz

N_PT_SRC = 3
EPS_INTERVAL = [0.0, 0.1, 0.3, 0.6, 1.0]
CONF = {
    "simulation_dimension": 1,
}
# inputs
N_GRIDS = 50
MODE_DEGREE = 4
INPUT_ = {
    "wave_vector": 2.0,
    "rayleigh_length": 1.0,
    "number_of_grids": N_GRIDS,  # 50,
    "maximal_length": 14.0,
    "minimal_spacing": 0.1,
    "min_z_displacement": 0.1,
    "max_z_displacement": 2.0,
    "mode_degree": MODE_DEGREE,
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
        ].reshape(N_GRIDS, N_GRIDS, MODE_DEGREE * MODE_DEGREE, 2, 2)
        np.save(
            f"fi_ms_eps_{mixing_eps:.2f}.npy",
            fi_ms_,
        )

        fi_ms_inv_ = classical_result[
            "covariance_matrix_for_entire_distribution"
        ].reshape(N_GRIDS, N_GRIDS, 2, 2)
        np.save(
            f"fi_ms_inv_eps_{mixing_eps:.2f}.npy",
            fi_ms_inv_,
        )


if __name__ == "__main__":
    classical_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            classical_incoherent_sources,
            distance_3_sources_xz,
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

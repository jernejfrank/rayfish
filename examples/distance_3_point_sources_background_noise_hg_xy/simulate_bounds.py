"""Keeping simulation concise."""

from __future__ import annotations

from math import factorial
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype.typing import Any, Dict, List
from hamilton import driver
from loguru import logger
from scipy.special import hermite

import rayfish
from rayfish.classical_fisher_information import fisher_information_remainder_channel
from rayfish.parameter_catalogue import distance_3_sources_xz

ROOT_DIR = Path(__file__).parent

N_PT_SRC = 3
CONF: dict[str, Any] = {"simulation_dimension": 1}
# inputs
N_GRIDS = 50
IMPORTANT_MODES = ((0, 0), (1, 0), (2, 0), (0, 2))
INPUT_: dict[str, object] = {
    "wave_vector": 2.0,
    "rayleigh_length": 1.0,
    "number_of_grids": N_GRIDS,  # 50,
    "maximal_length": 14.0,
    "minimal_spacing": 0.1,
    "min_z_displacement": 0.1,
    "max_z_displacement": 2.0,
    "numerical_integration_resolution": 12,
    "numerical_integration_support": 5.0,
    "batch_size": 250,
}
BETA_INTERVAL = [0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def _measurement_basis(
    modes: tuple[tuple[int, int], ...] = (),
) -> List[Dict[str, npt.NDArray[np.float64]]]:
    """Encodes the coefficients of the Hermite-Gaussian modes."""
    basis = []
    for mode in modes:
        (mode_x, mode_y) = mode

        b = {}
        hg_x = hermite(mode_x).coeffs[::-1] / np.sqrt(2**mode_x * factorial(mode_x))
        hg_y = hermite(mode_y).coeffs[::-1] / np.sqrt(2**mode_y * factorial(mode_y))

        b["x"] = hg_x
        b["y"] = hg_y
        basis.append(b)
    return basis


def simulate_bounds() -> None:
    """Simulates FI for MS and saves to file."""
    for i in BETA_INTERVAL:
        logger.info(f"Simulation for background noise {i:.0e}")
        input_ = dict(INPUT_.items())
        input_["measurement_basis"] = _measurement_basis(IMPORTANT_MODES)
        background_noise_modes = [i for _ in IMPORTANT_MODES]
        bucket_background_noise = i
        input_["background_noise"] = (
            *background_noise_modes,
            bucket_background_noise,
        )
        classical_result = classical_dr.execute(
            inputs=input_,
            final_vars=[
                "fisher_information_matrix_for_entire_distribution",
                "covariance_matrix_for_entire_distribution",
            ],
        )
        fi_ms_ = classical_result[
            "fisher_information_matrix_for_entire_distribution"
        ].reshape(N_GRIDS, N_GRIDS, len(IMPORTANT_MODES) + 1, 2, 2)
        np.save(
            ROOT_DIR / f"fi_ms_{i:.0e}.npy",
            fi_ms_,
        )

        fi_ms_inv_ = classical_result[
            "covariance_matrix_for_entire_distribution"
        ].reshape(N_GRIDS, N_GRIDS, 2, 2)
        np.save(
            ROOT_DIR / f"fi_ms_inv_{i:.0e}_with_remainder.npy",
            fi_ms_inv_,
        )


if __name__ == "__main__":
    classical_dr = (
        driver.Builder()
        .with_modules(
            rayfish,
            fisher_information_remainder_channel,
            distance_3_sources_xz,
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

"""Builds the Hamilton driver from the correct modules based on user input."""

from __future__ import annotations

from beartype.typing import Any, Dict
from hamilton import driver
from hamilton.execution import executors

import rayfish
import rayfish.quantum_fisher_information as qfi
from rayfish import PARAMETER_CATALOGUE

# Public API
__all__ = ("build_quantum_dag",)


def build_quantum_dag(
    light_characteristic: str,
    parameter_of_interest: str,
    config: Dict[str, Any] | None = None,
) -> driver.Driver:
    """Builds correct hamilton driver for the simulation."""
    base_dr = get_base_quantum_hamilton_builder(config=config)

    if light_characteristic == "Incoherent":
        base_dr = incoherent_hamilton_builder(base_dr=base_dr)
    elif light_characteristic == "Coherent":
        base_dr = coherent_hamilton_builder(base_dr=base_dr)
    else:
        raise NotImplementedError("Only Coherent and Incoherent cases implemented.")

    if parameter_of_interest == "distance_2_sources":
        return distance_2_sources_driver(base_dr=base_dr)
    else:
        raise NotImplementedError(
            f"We currently support the following parameters: {PARAMETER_CATALOGUE}"
        )


def get_base_quantum_hamilton_builder(
    config: Dict[str, Any] | None = None,
) -> driver.Builder:
    """Base driver common to all simulation scenarios."""
    dr = driver.Builder().with_modules(rayfish, qfi)
    if config:
        dr.with_config(config=config)
    return dr


def incoherent_hamilton_builder(base_dr: driver.Builder) -> driver.Builder:
    """Adding functions for incoherent sources."""
    from rayfish.quantum_fisher_information import incoherent_sources

    dr = (
        base_dr.with_modules(incoherent_sources)
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_local_executor(executors.SynchronousLocalTaskExecutor())
    )
    return dr


def coherent_hamilton_builder(base_dr: driver.Builder) -> driver.Builder:
    """Adding functions for coherent sources."""
    from rayfish.quantum_fisher_information import coherent_sources

    return base_dr.with_modules(coherent_sources)


def distance_2_sources_driver(base_dr: driver.Builder) -> driver.Driver:
    """Builds driver that simulates distance between two point sources."""
    from rayfish.parameter_catalogue import distance_2_sources

    return base_dr.with_modules(distance_2_sources).build()

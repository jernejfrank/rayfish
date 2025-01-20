"""Builds the Hamilton driver from the correct modules based on user input."""

from __future__ import annotations

from beartype.typing import Any, Dict
from hamilton import driver
from hamilton.execution import executors

import rayfish
import rayfish.quantum_fisher_information as qfi
from api.pydantic_models import LightType, SimulationConfig
from rayfish import ParameterCatalogue

# Public API
__all__ = ("build_quantum_dag",)


def build_quantum_dag(config: SimulationConfig) -> driver.Driver:
    """Builds correct hamilton driver for the simulation."""
    dag_config = {"dimension": config.simulation_dimension}
    base_dr = get_base_quantum_hamilton_builder(config=dag_config)

    if config.light_type == LightType.INCOHERENT:
        base_dr = incoherent_hamilton_builder(base_dr=base_dr)
    elif config.light_type == LightType.COHERENT:
        base_dr = coherent_hamilton_builder(base_dr=base_dr)
    else:
        raise NotImplementedError("Only Coherent and Incoherent cases implemented.")

    if config.estimation_parameter == ParameterCatalogue.dist_2_src.value:
        return distance_2_sources_driver(base_dr=base_dr)
    else:
        raise NotImplementedError(
            "We currently support the following "
            f"parameters: {[i.value for i in ParameterCatalogue]}"
        )


def get_base_quantum_hamilton_builder(
    config: Dict[str, Any],
) -> driver.Builder:
    """Base driver common to all simulation scenarios."""
    dr = driver.Builder().with_modules(rayfish, qfi)
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

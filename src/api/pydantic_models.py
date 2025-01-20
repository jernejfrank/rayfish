"""Helper module to store the Pydantic models."""

from enum import Enum

from beartype.typing import Any
from pydantic import BaseModel, ConfigDict, field_validator

__all__ = (
    "LightType",
    "SimulationDimension",
    "ParametersInput",
    "SimulationConfig",
)


class LightType(str, Enum):
    """Nature of emitted light."""

    COHERENT = "coherent"
    INCOHERENT = "incoherent"


class SimulationDimension(int, Enum):
    """Simulation for x-z or z-y-z plane."""

    ONE = 1  # x-z plane
    TWO = 2  # x-y-z plane


class SimulationConfig(BaseModel):
    """General parameters to choose simulation conditions."""

    model_config = ConfigDict(use_enum_values=True)

    light_type: LightType
    estimation_parameter: str
    simulation_dimension: SimulationDimension

    @field_validator("light_type", "estimation_parameter", mode="before")
    @classmethod
    def _to_lowercase(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value


class ParametersInput(BaseModel):
    """Input parameters for simulation."""

    number_of_point_sources: int = 2
    wave_vector: float = 2.0
    rayleigh_length: float = 1.0
    number_of_point_source_distributions: int = 20
    maximal_length: float = 4.0
    minimal_spacing: float = 0.1

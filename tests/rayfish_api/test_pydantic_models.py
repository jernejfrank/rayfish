"""Testing the pydantic models used to parse user inputs."""

import pytest
from beartype.typing import Any, Dict
from pydantic import ValidationError

from api.pydantic_models import SimulationConfig


@pytest.mark.parametrize(
    "light_type, estimation_parameter, simulation_dimension, expected",
    [
        (
            "Coherent",
            "p",
            1,
            {
                "light_type": "coherent",
                "estimation_parameter": "p",
                "simulation_dimension": 1,
            },
        ),
        (
            "incoherent",
            "P",
            2,
            {
                "light_type": "incoherent",
                "estimation_parameter": "p",
                "simulation_dimension": 2,
            },
        ),
        (
            "Coherent",
            "P",
            1,
            {
                "light_type": "coherent",
                "estimation_parameter": "p",
                "simulation_dimension": 1,
            },
        ),
    ],
)
def test_SimulationConfig_validates(
    light_type: str,
    estimation_parameter: str,
    simulation_dimension: int,
    expected: Dict[str, Any],
) -> None:
    """Tests SimulationConfig converts to lower case and validates inputs."""
    conf = SimulationConfig(
        light_type=light_type,  # type: ignore
        estimation_parameter=estimation_parameter,
        simulation_dimension=simulation_dimension,  # type: ignore
    )
    assert conf.model_dump() == expected


@pytest.mark.parametrize(
    "light_type, estimation_parameter, simulation_dimension",
    [
        (
            "iCoherent",
            "p",
            1,
        ),
        (
            "incoherent",
            "P",
            20,
        ),
        (
            "Coherent",
            100,
            1,
        ),
    ],
)
def test_SimulationConfig_raises_validation_error(
    light_type: str,
    estimation_parameter: str,
    simulation_dimension: int,
) -> None:
    """Tests SimulationConfig converts to lower case and validates inputs."""
    with pytest.raises(ValidationError):
        SimulationConfig(
            light_type=light_type,  # type: ignore
            estimation_parameter=estimation_parameter,
            simulation_dimension=simulation_dimension,  # type: ignore
        )

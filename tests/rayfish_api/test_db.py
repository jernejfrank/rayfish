"""Tests local persister for saving simulations."""

from pathlib import Path

import pandas as pd
import pytest
from beartype.typing import Any, Dict, Generator
from pandas.testing import assert_frame_equal

from api.db import SQLiteClient
from api.pydantic_models import ParametersInput, SimulationConfig


@pytest.fixture
def sim_conf() -> Dict[str, Any]:
    """Simulation config."""
    return SimulationConfig(
        light_type="coherent",  # type: ignore
        estimation_parameter="custom",
        simulation_dimension=1,  # type: ignore
    ).model_dump()


@pytest.fixture
def param_conf() -> Dict[str, Any]:
    """Simulation initial conditions."""
    return dict(ParametersInput())


@pytest.fixture
def client(tmp_path: Path) -> Generator[SQLiteClient]:
    """Connection to db."""
    client = SQLiteClient(custom_dir=str(tmp_path))
    client.init_main()
    yield client


def test_create_new_sim_id(
    client: SQLiteClient, sim_conf: Dict[str, Any], param_conf: Dict[str, Any]
) -> None:
    """Test that unique ids get created for non existing database entries."""
    sim_id, exists = client.get_simulation_id(sim_conf, param_conf)
    assert not exists
    sim_id_2, exists = client.get_simulation_id(sim_conf, param_conf)
    assert not exists
    assert sim_id != sim_id_2


def test_save_load_simulation_parameters(
    client: SQLiteClient, sim_conf: Dict[str, Any], param_conf: Dict[str, Any]
) -> None:
    """Test that main table gets populated."""
    test_id = "0"
    client.save_simulation_parameters(test_id, sim_conf, param_conf)
    sim_id, exists = client.get_simulation_id(sim_conf, param_conf)
    df = client.load("simulations").iloc[0]
    assert df["id"] == test_id
    assert df["light_type"] == sim_conf["light_type"]
    assert df["estimation_parameter"] == sim_conf["estimation_parameter"]
    assert df["simulation_dimension"] == sim_conf["simulation_dimension"]
    assert df["number_of_point_sources"] == param_conf["number_of_point_sources"]
    assert df["wave_vector"] == param_conf["wave_vector"]
    assert df["rayleigh_length"] == param_conf["rayleigh_length"]
    assert (
        df["number_of_point_source_distributions"]
        == param_conf["number_of_point_source_distributions"]
    )
    assert df["maximal_length"] == param_conf["maximal_length"]
    assert df["minimal_spacing"] == param_conf["minimal_spacing"]
    assert exists
    assert sim_id == test_id


def test_save_load_dataframe(client: SQLiteClient) -> None:
    """Test full cycle of saving and loading via db."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    client.save_simulation_results("test", df)
    result = client.load("test")
    result.drop("index", axis=1, inplace=True)
    assert_frame_equal(df, result)

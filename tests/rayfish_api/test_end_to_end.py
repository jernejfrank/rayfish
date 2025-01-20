"""Testing the complete run of the Rayfish app."""

from pathlib import Path

from pandas.testing import assert_frame_equal

from api import RayfishApp
from api.pydantic_models import ParametersInput, SimulationConfig


def test_end_to_end_rayfish_app_coherent(tmp_path: Path) -> None:
    """End to end test for running a rayfish simulation."""
    sim_config = SimulationConfig(
        light_type="coherent",  # type: ignore
        estimation_parameter="distance_2_sources",
        simulation_dimension=1,  # type: ignore
    )
    param_config = ParametersInput(number_of_point_source_distributions=1)

    app = RayfishApp(
        simulation_config=sim_config,
        input_parameters=param_config,
        save_dir=str(tmp_path),
    )

    sim_id = app.sim_id
    assert not app.does_simulation_exist
    assert len(app.get_existing_simulations()) == 0

    result = app.run_simulation()

    del app

    new_app = RayfishApp(
        simulation_config=sim_config,
        input_parameters=param_config,
        save_dir=str(tmp_path),
    )
    assert new_app.does_simulation_exist
    assert len(new_app.get_existing_simulations()) == 1
    assert sim_id == new_app.sim_id

    new_result = new_app.load_simulation()
    assert_frame_equal(
        result["quantum"].statistical_bounds_dataframe,
        new_result["quantum"].statistical_bounds_dataframe,
    )


def test_end_to_end_rayfish_app_incoherent(tmp_path: Path) -> None:
    """End to end test for running a rayfish simulation."""
    sim_config = SimulationConfig(
        light_type="incoherent",  # type: ignore
        estimation_parameter="distance_2_sources",
        simulation_dimension=1,  # type: ignore
    )
    param_config = ParametersInput(number_of_point_source_distributions=1)

    app = RayfishApp(
        simulation_config=sim_config,
        input_parameters=param_config,
        save_dir=str(tmp_path),
    )

    sim_id = app.sim_id
    assert not app.does_simulation_exist
    assert len(app.get_existing_simulations()) == 0

    result = app.run_simulation()

    del app

    new_app = RayfishApp(
        simulation_config=sim_config,
        input_parameters=param_config,
        save_dir=str(tmp_path),
    )
    assert new_app.does_simulation_exist
    assert len(new_app.get_existing_simulations()) == 1
    assert sim_id == new_app.sim_id

    new_result = new_app.load_simulation()
    assert_frame_equal(
        result["quantum"].statistical_bounds_dataframe,
        new_result["quantum"].statistical_bounds_dataframe,
    )

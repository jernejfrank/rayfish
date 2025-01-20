"""Dashboard backend.

This module either collects existing results from the database or generates the results
if they do not yet exist and saves them to the database.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from beartype.typing import Dict, Optional
from hamilton import driver

from api.db import SQLiteClient
from api.pydantic_models import ParametersInput, SimulationConfig
from api.quantum_dag_builder import build_quantum_dag
from api.visualize import PointSourceDistribution, StatisticalBounds

# Public API
__all__ = ("RayfishApp",)


def init_rayfish_drivers(
    simulation_config: SimulationConfig,
) -> Dict[str, driver.Driver]:
    quantum_dr = build_quantum_dag(simulation_config)
    return {"quantum": quantum_dr}


def create_dataframe_from_numpy_array(
    grid_spacing: npt.NDArray[np.float64], covariance_matrix: npt.NDArray[np.float64]
) -> pd.DataFrame:
    """Helper to convert simulation result into a dataframe."""
    covariance_matrix_real = np.real(covariance_matrix)
    df_covariance = pd.DataFrame(
        np.real(covariance_matrix_real.reshape(covariance_matrix_real.shape[0], -1)),
        index=grid_spacing,
    )

    num_rows, num_cols = (
        covariance_matrix_real.shape[1],
        covariance_matrix_real.shape[2],
    )
    df_covariance.columns = [
        f"var_{i}_{j}" if i == j else f"cov_{i}_{j}"
        for i in range(num_rows)
        for j in range(num_cols)
    ]
    return df_covariance


class RayfishApp:
    """Running the simulations, storing and retrieving them from the db."""

    def _init_point_source_distribution(self) -> PointSourceDistribution:
        results = self._simulators["quantum"].execute(
            final_vars=[
                "coordinates",
                "spacing",
                "number_of_point_sources",
            ],
            inputs=self.input_parameters,
        )
        return PointSourceDistribution(
            number_of_point_sources=results["number_of_point_sources"],
            grid_spacing=results["spacing"],
            coordinates=results["coordinates"],
        )

    def __init__(
        self,
        simulation_config: SimulationConfig,
        input_parameters: ParametersInput,
        save_dir: Optional[str] = None,
    ):
        self.simulation_config = simulation_config
        self.input_parameters = dict(input_parameters)

        self.simulation_persister = SQLiteClient(custom_dir=save_dir)
        self.simulation_persister.init_main()
        self.sim_id, self._simulation_exists = (
            self.simulation_persister.get_simulation_id(
                dict(simulation_config), dict(input_parameters)
            )
        )

        # if isinstance(self.sim_id, tuple):
        #     self.sim_id = self.sim_id[0] # some weird reason it returns it as a tuple

        self._simulators = init_rayfish_drivers(simulation_config)
        self._source_distribution = self._init_point_source_distribution()

    def update_simulation_inputs(self, new_inputs: ParametersInput) -> "RayfishApp":
        """Updates new input values for running the simulation."""
        return RayfishApp(self.simulation_config, new_inputs)

    @property
    def point_source_distribution(self) -> pd.DataFrame:
        """Dataframe collecting all point source distributions."""
        return self._source_distribution.particle_distribution_dataframe

    @property
    def grid_spacing(self) -> npt.NDArray[np.float64]:
        """All indices of the point source distribution."""
        return self._source_distribution.grid_spacing

    def generate_distribution_figure(self, selected_spacing: float) -> go.Figure:
        """Plots point source distribution for a specific grid spacing."""
        return self._source_distribution.generate_figure(selected_spacing)

    @property
    def does_simulation_exist(self) -> bool:
        """Outputs if simulation table exists in database."""
        return self._simulation_exists

    def get_existing_simulations(self) -> pd.DataFrame:
        """Fetches main table with all simulation id listed."""
        df = self.simulation_persister.load("simulations")
        return df

    def run_simulation(self) -> Dict[str, StatisticalBounds]:
        """Execute the DAG to obtain estimation boundaries."""
        output = {}
        for case_, dr in self._simulators.items():
            result = dr.execute(
                inputs=self.input_parameters,
                final_vars=["covariance_matrix_for_entire_distribution"],
            )

            temp_df = create_dataframe_from_numpy_array(
                self.grid_spacing, result["covariance_matrix_for_entire_distribution"]
            )

            statistical_bound = StatisticalBounds(
                name=case_, _df_covariance=temp_df.copy()
            )
            output[case_] = statistical_bound

            temp_df["measurement_case"] = case_
            self.simulation_persister.save_simulation_results(self.sim_id, temp_df)
        self.simulation_persister.save_simulation_parameters(
            self.sim_id, dict(self.simulation_config), self.input_parameters
        )
        self._simulation_exists = True

        return output

    def load_simulation(self) -> Dict[str, StatisticalBounds]:
        """Load existing simulation results from database."""
        if not self._simulation_exists:
            raise ValueError(
                "This simulation run is not saved. Run the simulation first."
            )
        loaded_df = self.simulation_persister.load(self.sim_id)
        cases = loaded_df["measurement_case"].unique()
        output = {}
        for case_ in cases:
            filtered_df = loaded_df[loaded_df["measurement_case"] == case_]
            filtered_df.drop(columns="measurement_case", inplace=True)
            filtered_df.set_index(filtered_df.columns[0], drop=True, inplace=True)
            filtered_df.index.name = None
            statistical_bound = StatisticalBounds(
                name=case_, _df_covariance=filtered_df
            )
            output[case_] = statistical_bound
        return output

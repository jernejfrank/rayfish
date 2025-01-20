"""Visualizing distributions and statistical boundaries."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from beartype.typing import Dict, List, Optional, Union

# Public API
__all__ = ("PointSourceDistribution",)


def _create_distribution_base_figure(df: pd.DataFrame, num_particles: int) -> go.Figure:
    """Creates base figure that we will plot scatter point into."""
    scene = {
        "xaxis": {},
        "yaxis": {},
        "zaxis": {},
    }

    for c_i, coord in enumerate(scene.keys()):
        min_coord = min([
            df.min(axis=0)[c_i * num_particles + i] for i in range(num_particles)
        ])
        max_coord = max([
            df.max(axis=0)[c_i * num_particles + i] for i in range(num_particles)
        ])
        scene[coord]["range"] = [min_coord - 0.5, max_coord + 0.5]
        scene[coord]["autorange"] = False

    scene["xaxis"]["title"] = "x"
    scene["yaxis"]["title"] = "y"
    scene["zaxis"]["title"] = "z"

    scene["camera"] = {"eye": {"x": 2, "y": 2, "z": 1.5}}
    fig = go.Figure()
    fig.update_layout(scene=scene)
    return fig


def _transpose_dataframe_to_particle_per_row(
    df: Union[pd.Series, pd.DataFrame], num_particles: int
) -> pd.DataFrame:
    """Helper function to convert single row into dataframe.

    Each column in row represents a coordinate. This creates a dataframe where each
    row is a point source and the three columns are the x,y,z coordinates.
    """
    particle_coordinates = {"x": [], "y": [], "z": []}

    for c_i, coord in enumerate(particle_coordinates.keys()):
        particle_coordinates[coord] = [
            df[c_i * num_particles + i] for i in range(num_particles)
        ]
    return pd.DataFrame(particle_coordinates)


@dataclass
class PointSourceDistribution:
    """Class to store point source distribution and visualize it."""

    number_of_point_sources: int
    grid_spacing: npt.NDArray[np.float64]
    coordinates: npt.NDArray[np.float64]
    _df: Optional[pd.DataFrame] = None
    _base_fig: Optional[go.Figure] = None

    def __post_init__(self):
        """Inits the dataframe."""
        self._df = pd.concat(
            [
                pd.DataFrame(self.grid_spacing, columns=["spacing"], dtype=np.float64),
                pd.DataFrame(self.coordinates, dtype=np.float64),
            ],
            axis=1,
        ).set_index("spacing")
        self._base_fig = _create_distribution_base_figure(
            self._df, self.number_of_point_sources
        )

    @property
    def particle_distribution_dataframe(self) -> pd.DataFrame:
        """Dataframe storing point source positions."""
        if self._df is not None:
            return self._df
        raise ValueError("No dataframe present.")

    def generate_figure(self, selected_grid_spacing: float) -> go.Figure:
        """Point source distribution figure.

        Returns:
            A plotly figure of a single distribution plotted via 3d scatter.
        """
        if self._df is None:
            raise ValueError("No dataframe.")

        if self._base_fig is None:
            raise ValueError("No initial figure generated.")

        if selected_grid_spacing not in self._df.index:
            raise ValueError("Grid spacing does not exist.")

        distribution_dataframe = _transpose_dataframe_to_particle_per_row(
            self._df.loc[selected_grid_spacing], self.number_of_point_sources
        )

        self._base_fig.add_trace(
            go.Scatter3d(
                x=distribution_dataframe["x"],
                y=distribution_dataframe["y"],
                z=distribution_dataframe["z"],
                mode="markers",
                marker={
                    "color": distribution_dataframe.index,
                    "colorbar": {"title": "Particles"},
                },
            ),
        )
        self._base_fig.update_layout(width=800, height=600)
        return self._base_fig


def _create_estimation_base_figure() -> go.Figure:
    """Creates base figure that we will plot statistical boundaries."""
    scene = {
        "xaxis": {"title": "Spacing"},
        "yaxis": {"title": "Covariance Matrix"},
    }

    fig = go.Figure()
    fig.update_layout(scene=scene)
    return fig


@dataclass
class StatisticalBounds:
    """Class to store covariance matrix and visualize it."""

    name: str
    _df_covariance: Optional[pd.DataFrame] = None
    _base_fig: Optional[go.Figure] = None

    def __post_init__(self):
        """Setting up the figure to plot on."""
        if self._df_covariance is not None:
            self._df_covariance.columns = [
                f"{self.name}_{i}" for i in self._df_covariance.columns
            ]

        self._base_fig = _create_estimation_base_figure()

    @property
    def statistical_bounds_dataframe(self) -> pd.DataFrame:
        """Dataframe collecting simulation results."""
        if self._df_covariance is not None:
            return self._df_covariance
        raise ValueError("No dataframe present.")

    def name_parameters(self, rename_map: Dict[str, str]) -> None:
        """Rename matrix elements with user input.

        The dictionary key needs to be the names of the existing columns, we suggest
        the following naming convention:
        rename_map = {
                    "name_var_0_0" : "var_my_first_param",
                     "name_cov_0_1" : "cov_between_first_param_and_second_param",
                     .
                     .
                     .
                    }

        Args:
            rename_map: dictionary mapping old names to new column names

        Returns:
            None - dataframe is modified in place.
        """
        if self._df_covariance is not None:
            self._df_covariance.columns = [
                rename_map[old_name] for old_name in self._df_covariance.columns
            ]
            return
        raise ValueError("No dataframe present.")

    def generate_figure(
        self,
        selected_estimation_parameters: Optional[List[str]] = None,
        reset_base: bool = False,
    ) -> go.Figure:
        """Creates the figure of the statistical bounds."""
        if reset_base:
            self._base_fig = _create_estimation_base_figure()

        if self._df_covariance is None:
            raise ValueError("No dataframe.")

        if self._base_fig is None:
            raise ValueError("No initial figure generated.")

        if selected_estimation_parameters is None:
            selected_estimation_parameters = list(self._df_covariance.columns)

        for column in selected_estimation_parameters:
            if column in self._df_covariance.columns:
                self._base_fig.add_trace(
                    go.Scatter(
                        x=self._df_covariance.index.astype(float),
                        y=self._df_covariance[column],
                        mode="lines+markers",
                        name=column,
                    )
                )
            else:
                raise ValueError("Column not present in dataframe.")

        return self._base_fig

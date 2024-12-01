"""Dataframe manipulation to display the underlying distribution of point sources."""

import pandas as pd
import plotly.graph_objects as go
from beartype.typing import Any, Dict, Tuple

# Public API
__all__ = (
    "create_distribution_dataframe",
    "get_distribution_figure",
)


def _create_base_figure(df: pd.DataFrame, num_particles: int) -> go.Figure:
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


def create_distribution_dataframe(
    result: Dict[str, Any],
) -> Tuple[pd.DataFrame, go.Figure]:
    """Combines spacing and coordinates into single dataframe."""
    distribution_df = pd.concat(
        [
            pd.DataFrame(result["spacing"], columns=["spacing"]),
            pd.DataFrame(result["coordinates"]),
        ],
        axis=1,
    ).set_index("spacing")

    base_fig = _create_base_figure(
        df=distribution_df, num_particles=result["number_of_point_sources"]
    )
    return distribution_df, base_fig


def get_distribution_figure(
    particle_distribution: pd.DataFrame, num_particles: int, base_fig: go.Figure
) -> go.Figure:
    """Point source distribution figure.

    Args:
        particle_distribution: Dataframe of all particle coordinates.
        num_particles: number of point sources in the distribution.
        base_fig: the underlying figure that was created to fit all distributions in
            terms of scaling and view.

    Returns:
        A plotly figure of a single distribution ploted via 3d scatter.
    """
    distribution_dataframe = _transpose_dataframe_to_particle_per_row(
        particle_distribution=particle_distribution, num_particles=num_particles
    )

    base_fig.add_trace(
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
    return base_fig


def _transpose_dataframe_to_particle_per_row(
    particle_distribution: pd.DataFrame, num_particles: int
) -> pd.DataFrame:
    """Helper function to covert single row into datafrme.

    Each column in row represents a coordinate. This creates a dataframe where each
    row is a point source and the three columns are the x,y,z coordinates.
    """
    particle_coordinates = {"x": [], "y": [], "z": []}

    for c_i, coord in enumerate(particle_coordinates.keys()):
        particle_coordinates[coord] = [
            particle_distribution[c_i * num_particles + i] for i in range(num_particles)
        ]
    return pd.DataFrame(particle_coordinates)

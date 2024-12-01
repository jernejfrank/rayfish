"""Create covariance matrix plot."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from beartype.typing import Any, Dict

# Public API
__all__ = (
    "create_covariance_dataframe",
    "get_covariance_figure",
)


def create_covariance_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """Converts numerical results to a plotly figure."""
    qfi = np.real(result["covariance_matrix_for_entire_distribution"])
    gamma = np.imag(result["covariance_matrix_for_entire_distribution"])
    x = result["spacing"]
    dfs = {
        "spacing": x,
        "var_xy": qfi[:, 0, 0],
        "var_z": qfi[:, 1, 1],
        "cov_xyz_1": qfi[:, 0, 1],
        "cov_xyz_2": qfi[:, 1, 0],
        "gamma_xy": gamma[:, 0, 0],
        "gamma_z": gamma[:, 1, 1],
        "cov_gamma_1": gamma[:, 0, 1],
        "cov_gamma_2": gamma[:, 1, 0],
    }

    df = pd.DataFrame.from_dict(dfs)
    df = df.set_index("spacing")
    return df


def _create_base_figure() -> go.Figure:
    """Creates base figure that we will plot statistical boundaries."""
    scene = {
        "xaxis": {"title": "spacing"},
        "yaxis": {"title": "Covariance Matrix"},
    }

    fig = go.Figure()
    fig.update_layout(scene=scene)
    return fig


def get_covariance_figure(df: pd.DataFrame) -> go.Figure:
    """Plot for covariance matrix."""
    base_fig = _create_base_figure()
    for column in df.columns:
        base_fig.add_trace(
            go.Scatter(
                x=df.index.astype(float),
                y=df[column],
                mode="lines+markers",
                name=column,
            )
        )

    return base_fig

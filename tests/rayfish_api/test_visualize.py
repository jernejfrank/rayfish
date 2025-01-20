"""Testing visualization of point source distribution and statistical boundaries."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from api.visualize import (
    PointSourceDistribution,
    StatisticalBounds,
    _transpose_dataframe_to_particle_per_row,
)


@pytest.fixture
def particle_distribution_df() -> pd.DataFrame:
    """Distribution of particles."""
    manual_df = pd.DataFrame(
        {
            "spacing": np.array([0.1, 0.2, 0.3]),
            0: np.array([-0.1, -0.2, -0.3]),
            1: np.array([0.1, 0.2, 0.3]),
            2: np.array([0.0, 0.0, 0.0]),
            3: np.array([0.0, 0.0, 0.0]),
            4: np.array([-0.1, -0.2, -0.3]),
            5: np.array([0.1, 0.2, 0.3]),
        },
        dtype=np.float64,
    ).set_index("spacing")
    return manual_df


def test_init_PointSourceDistribution(particle_distribution_df: pd.DataFrame) -> None:
    """Checks if dataframe gets created."""
    num_particles = 2
    grid_spacing = np.array([0.1, 0.2, 0.3])
    coords = np.array([
        [-0.1, 0.1, 0.0, 0.0, -0.1, 0.1],
        [-0.2, 0.2, 0.0, 0.0, -0.2, 0.2],
        [-0.3, 0.3, 0.0, 0.0, -0.3, 0.3],
    ])

    distributon = PointSourceDistribution(
        number_of_point_sources=num_particles,
        grid_spacing=grid_spacing,
        coordinates=coords,
    )

    assert_frame_equal(
        particle_distribution_df, distributon.particle_distribution_dataframe
    )


def test_transpose_dataframe(particle_distribution_df: pd.DataFrame) -> None:
    """Transposing the dataframe to have coordinates for each point source."""
    new_df = _transpose_dataframe_to_particle_per_row(
        particle_distribution_df.loc[0.1], 2
    )
    transposed_df = pd.DataFrame({
        "x": [-0.1, 0.1],
        "y": [0.0, 0.0],
        "z": [-0.1, 0.1],
    })

    assert_frame_equal(new_df, transposed_df)


def test_StatisticalBounds_init() -> None:
    """Check if we correctly namespace columns."""
    name = "test"
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    cov_mat = StatisticalBounds(name, df)

    assert (
        cov_mat.statistical_bounds_dataframe.columns == ["test_a", "test_b", "test_c"]
    ).all()


def test_rename_columns() -> None:
    """Checks if columns get properly renamed."""
    name = "test"
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    cov_mat = StatisticalBounds(name, df)
    rename_map = {"test_a": "diff_a", "test_b": "diff_b", "test_c": "diff_c"}
    cov_mat.name_parameters(rename_map)
    assert (
        cov_mat.statistical_bounds_dataframe.columns == ["diff_a", "diff_b", "diff_c"]
    ).all()

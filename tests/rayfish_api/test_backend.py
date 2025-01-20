"""Test for rayfish api."""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from api.backend import create_dataframe_from_numpy_array


def test_df_from_np() -> None:
    """Correct conversion between numpy covariance matrix and pandas dataframe."""
    df = pd.DataFrame(
        {
            "var_0_0": [1, 5],
            "cov_0_1": [2, 6],
            "cov_1_0": [3, 7],
            "var_1_1": [4, 8],
        },
        index=[100, 200],
    )
    np_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    spacing = np.array([100, 200])
    new_df = create_dataframe_from_numpy_array(spacing, np_array)
    assert_frame_equal(df, new_df)

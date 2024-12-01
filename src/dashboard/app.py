"""Rayfish UI."""

from __future__ import annotations

import streamlit as st
from beartype.typing import Any, Dict, List
from hamilton import driver

from dashboard.particle_distribution_plot import (
    create_distribution_dataframe,
    get_distribution_figure,
)


# cache to avoid rebuilding the Driver
@st.cache_resource
def get_hamilton_driver() -> driver.Driver:
    """Builds driver that simulates the bounds from scratch."""
    from hamilton.execution import executors

    import rayfish
    import rayfish.quantum_fisher_information as qfi
    from rayfish.parameter_catalogue import common_transforms, distance_2_sources
    from rayfish.quantum_fisher_information import incoherent_sources

    dr = (
        driver.Builder()
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_modules(
            rayfish, qfi, incoherent_sources, common_transforms, distance_2_sources
        )
        .with_config({"dimension": 1})
        .with_local_executor(executors.SynchronousLocalTaskExecutor())
        .build()
    )
    return dr


# cache results for the set of inputs
@st.cache_data
def _execute(
    final_vars: List[str],
    inputs: Dict[str, Any] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Generic utility to cache Hamilton results."""
    dr = get_hamilton_driver()
    return dr.execute(final_vars=final_vars, inputs=inputs, overrides=overrides)  # type: ignore


def execute(
    final_vars: List[str],
    inputs: Dict[str, Any] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Executes Hamilton DAG to get specified outputs."""
    return _execute(final_vars=final_vars, inputs=inputs, overrides=overrides)


def app() -> None:
    """Streamlit app."""
    st.title("Rayfish")

    inputs_ = {
        "wave_vector": 2.0,
        "rayleigh_length": 1.0,
        "number_of_point_source_distributions": 20,
        "maximal_length": 4.0,
        "minimal_spacing": 0.1,
    }

    edited_inputs = inputs_  # st.data_editor(inputs_)

    result = execute(
        final_vars=["coordinates", "spacing", "number_of_point_sources"],
        inputs=edited_inputs,
    )

    distribution_df, base_fig = create_distribution_dataframe(result)

    selected_indices = st.select_slider("Grid Spacing:", distribution_df.index)
    fig_distribution = get_distribution_figure(
        distribution_df.loc[selected_indices],
        result["number_of_point_sources"],
        base_fig,
    )
    st.plotly_chart(fig_distribution)

    # # run the base data that always needs to be displayed
    # data = execute(["lyapunov_covariance_matrix_for_entire_distribution", "spacing"])

    # # display the base dataframe and plotly chart
    # st.dataframe(data["balance_per_job"])
    # st.plotly_chart(data["balance_per_job_boxplot"])

    # # get the selection options from `data`
    # # store the selection in the state `selected_job`
    # st.selectbox("Select a job", options=data["all_jobs"], key="selected_job")
    # # get the value from the dict
    # st.plotly_chart(execute(["job_hist"])["job_hist"])


if __name__ == "__main__":
    app()

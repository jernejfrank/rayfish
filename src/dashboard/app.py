"""Rayfish UI."""

from __future__ import annotations

import streamlit as st
from beartype.typing import Any, Dict, List
from hamilton import driver
from pydantic import BaseModel, ValidationError

from dashboard.particle_distribution_plot import (
    create_distribution_dataframe,
    get_distribution_figure,
)
from dashboard.quantum_dag_builder import build_quantum_dag
from dashboard.statistical_boundaries_plot import (
    create_covariance_dataframe,
    get_covariance_figure,
)
from rayfish import PARAMETER_CATALOGUE


class ParametersInput(BaseModel):
    """Input parameters for simulation."""

    wave_vector: float
    rayleigh_length: float
    number_of_point_source_distributions: int
    maximal_length: float
    minimal_spacing: float


DEFAULT_SCENARIO = ParametersInput(**{
    "wave_vector": 2.0,
    "rayleigh_length": 1.0,
    "number_of_point_source_distributions": 20,
    "maximal_length": 4.0,
    "minimal_spacing": 0.1,
})


def execute(
    dr: driver.Driver,
    final_vars: List[str],
    inputs: Dict[str, Any] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Executes Hamilton DAG to get specified outputs."""
    return dr.execute(final_vars=final_vars, inputs=inputs, overrides=overrides)  # type: ignore


def app() -> None:
    """Streamlit app."""
    st.title("Rayfish")

    col1, col2 = st.columns(2)
    with col1:
        select_light_characteristic = st.selectbox(
            "Select a case:", ["Coherent", "Incoherent"], index=1
        )
    with col2:
        select_parameter_of_interest = st.selectbox(
            "Select a parameter:", PARAMETER_CATALOGUE, index=0
        )

    st.write("## Simulation Parameters")
    inputs_ = DEFAULT_SCENARIO
    try:
        inputs_ = ParametersInput(**st.data_editor(inputs_))  # type: ignore
    except ValidationError as e:
        st.error(f"You entered a wrong datatype: {e}")

    quantum_dr = build_quantum_dag(
        light_characteristic=select_light_characteristic,
        parameter_of_interest=select_parameter_of_interest,
        config={"dimension": 1},
    )

    result = execute(
        dr=quantum_dr,
        final_vars=[
            "coordinates",
            "spacing",
            "number_of_point_sources",
            "covariance_matrix_for_entire_distribution",
        ],
        inputs=inputs_.model_dump(),
    )

    distribution_df, base_fig = create_distribution_dataframe(result)

    if st.checkbox("Show Plot", value=False):
        st.write("### Point source distribution preview")
        selected_indices = st.select_slider("Grid Spacing:", distribution_df.index)
        fig_distribution = get_distribution_figure(
            distribution_df.loc[selected_indices],
            result["number_of_point_sources"],
            base_fig,
        )
        st.plotly_chart(fig_distribution, use_container_width=True)

    covariance_matrix_df = create_covariance_dataframe(result)

    st.write("### Statistical Boundaries via Covariance Matrix")
    selected_columns = st.multiselect(
        "Select matrix elements to plot:",
        options=covariance_matrix_df.columns,
        default=["var_xy", "var_z"],
    )

    if selected_columns:
        fig_covariance = get_covariance_figure(covariance_matrix_df[selected_columns])
        st.plotly_chart(fig_covariance, use_container_width=True)

    st.table(covariance_matrix_df)


if __name__ == "__main__":
    app()

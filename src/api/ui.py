"""Rayfish UI."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from api.backend import RayfishApp
from api.pydantic_models import ParametersInput, SimulationConfig
from rayfish.parameter_catalogue.common_transforms import (
    ParameterCatalogue,  # type: ignore
)


def simulation_config_menu() -> SimulationConfig:
    """Select general simulation parameters that remain for all statistical bounds."""
    col1, col2, col3 = st.columns(3)
    with col1:
        select_light_characteristic = st.selectbox(
            "Light Type:", ["Coherent", "Incoherent"], index=1
        )
    with col2:
        select_parameter_of_interest = st.selectbox(
            "Estimation Parameter:",
            tuple(i.value for i in ParameterCatalogue),
            index=1,
        )
    with col3:
        select_dim = st.selectbox("Simulation Dimension:", ["x-z", "x-y-z"], index=0)
        select_simulation_dimension = 1 if select_dim == "x-z" else 2

    simulation_config = SimulationConfig(
        light_type=select_light_characteristic,  # type: ignore
        estimation_parameter=select_parameter_of_interest,
        simulation_dimension=select_simulation_dimension,  # type: ignore
    )
    return simulation_config


def get_existing_simulations(
    rayfish_app: RayfishApp, inputs_: ParametersInput
) -> ParametersInput:
    """Display history of simulations."""
    if st.checkbox("Show Existing Simulations", value=False):
        st.write("## Existing Simulations")
        param_df = rayfish_app.get_existing_simulations()
        if param_df.empty:
            st.write("No simulations exist.")
        else:
            st.dataframe(param_df)
            most_recent_row = param_df.loc[
                param_df["created_at"].idxmax(), list(dict(ParametersInput()).keys())
            ]  # type: ignore
            inputs_ = ParametersInput(**most_recent_row.to_dict())

    return inputs_


def update_input_parameters(
    rayfish_app: RayfishApp, inputs_: ParametersInput
) -> RayfishApp:
    """Update simulation initial conditions."""
    st.write("## Simulation Parameters")
    try:
        inputs_ = ParametersInput(**st.data_editor(inputs_, use_container_width=True))  # type: ignore
        rayfish_app = rayfish_app.update_simulation_inputs(inputs_)
    except ValidationError as e:
        st.error(f"You entered a wrong value: {e}")

    return rayfish_app


def plot_source_distribution(rayfish_app: RayfishApp) -> None:
    """Plot point source distribution in space."""
    if st.checkbox("Show Plot", value=False):
        st.write("### Point source distribution preview")
        selected_indices = st.select_slider("Grid Spacing:", rayfish_app.grid_spacing)
        fig_distribution = rayfish_app.generate_distribution_figure(selected_indices)
        st.plotly_chart(fig_distribution, use_container_width=True)


def plot_statistical_bounds(rayfish_app: RayfishApp) -> None:
    """Plot resulting covariance matrix elements."""
    statistical_bounds_dict = {}
    if not rayfish_app.does_simulation_exist:
        if st.button("Run"):
            statistical_bounds_dict = rayfish_app.run_simulation()
    else:
        statistical_bounds_dict = rayfish_app.load_simulation()

    if statistical_bounds_dict:
        estimation_parameters = []
        for covariance_matrix in statistical_bounds_dict.values():
            estimation_parameters.extend(
                covariance_matrix.statistical_bounds_dataframe.columns
            )

        selected_columns = st.multiselect(
            "Select matrix elements to plot:",
            options=estimation_parameters,
            default=[name for name in estimation_parameters if "var" in name],
        )

        if selected_columns:
            figures = []
            for case, covariance_matrix in statistical_bounds_dict.items():
                specific_columns = [c for c in selected_columns if case in c]
                figures.append(covariance_matrix.generate_figure(specific_columns))

            fig_covariance = go.Figure()
            for figure in figures:
                fig_covariance.add_traces(figure.data)

            st.plotly_chart(fig_covariance, use_container_width=True)


def app() -> None:
    """Streamlit app."""
    st.title("Rayfish")

    simulation_config = simulation_config_menu()

    inputs_ = ParametersInput()
    rayfish_app = RayfishApp(simulation_config, inputs_)
    inputs_ = get_existing_simulations(rayfish_app, inputs_)
    rayfish_app = update_input_parameters(rayfish_app, inputs_)

    plot_source_distribution(rayfish_app)

    st.write("### Statistical Boundaries via Covariance Matrix")

    plot_statistical_bounds(rayfish_app)


if __name__ == "__main__":
    app()

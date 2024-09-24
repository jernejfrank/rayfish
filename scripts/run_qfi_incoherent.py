"""Testing 2 point soruces."""

if __name__ == "__main__":
    from hamilton import driver
    from hamilton.execution import executors

    import rayfish
    import rayfish.incoherent_sources as qfi
    from rayfish.parameter_catalogue import common_transforms, distance_2_sources

    dr = (
        driver.Builder()
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_modules(rayfish, qfi, common_transforms, distance_2_sources)
        .with_config({"dimension": 1})
        .with_local_executor(executors.SynchronousLocalTaskExecutor())
        .build()
    )

    dr.display_all_functions("./docs/DAG.png")

    result = dr.execute(
        inputs={
            "wave_vector": 2.0,
            "rayleigh_length": 1.0,
            "number_of_point_source_distributions": 1,
            "maximal_length": 1.0,
            "minimal_spacing": 0.5,
        },
        final_vars=["lyapunov_covariance_matrix_for_entire_distribution"],
    )

    print(result)

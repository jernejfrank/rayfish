"""Testing 2 point soruces."""

if __name__ == "__main__":
    from hamilton import driver

    import rayfish
    import rayfish.incoherent_sources as qfi

    dr = driver.Builder().with_modules(rayfish, qfi).build()

    dr.display_all_functions("./docs/DAG.png")

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Classical fisher information matrix calculation for arbitrary seperable x-y polynomials
- Hermite-Gaussian basis as an example to check validity
- Added 2 pt sources for only 1D x distance estimation
- Added 3pt sources only x-z dim
- Visualization helper for figure plots
- covariance matrix as inverse of the fisher information

### Changed
- renamed 2 pt sources to be specific its x-y dimension
- old covariance matrix renamed to fisher information

### Deprecated

### Removed

### Fixed
- Normalisation for classical FI
- Number of simulations gets extracted from coordinates shape

### Security


## [0.4.0] - 2025-12-01

### Added
- backend with RayfishApp to wire all DAGs and easier to plot data
- sqlite database to persist simulation results
- pydantic models for input verification

### Changed
- streamlit UI refactored to delegate execution to backend
- examples refactored from manual DAGs to using RayfishApp api
- minor typos and naming conventions fixed / tests updated

### Security
- injecting sql queries directly via strings since it is a local database


## [0.3.0] - 2024-12-01

### Added
- streamlit dashboard webapp to run in local mode
- automatic building of Hamilton driver based on scenario
- on-demand computation using input/output
- plotting helpers for transforming numpy results to pandas dataframes that are used in plotly

### Changed
- package structure redesign to have `src` as root

### Removed
- removed unnecessary packages that are not used


## [0.2.0] - 2024-10-03

### Added
- collapse grid for x and y separately
- coherent covariance matrix
- example of 2 coherent point sources

## Changed
- parameter catalogue two point sources removed conditional for 2 dimensional since
it doesn't occur as a use-case
- refactored coherent and incoherent sources under same quantum_fisher_information folder
and moved tau_submat into common submodule to share between them


## [0.1.1] - 2024-09-29

### Added
- creating 2D grid in x-y dimension
- GitHub CI action to run ruff linting and pytest for PR/push to main

### Changed
- pull-request template asks for changelog completion, removed some default prompts

### Fixed
- 2 sources Jacobian uses `pipe_output` since it was released in sf-hamilton v1.78


## [0.0.1] - 2024-09-24

### Added
- computing quantum covariance matrix for incoherent sources
- example replicating Tsang's 2 point sources

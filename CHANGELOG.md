# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.3.0] - 2026-07-03

### Added
- Classical Fisher information matrix for arbitrary separable x-y polynomial mode
  bases via closed-form Gaussian overlaps and derivatives
- Hermite-Gaussian x-y measurement basis with configurable mode degree and
  mode-set filtering
- Remainder channel bucketing all unresolved light, completing the resolved
  modes to a valid POVM
- Detector noise floor (dark counts / background light) in the mode-sorting
  Fisher information, `J = τ² Σ ∂P∂P / (τP + b)`
- Nearest-neighbour mode cross-talk model: Givens-rotation chain applied to the
  x- and y-mode indices independently
- Direct-imaging (DI) benchmark via numerical quadrature of the image-plane
  intensity distribution
- Arbitrary point-source intensity distributions (equal-intensity fallback and
  single-source contrast parametrization)
- Intensity as an estimated or nuisance parameter: 4N-parameter catalogue
  Jacobians, Schur-complement nuisance covariance, and spatial-block covariance
- Parameter catalogues for 2 sources (x separation) and 3 sources (x-z crack
  model)
- Visualization helpers shipped as `rayfish.visualize_helpers` behind the
  `viz` extra

### Changed
- QFI computation vectorized over point-source distributions
- Renamed 2-point-source catalogue to be specific about its x-dimension
- Old covariance matrix renamed to Fisher information
- Examples reorganized per scenario (hello_world; 3-source ideal_main_figures,
  bucket_vs_mode_cutoff, mixing_modes, background_noise,
  ideal_unequal_intensities, unknown_intensities)
- Package version read from installed metadata instead of a bundled VERSION
  file (fixes clean wheel installs)

### Fixed
- Missing conjugation of complex mode coefficients in the derivative overlap
  closed forms (real-valued bases, e.g. Hermite-Gaussian, were unaffected)
- DI probability derivatives extended to the 4N parameter layout used by the
  catalogue Jacobians
- Schur complement in the nuisance covariance now takes the real part of the
  full QFIM instead of silently casting a complex result
- Near-zero mode probabilities are clamped instead of dropped, keeping the
  Fisher information continuous close to the origin
- Mode-set filtering preserves the caller's requested channel ordering
- Normalisation for classical FI
- Number of simulations gets extracted from coordinates shape


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

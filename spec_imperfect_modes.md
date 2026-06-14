# Imperfect Mode Sorting Spec

## Goal
Add a minimal, physically meaningful imperfect-measurement model to the `distance_3_point_sources_hg_x` and `distance_3_point_sources_hg_xy` estimation examples so we can answer the referee's concern about mode mismatch, leakage, and finite sorting fidelity.

The implementation should preserve the existing analytic overlap machinery. The model should perturb the **measurement basis at the amplitude level**, not the probability level.

## Core Idea

Keep the ideal Hermite-Gaussian measurement basis and apply a fixed mixing matrix `U` to the **mode amplitudes** before squaring:

```text
a'_m = sum_b U_{m b} a_b
p'_m = |a'_m|^2
```

where:
- `a_b` is the ideal overlap amplitude between mode `b` and the source field,
- `U` is a parameter-independent matrix representing mode-sorter imperfection,
- `p'_m` is the observed mode probability.

This is the minimal extension that:
- remains physically valid,
- does not require new Gaussian-envelope integrals,
- gives a genuine field-level mismatch model,
- allows derivative propagation analytically.

## Why This Approach

### What it captures
- finite extinction ratio between neighboring modes,
- mode mixing from sorter imperfections,
- leakage out of the ideal channel basis,
- degraded Fisher information from non-ideal measurement optics.

### What it avoids
- no need to re-derive overlaps for a modified Gaussian PSF,
- no need to rotate probability vectors directly,
- no need to abandon the current analytic FI pipeline.

## Model Definition

### 1. Ideal amplitudes
Use the existing analytic overlap functions in `src/rayfish/gaussian_overlaps.py` to compute ideal mode amplitudes.

For each mode `b` and each point source `j`:

```text
a_{b j} = <phi_b | psi_j>
```

The total detected probability for mode `b` is then:

```text
p_b = (1/N) sum_j |a_{b j}|^2
```

### 2. Mode mixing
Introduce a fixed `B x B` mixing matrix `U` over the measured modes:

```text
a'_{m j} = sum_b U_{m b} a_{b j}
```

Recommended choices:
- **orthogonal/unitary `U`** for pure mode mixing,
- **slightly non-unitary `U`** if a simple loss/leakage model is needed.

### 3. Optional loss bucket
If the mode set is truncated, add a final loss outcome:

```text
p_loss = 1 - sum_m p'_m
```

This keeps normalization explicit and makes truncation physically interpretable.

## Derivatives

No new analytic derivation is required if `U` is parameter-independent.

For a parameter `theta`:

```text
partial_theta a'_m = sum_b U_{m b} partial_theta a_b
```

and

```text
partial_theta p'_m
  = (2/N) sum_j Re[ conj(a'_{m j}) * partial_theta a'_{m j} ]
```

This means the current analytic derivative logic can be reused after inserting the mixing step.

## Implementation Scope

### Files to touch
- `src/rayfish/classical_fisher_information/incoherent/fisher_information_matrix.py`
- `src/rayfish/gaussian_overlaps.py`
- `examples/distance_3_point_sources_hg_x/simulation.ipynb`
- `examples/distance_3_point_sources_hg_xy/simulation.ipynb`

### Recommended code changes
1. Add a helper that computes the ideal complex mode amplitudes for each mode and source.
2. Add a helper that applies a fixed mixing matrix `U` to those amplitudes.
3. Recompute probabilities from the mixed amplitudes.
4. Recompute derivatives using the same linear mixing rule.
5. Add an optional loss channel if the sorted mode set is truncated.

## Minimal Simulation Plan

### Parameters to sweep
- mode truncation size, e.g. `mode_degree` or `number_of_modes`,
- mismatch strength `epsilon`,
- random realization index for the mixing matrix,
- defect spacing `delta z`.

### Mixing matrix family
Use a one-parameter family such as:

```text
U(epsilon) = exp(epsilon A)
```

where `A` is a random skew-symmetric or anti-Hermitian matrix.

Alternative simpler construction:
- `U = (1 - epsilon) I + epsilon R`, then orthogonalize/unitarize.

### Monte Carlo procedure
For each `epsilon`:
1. draw `K` random `U` matrices,
2. compute the Fisher information / CRB,
3. aggregate mean and standard deviation across the `K` draws.

This gives error bars that quantify sorter variability.

## Plots to Produce

### Plot 1: Fisher information degradation
Show `FI_mismatch / FI_ideal` versus `epsilon`.

Purpose:
- demonstrates robustness or degradation of estimation performance under mismatch.

### Plot 2: CRB versus defect distance
For several mismatch strengths, plot the CRB or variance bound versus `delta z`.

Purpose:
- shows whether the sub-Rayleigh scaling advantage survives imperfect sorting.

### Plot 3: Mode power redistribution
For a representative `delta z`, show how the total probability is redistributed across modes as `epsilon` increases.

Purpose:
- provides an intuitive physical explanation of the FI change.

## Acceptance Criteria

The work is complete when:
- the ideal estimation example still runs unchanged,
- the imperfect-mode model can be enabled with a single flag or config object,
- probabilities remain normalized after mode mixing and optional loss handling,
- derivatives remain analytic and consistent with the existing FI code,
- at least one figure shows mean `+-` standard deviation over random mismatch realizations,
- the example is reproducible from the existing `distance_3_point_sources_hg_x` and/or `distance_3_point_sources_hg_xy` notebooks.

## Recommended Scope for the Paper

Keep the claim modest:
- the estimation advantage persists under finite mode truncation and controlled sorter mismatch,
- the reported curves are robust to a simple but physically meaningful model of imperfection.

Do not claim a full aberration or detector-noise model unless those are implemented separately.

## Unequal Intensities Plan

Add a second robustness study for source brightness imbalance.

### Goal
Model unequal source intensities as a known input distribution, not as additional parameters to estimate.

This keeps the inference problem unchanged while testing whether the estimation advantage survives when the source is not perfectly uniform.

### Minimal Model
For the three-source crack example, replace the equal weights

```text
I = [1/3, 1/3, 1/3]
```

by a perturbed distribution controlled by a single imbalance parameter `epsilon`.

One concrete choice:

```text
I_top    = 1/3 + epsilon/4
I_middle = 1/3 - epsilon/4
I_bottom = 1/3 - epsilon/2
```

This preserves normalization:

```text
I_top + I_middle + I_bottom = 1
```

and introduces an asymmetry that is easy to interpret physically.

### Interpretation
- the top source is slightly brighter,
- the middle source is slightly dimmer,
- the bottom source is dimmer still.

This is a simple brightness-gradient model, not a full source-estimation problem.

### Implementation Scope
Reuse the existing point-source coordinates and modify only the source weights in the mixture model:

```text
p(mode) = sum_j I_j |<phi_mode | psi_j>|^2
```

and analogously for derivatives:

```text
partial_theta p(mode) = sum_j I_j partial_theta |<phi_mode | psi_j>|^2
```

Because the intensities are assumed known, they only act as weights and do not add new estimated parameters.

### Why This Is Minimal
- no new optimization variables,
- no new Jacobian structure,
- no change to the Fisher information parameter dimension,
- no new analytic overlap derivations.

### Plot Idea
Produce an error-style robustness plot versus `epsilon`:
- x-axis: intensity imbalance `epsilon`,
- y-axis: FI ratio or CRB ratio relative to the equal-intensity baseline,
- optional shaded band if combined with random mode-mismatch realizations.

### Recommended Discussion Text
State that the equal-brightness assumption is a convenient idealization and that the code can accommodate known, nonuniform source intensities through fixed mixture weights. The first perturbative test uses a simple linear imbalance across the three sources, which is sufficient to show whether the estimation advantage is stable against moderate brightness asymmetry.

# Experimental Imperfections Execution Spec

## Summary
Implement this work in separate commits aligned to the major headers below. The sequence starts with baseline FI regression tests for the ideal model, then refactors the classical FI path to support amplitude-level reuse, then adds the imperfect-mode path, then adds the reproducible simulation example. No new dependencies are needed. Each commit must pass the existing pre-commit hooks, formatting and lint checks, and relevant test suite before moving to the next commit.

## Commit Plan
### Commit 1: Ideal FI regression tests
- Add classical Fisher information regression tests for the current ideal model only.
- Cover the 2-point-source invariance cases:
  - sweep x-separation at fixed z and assert the projected FI stays constant,
  - sweep z-separation at fixed x and assert the projected FI stays constant.
- Cover the 3-point-source edge cases:
  - vary only the symmetric in-plane pair with the z-displaced source fixed and assert the targeted projected FI stays constant,
  - vary only the z-displaced source with the in-plane pair fixed and assert the targeted projected FI stays constant.
- Use deterministic coordinate sets and explicit expected constants where available; otherwise assert constancy across the sampled family to tight tolerance.
- Do not change any existing tests.

### Commit 2: Classical FI amplitude refactor
- Refactor the classical incoherent FI implementation so ideal probabilities and derivatives are computed through shared internal amplitude helpers.
- Reuse existing overlap functions for:
  - complex amplitudes,
  - `s`, `t`, `p` amplitude derivatives.
- Preserve the current public function names and default behavior.
- Keep the ideal path numerically identical to the current implementation.

### Commit 3: Imperfect mode sorting core path
- Add the imperfect-mode module as a new additive module under the classical incoherent FI package.
- Gate activation with Hamilton config overrides using the existing node names:
  - `imperfect_mode_sorting=False` by default,
  - `True` switches to the imperfect implementation at runtime.
- Implement the unitary redistribution regime:
  - apply a parameter-independent mixing matrix over mode amplitudes,
  - recompute probabilities and derivatives analytically after mixing.
- Implement the lossy truncation regime:
  - support `include_loss_channel`,
  - define `p_loss = 1 - sum_m p'_m`,
  - define `dp_loss/dtheta = -sum_m dp'_m/dtheta`,
  - include the loss bucket in FI accumulation only when enabled.
- Add focused tests for:
  - identity mixing reproducing the ideal path,
  - norm-preserving mixing keeping total probability normalized,
  - loss-channel bookkeeping and derivative conservation.

### Commit 4: Reproducible imperfection simulation
- Create a new example folder with a Python script for the imperfection study.
- Use the existing 3-point-source setup and current repo dependencies only.
- Add seeded Monte Carlo generation of mixing matrices and sweeps over `epsilon`.
- Save results and plots from the script into the new example folder.
- Produce separate outputs for:
  - unitary redistribution,
  - lossy truncation with explicit `1 - p` loss capture.

## Implementation Details
### FI regression tests
- Put the new tests at the classical FI integration level, not at the overlap primitive level.
- Assert on the final Fisher information after Jacobian projection.
- Use small deterministic sweeps so the tests stay fast and stable.

### Imperfect-mode behavior
- The ideal measurement basis remains unchanged.
- Mixing occurs only at the amplitude level before squaring.
- The unitary case answers: how much information changes when power is redistributed among retained modes.
- The lossy case answers: how much information is left on the table when truncation prevents full capture of redistributed power.

## Plot Requirements
- Plot 1: Unitary redistribution
  - FI and/or CRB under norm-preserving mode mixing.
  - Shows the effect of mismatch when information is only reshuffled across measured modes.

- Plot 2: Lossy truncation
  - FI and/or CRB when unmeasured probability is assigned to `p_loss = 1 - sum p'`.
  - Shows the FI penalty from truncated detection, distinct from pure redistribution.

- Plot 3: Mode redistribution
  - Per-mode probability distribution across mode index for representative `epsilon` values.

## Assumptions
- Unequal intensities remain out of scope for this PR.
- Runtime selection uses Hamilton config overrides, not a separate public DAG branch.
- Existing tests must remain unchanged.

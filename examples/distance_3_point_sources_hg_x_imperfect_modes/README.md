# Imperfect Mode Sorting Simulation

This example provides a reproducible Python script for the imperfect mode-sorting
study described in `execution_spec_exp_imperfections.md`.

It produces two regimes:

- `unitary`: information is redistributed among the retained Hermite-Gaussian modes.
- `lossy`: a larger unitary is truncated back to the retained basis, and the missing
  probability is tracked via `1 - sum(p)`.

Run it with:

```bash
uv run python examples/distance_3_point_sources_hg_x_imperfect_modes/simulation.py
```

Outputs are written into the same example directory.

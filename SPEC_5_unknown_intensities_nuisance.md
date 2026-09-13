# Spec 5 — Unknown / fluctuating source intensities as nuisance parameters

## Goal
Give the **quantitative** backing for the part of Referee A1 / B3 that the current
appendix (`app:unequal_intensities`) only argues qualitatively: when the source
brightnesses are **unknown** (or fluctuate shot-to-shot), do they act as nuisance
parameters that degrade width/depth estimation, and does mode sorting (MS) still track
the quantum bound sub-Rayleigh?

Concretely, produce the **nuisance-adjusted** Cramér–Rao bounds for `δx, δz` when the
brightness contrast `r_c` is co-estimated rather than known, for both the quantum (QFIM)
and mode-sorting (classical FIM) bounds, and the resulting MS efficiency.

## Relationship to existing work (do not redo)
- **Spec 4 / `app:unequal_intensities` is done** for the *known* unequal case: the
  contrast sweep `r_c = I_mid/I_out` (`intensity="contrast"`), the closed-form QFIM and MS
  FIM threaded with `I_v`, and the four figures (info-vs-contrast, reallocation collapse,
  efficiency, FI↔CRB transfer). This spec is **Spec 4 Phase B + C only**, now that Phase A
  machinery exists.
- The current paper already states the *conditional* result: at **known** `r_c` the MS→QFI
  efficiency `H⁻¹/J⁻¹` is flat ≈ 1 sub-Rayleigh. Spec 5 turns the "unknown/fluctuating"
  sentence in the referee response (A 4a / B 3) from an argument into a computed curve.

## The physics, and how it is balanced by what we already have

**What "unknown intensity" changes.** With `r_c` known, the estimator's job is 2D
(`θ = (δx, δz)`) and the bound is `J_θθ⁻¹`. If `r_c` is unknown it becomes a third
parameter to be co-estimated; the relevant width/depth bound is the `θ`-block of the
*inverse of the enlarged FIM*, i.e. the **Schur complement**

```
J_θθ^marg = J_θθ − J_θη J_ηη⁻¹ J_ηθ ,   CRB_θ^unknown = (J_θθ^marg)⁻¹  ⪰  J_θθ⁻¹,
```

where `η = r_c` (single nuisance knob; `Σ_v I_v = 1` leaves one free contrast direction).
The inflation `CRB_θ^unknown / CRB_θ^known` is controlled entirely by the **cross-block
`J_θη`** — how much the geometry score and the brightness score overlap across channels.

**Why we expect the inflation to be small (the balance).** This is the quantitative
version of the argument already in the response:
- The brightness score `∂_r P_j` acts dominantly through total-flux / the fundamental
  `(0,0)` channel (changing `r_c` reweights how much light the central vs outer sources
  dump into the bright low-order modes).
- The geometry scores `∂_{δx} P_j`, `∂_{δz} P_j` act through the *information-bearing*
  higher orders `(1,0)` (width) and `(2,0),(0,2)` (depth) — the same modes the main text
  already identifies.
- Sub-Rayleigh these score vectors live in nearly disjoint channels, so `J_θη ≈ 0`, the
  Schur complement `≈ J_θθ`, and marginalising the unknown brightness costs almost
  nothing. Spec 5 **measures** `J_θη` and the inflation factor instead of asserting it.

**Detection.** The `δz²` Chernoff scaling is structural — it comes from the activation of
the `(2,0),(0,2)` channels whose occupancy grows as `δz²`; a per-source weight only
multiplies the channel and cannot move the `δz²` onset. Unknown brightness makes detection
a *composite* test, but the scaling argument is weight-independent. Detection recomputation
(the `|±>` parity simplification in `app:QCB_calculation` may break under asymmetry → fall
back to numerics) is **out of scope here** and stays with Spec 4; Spec 5 is estimation.

## Code changes

The known-unequal path already weights `P_j` and `∂_coord P_j` by `point_source_intensities`.
The only genuinely new ingredient is the **derivative with respect to the intensity knob**,
and the **marginalisation**. The intensity derivative is essentially free analytically:
since `P_j = Σ_v I_v |⟨φ_j|ψ_v⟩|²`, `∂P_j/∂I_v = |⟨φ_j|ψ_v⟩|²` is exactly the *unweighted*
per-source overlap already computed inside `probability__1D`.

### Classical (mode sorting) — `classical_fisher_information/fisher_information_remainder_channel.py`
1. **New node `intensity_derivatives_probability`** (mirror of `derivatives_probability__1D`):
   return `∂P_j/∂I_v = p_{v,j}` (the unweighted overlap), shape `(M, B, V)`. The remainder
   channel derivative is `∂P_rem/∂I_v = −Σ_b ∂P_b/∂I_v`, same bucketing as the coordinate path.
2. **Chain to the contrast knob.** With `I_v(r_c)` from `intensity_distribution`, form
   `∂P_j/∂r_c = Σ_v (∂P_j/∂I_v)(dI_v/dr_c)`. Keep a single nuisance direction `r_c`
   (recommended) to match the appendix; the general `V−1` independent-weight case is a
   later generalisation.
3. **Assemble the enlarged FIM** in the same Poissonian form already used,
   `J_{a,b} = τ Σ_j ∂_a P_j ∂_b P_j / (τ P_j + b_j)`, but with the index `a,b` ranging over
   `[δx, δz, r_c]` (after the `coordinate_jacobian` maps the `3N` coordinate scores to
   `δx, δz`, append the `r_c` score row/column directly — it is not a coordinate, so it
   bypasses the jacobian). Cleanest: build the `3` blocks `J_θθ` (existing), `J_θη`, `J_ηη`.
4. **New node `covariance_matrix_marginalised`**: return both `J_θθ⁻¹` (known) and the
   Schur-complement bound `(J_θθ − J_θη J_ηη⁻¹ J_ηθ)⁻¹` (unknown). The existing
   `covariance_matrix_for_entire_distribution` stays as the known-intensity result.

### Quantum (QFIM) — `quantum_fisher_information/incoherent_sources/`
1. **`operator_matrices.density_derivatives`**: currently emits only the `3N` coordinate
   derivative matrices `D_a` (sparse, scaled by `I_v`). Add the **intensity derivative(s)**
   `∂R/∂I_v = |ψ_v⟩⟨ψ_v|`, i.e. the diagonal `R_Δ`-block direction (different sparsity from
   the coordinate `D_a`, which couple `ψ_v` to `∂ψ_v`). For the single knob, emit
   `∂R/∂r_c = Σ_v (dI_v/dr_c)|ψ_v⟩⟨ψ_v|`. Generalise the sparse-entry construction so the
   downstream loop accepts these extra directions.
2. **`lyapunov_operator.lyapunov_operators`**: already loops over `density_derivatives[i]`
   solving the reduced Sylvester equation for each SLD direction, so it extends to the new
   intensity directions for free **provided** the reduced-block bookkeeping (the
   `diag(I_1,…,I_V,0)` zero eigenvalue note) still holds for the `|ψ_v⟩⟨ψ_v|` direction —
   verify, since that derivative does not enlarge the support the way coordinate
   derivatives do.
3. **`quantum_fisher_information_matrix.fisher_information_matrix_for_entire_distribution`**:
   build the enlarged QFIM over `[δx, δz, r_c]` (jacobian on the coordinate block,
   intensity row/col appended), then add a marginalised-covariance node identical in spirit
   to the classical Schur complement.

### Parameter catalogue — `parameter_catalogue/intensity_distribution.py` (+ `common_transforms`)
1. Add a node `intensity_contrast_jacobian` returning `dI_v/dr_c` for the `contrast`
   parametrisation (`I_other = 1/(r+V−1)`, `I_target = r·I_other` ⇒
   `dI_target/dr = (V−1)/(r+V−1)²`, `dI_other/dr = −1/(r+V−1)²`).
2. Decide how `r_c` joins the estimated-parameter vector (extend `coordinate_jacobian`
   to a block-diagonal `[coordinate-block ⊕ 1]`, or carry the intensity score separately
   and combine at FIM-assembly time — the latter avoids touching `distance_3_sources_xz`).

### Driver / simulation — `examples/3_point_sources/ideal_unequal_intensities/simulate_bounds.py`
1. Request the new marginalised-covariance `final_vars` for both classical and quantum
   drivers, save as `*_inv_marg_*.npy` alongside the existing known-intensity arrays.
2. No new sweep needed — reuse the existing `CONTRASTS` grid; the nuisance result is a
   second curve on the same axes.

### Fluctuating intensities (Phase C, model-dependent, smaller)
- **Static-but-unknown across the ensemble** = exactly the nuisance computation above
  (marginalised bound); no new code.
- **Genuine shot-to-shot fluctuation** with a chosen distribution `p(r_c)`: add a node that
  averages the channel means `⟨λ_j⟩ = τ⟨P_j(r_c)⟩` (fast fluctuations) or averages the FIM
  over `p(r_c)` (slow), then reuses the same inversion. Because every bound is smooth in
  `r_c` (shown by the appendix), symmetric zero-mean fluctuations perturb the bounds only at
  **second order** — so this is expected to be a small correction; the deliverable is to
  confirm the size.

## Deliverables (figures / numbers)
1. **Nuisance-adjusted CRB vs contrast**: overlay `CRB_θ^known` (have) and
   `CRB_θ^unknown` (Schur complement) for width and depth, vs `r_c`, sub-Rayleigh. The gap
   *is* the cost of not knowing the brightness.
2. **MS efficiency under unknown brightness**: `(J_θθ^marg)⁻¹ / (H_θθ^marg)⁻¹` vs `r_c`,
   to be compared against the known-intensity efficiency curve already in
   `ms_efficiency_vs_contrast.pdf` — the claim is that it stays flat and close to unity
   sub-Rayleigh.
3. **The cross-block `J_θη` (or the inflation factor) vs separation**: the direct,
   physical readout that brightness and geometry scores are near-orthogonal sub-Rayleigh.
4. (Optional, Phase C) bound vs fluctuation amplitude for a chosen `p(r_c)`.

## How to interpret the outputs (physics mapping)
- **`CRB_θ^unknown ≈ CRB_θ^known` sub-Rayleigh** ⇒ unknown brightness is (nearly) free to
  marginalise; the protocol does not require calibrated intensities. This *upgrades* the
  current response sentence to a measured statement.
- **Efficiency curve stays flat ≈ 1** ⇒ MS still attains the (nuisance-adjusted) quantum
  limit — i.e. the central claim of the paper survives the unknown-brightness relaxation,
  not just the known one.
- **`J_θη` small** ⇒ confirms, in code, the "brightness lives in the fundamental mode,
  geometry in the higher orders" decoupling that the main text already invokes for the
  mode structure — same physics, now cross-checked from the score vectors.
- **Inflation growing near the Rayleigh scale** is acceptable and on-message: that is where
  the modes overlap, the decoupling weakens, and direct imaging is preferable anyway.

## Open questions to pin down before starting
1. Single contrast knob `r_c` (recommended, matches appendix) vs fully independent
   `V−1` weights. Start with `r_c`.
2. Quantum side: confirm the reduced Sylvester/`lyapunov_operators` bookkeeping is valid
   for the `∂R/∂I_v = |ψ_v⟩⟨ψ_v|` direction (zero-eigenvalue block) before trusting the
   enlarged QFIM; otherwise fall back to a finite-difference QFIM in `r_c` as a check.
3. Phase C fluctuation model: distribution shape, and fast (average means) vs slow
   (average FIM) regime.
4. Whether to also co-estimate the *overall* flux `τ` (usually a benign global scale,
   separable from `r_c`); likely leave fixed.

## Effort
Moderate, and **much smaller than Spec 4 Phase A** because the weighted `P_j`, `∂_coord P_j`,
`R`, and coordinate `D_a` already exist. New work is: one intensity-derivative node per side,
the `dI_v/dr_c` jacobian, the Schur-complement covariance node, and (quantum) generalising
the sparse derivative-matrix construction + verifying the SLD solve. No new sweep, no new
parameter geometry. Phase C is additive and optional.

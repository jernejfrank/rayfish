# Spec 4 — Unequal / unknown / fluctuating source intensities (priority 4, heavy)

## Goal
Relax the equal-intensity assumption `I_v = 1/V` and answer Referee B3 / A1 head-on:
- Does the closed-form QFIM survive for **unequal but known** intensities?
- If intensities are **unknown**, do they act as nuisance parameters that degrade
  width/depth estimation?
- If intensities **fluctuate**, what is the effect on estimation and detection?
- **Does MS remain (near-)optimal in the sub-Rayleigh regime, and does the `δz²` detection
  scaling survive?**

## Why this is the heavy one
Unlike Specs 1–3 (which reuse existing `P_j`, `∂P_j`), this touches the **core
derivations and the density-matrix construction**:
- the closed-form QFIM derivation (App. `qfi_derivation`) assumed `R_Δ = (1/V) 𝟙_V`;
- the fidelity / QCB derivation (App. `QCB_calculation`) used equal `1/3` weights and the
  L/R parity symmetry (`|±>` states);
- the classical FIM (App. `ms_incoherent`) sums per-source probabilities with `1/V`.
Both analytic re-derivation **and** code changes (per-source intensities) are required.

## Current state (what changes)
- Ensemble: `R = Σ_v I_v |ψ_v><ψ_v|`, currently `I_v = 1/V` (note `noteintensity`,
  Eq. `incoherent_ensemble`).
- QFIM derivation uses `R_Δ = (1/V)𝟙` and sparse `D_a^{∂u,u} = 1/V`.
- MS probabilities: `P_{j,l} = (1/V) Σ_v |<φ_{j,l}|ψ_v>|²`.
- QCB: `ρ_0, ρ_1` with equal `1/3` weights; block-diagonal `M` via `|±>` parity.

Replacing `1/V → I_v` (and `1/3 →` per-source weights) propagates through all of these.

## Parameterisation (3-source crack)
By the L↔R symmetry of the model, start minimal:
- edges `I_L = I_R = I_e`, floor/centre `I_Z (or I_C) = I_f`, with `2 I_e + I_f = 1`
  ⇒ a **single asymmetry knob** (e.g. ratio `r = I_f / I_e`).
- Generalise to fully independent `I_v` only if needed.
Physical motivation: the floor source (deeper, `-δz`) may collect less light than the
edges — a natural brightness asymmetry tied to depth.

## Phased plan

### Phase A — Known unequal intensities (extension, moderate)
1. Re-derive the QFIM with `R_Δ = diag(I_1,…,I_V)` (still diagonal, invertible). Sparsity
   becomes `D_a^{∂u,u} = D_a^{u,∂u} = I_u`; the reduced Sylvester eq.
   (`L_Δ G_Δ + G_Δ L_Δ = …`) carries `I_v` factors. **Expectation:** still closed-form
   with `1/V → I_v` threaded through — confirm.
2. Thread `I_v` into the classical FIM probabilities and the QCB `ρ_0, ρ_1`.
3. Re-run FI/CRB (estimation) and Chernoff (detection) vs the asymmetry knob `r`.
4. **Deliverable:** does MS still approach QFI for width/depth, and does the `δz²/12`
   Chernoff scaling survive, as `r` departs from 1? (The `|±>` parity simplification in
   App. QCB may break — redo or fall back to numerics.)

### Phase B — Unknown intensities as nuisance parameters (the core B3 question)
1. Enlarge the parameter set to include the intensity knob(s); add `∂P/∂I_v` (and the
   QFIM rows/columns for intensities).
2. Obtain the width/depth bound **marginalising** the unknown intensities — i.e. the
   Schur complement of the intensity block (or invert the full QFIM and read the
   width/depth sub-block).
3. **Deliverable:** quantify how much width/depth precision degrades when intensities are
   unknown vs known; check whether MS still tracks the (nuisance-adjusted) quantum bound
   sub-Rayleigh.

### Phase C — Fluctuating intensities (model choice)
1. Model `I_v` as random (shot-to-shot) with a chosen distribution; average the FI
   (and/or the states for detection) over fluctuations.
2. **Deliverable:** effect of intensity fluctuations on estimation precision and on
   crack detectability; does the MS advantage persist?

## Code touch-points (to confirm when starting)
- `parameter_catalogue.distance_3_sources_xz` — add per-source intensity inputs.
- density-matrix / ensemble construction (`rayfish`) — weight by `I_v` not `1/V`.
- `classical_fisher_information.incoherent` — `1/V → I_v`; add `∂/∂I_v` for Phase B.
- QCB construction — per-source weights in `ρ_0, ρ_1`.

## Open questions to pin down
1. Asymmetry parameterisation: single ratio `r` (recommended start) vs fully independent
   `I_v`.
2. For Phase B, are intensities a true nuisance (unknown, to be marginalised) or assumed
   calibrated? B3 phrasing ("unknown or fluctuating") ⇒ both Phase B and C are in scope.
3. Phase C fluctuation model (distribution, correlated vs independent across sources).
4. Whether the `|±>` parity simplification in App. QCB survives asymmetry, or we go
   numerical.

## Referee relevance
- Directly answers Referee B3 (unknown/fluctuating intensities, MS optimality
  sub-Rayleigh) and Referee A1 (unequal intensities robustness).
- This is the item flagged as `[To be completed]` in `response.md` (R2, point 3) and
  referenced in the new main-text outlook paragraph.

## Dependencies / sequencing
- Independent of Specs 1–3 (those reuse existing machinery; this changes the core).
- Heaviest of the four — analytic re-derivation + core code changes + new parameters.
- Do **last**; consider doing Phase A alone first as a self-contained robustness result.

## Effort
Large relative to Specs 1–3. Phase A is a contained extension; Phases B–C add nuisance/
fluctuation modelling and are the substantive new analysis.

# Spec 3 — Detector noise floor (dark counts + background light) (priority 3)

## Goal
Model genuine **detector noise** (dark counts, background light) as a θ-independent
additive count floor per channel, and quantify how it degrades estimation — with the
headline result being a **MS-vs-DI comparison under a noise floor**.

## What is / isn't currently modelled
The current FI (Eq. `eq:FI_MS`),
```
(J_Π)_{a,b} = τ Σ_j (1/P_j(θ)) ∂_a P_j ∂_b P_j ,
```
is the Fisher information for **Poisson photon counts** with channel mean
`λ_j = τ P_j(θ)`. What it captures:
- total photon budget τ (prefactor),
- the `1/P_j` inverse-variance weighting = **photon shot noise** (variance = mean).

Shot noise is therefore the **only** noise currently present. NOT modelled: dark counts,
background light, readout/electronic Gaussian noise, excess noise.

Note: *uniform* detector inefficiency η is benign — it thins the Poisson process,
`λ_j → η τ P_j`, i.e. rescales `τ → η τ`; it scales the FI but does not change the
MS-vs-DI structure. (This is the only "loss" the Poisson model legitimately absorbs.
The earlier referee-response wording conflated this with detector noise and must be
softened — see "Doc cleanup" below.)

## Model
Add a θ-independent floor `b_j` (dark counts + background) to each channel mean:
```
λ_j = τ P_j(θ) + b_j
```
Since `b_j` is θ-independent, `∂_a λ_j = τ ∂_a P_j`, but it enters the **denominator**:
```
J_{a,b} = τ² Σ_j  (∂_a P_j ∂_b P_j) / (τ P_j + b_j)
```
- Recovers the current formula as `b_j → 0`.
- One-line change to the FI assembler; reuses existing `P_j`, `∂P_j`.
- Simplest parameterization: uniform background `b_j = b` (single knob), optionally a
  per-channel `b_j` if a realistic background profile is wanted.

## Why it matters physically (headline)
The floor sits in the denominator, so it preferentially kills channels where `P_j` is
**small** — exactly the information-bearing high-order modes:
- depth info lives in `(2,0)/(0,2)` with `P ≈ δz²/24` (tiny sub-Rayleigh) ⇒ a noise floor
  swamps those modes first ⇒ depth estimation degrades before width.
- So detector noise attacks MS precisely where its advantage originates.

Genuine tension to quantify (MS vs DI under floor `b`):
- MS concentrates signal into few modes ⇒ high per-channel SNR for dominant modes,
- but the depth channels are intrinsically faint ⇒ most vulnerable to a floor,
- DI spreads signal over many pixels, each with its own background.

This makes "MS vs DI under a noise floor" a real robustness result, not box-ticking.

## Deliverable
- Background term wired into the FI denominator (`τ P_j + b_j`).
- MS-vs-DI FI / CRB curves vs the noise floor `b` (sweep), for width δx and depth δz.
- Short figure + paragraph (Supplemental Material), highlighting that depth (faint
  second-order modes) degrades first.

## Scope / non-goals
- **Readout / Gaussian noise** is a *different* model (Poisson → Gaussian,
  `J = Σ_j ∂_aλ_j ∂_bλ_j / σ_j²`). Out of scope here; the additive-background Poisson
  version above is the minimal, most defensible extension. Mention as possible follow-up.

## Open questions to pin down
1. Uniform floor `b` vs per-channel `b_j` (and for DI, per-pixel background). Start with
   uniform `b` for a clean single-knob comparison.
2. Units/normalisation of `b` relative to τ (express as a background-to-signal ratio or
   counts per channel so the sweep is interpretable).
3. For DI, define the per-pixel background consistently with the MS per-channel `b` so the
   comparison is fair (same total background power? same per-mode/pixel rate?). This
   choice drives the MS-vs-DI conclusion — decide and state it.

## Doc cleanup (response.md, A3)
Current sentence overclaims: "Detector photon loss is already accounted for through the
Poissonian counting statistics." Revise to claim only what's true:
- shot noise **is** captured (the `1/P_j` weighting),
- uniform inefficiency η rescales τ,
- dark counts / background / readout are addressed separately (point to Spec 3 result
  once done).

## Dependencies / sequencing
- Reuses existing `P_j`, `∂P_j` (same machinery as Specs 1–2).
- Pairs naturally with Spec 2: the bucket port also has its own floor `b_bucket`.
- Independent of the unequal-brightness work.
- Do after Specs 1–2.

## Referee relevance
- Directly addresses Referee A3 ("detector noise") and Referee A's "background light".
- Strengthens B5 (experimental realism).

## Effort
Small — one term in the FI denominator + a sweep over `b` and a figure. Comparable to
Specs 1–2.

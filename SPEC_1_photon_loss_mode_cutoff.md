# Spec 1 — Photon loss due to mode cutoff (priority 1, "free lunch")

## Goal
Quantify and report the fraction of collected signal that falls **outside** the measured
Hermite-Gaussian (HG) mode bank when the sorter is truncated at a cutoff (e.g. the
`(2,2)` corner). This is a measurement-design loss, not optical/detector loss.

## Definition
For the three-source crack model, the leaked fraction is
```
1 - p(δx, δz),   with   p(δx, δz) = Σ_{(j,l) ∈ measured}  P_{j,l}(δx, δz)
```
and lost photons = `N · (1 - p)` (equivalently rate `τ · (1 - p)`).

`P_{j,l}` are the per-channel detection probabilities already used in the paper
(Eq. for `P_{j,l}`, closed form in App. "Classical FI for an arbitrary polynomial
measurement basis", Eq. `three_sources_incoherent_ms_prob`).

## Why it's nearly free
- The simulation already computes 16 modes (`MODE_DEGREE = 4`, orders 0–3 in each of
  x,y), so a `(2,2)` cutoff is just summing a **subset** — no new mode evaluations.
- Completeness gives `Σ_{all} P_{j,l} = 1` per source even for a **defocused** source
  (in-focus HG modes span the full transverse space; the cross-talk rotation `U(ε)` is
  orthonormal). So `1 - p` is a well-defined tail.
- Likely a **closed form**:
  - lateral displacement → HG-order content is **Poisson** (mean ∝ δx², consistent with
    `P_{1,0} ≈ δx²/12`) ⇒ tail beyond cutoff is a Poisson tail.
  - defocus → squeezing-like, even modes only ⇒ its own closed-form tail.
  Target: analytic "fraction lost vs cutoff", not only a numerical map.

## Deliverable
- A `1 - p` map (or curve) over `(δx, δz)` for the chosen cutoff.
- Optionally the closed-form Poisson/even-mode tail expression.
- One short figure + paragraph (Supplemental Material likely) reporting the leaked
  fraction in the sub-Rayleigh regime.

## Framing / pitfalls
- **Not optical loss.** Uniform detector inefficiency η is already absorbed into the
  Poissonian rate τ (rescales all `P_j` equally, does not change MS-vs-DI). This quantity
  is sorter **completeness/efficiency** = fraction of collected signal the finite bank
  uses. Keep the two distinct so a referee does not conflate them.
- Distinguish from Spec 2's "information loss": photon-fraction loss `1-p` ≠
  Fisher-information loss `H - J`. You can leak many photons that carry little
  information, and vice-versa. Present both notions as complementary.

## Open questions to pin down
1. Cutoff shape: box (`j,l ≤ 2`, 9 modes) vs total-order (`j+l ≤ 2`, 6 modes). Both
   contain the info-bearing modes `(1,0),(2,0),(0,2)`; "box" matches the `(2,2)`-corner
   language. Pick one and state it.

## Referee relevance
- Directly answers Referee A3 ("sensitivity to mode truncation / loss").
- Supports B5 (experimental realism).

## Effort
Very small — subset sum over already-computed probabilities + one plot; optional analytic
tail. Do this **first**.

# Spec 2 — All-other-modes "bucket" measurement channel (priority 2)

## Goal
Model a realistic sorter as the finite HG mode bank **plus one "everything else" output**
(a dump/bucket port), and add this bucket channel to the **estimation** Fisher
information (protocol A). Show the resulting improvement (MS curve moves toward the QFI),
especially at larger separations.

## Setup
Completing the POVM:
```
{Π_{j,l}}_measured  ∪  {Π_bucket},   Π_bucket = 1 - Σ_measured Π_{j,l}
```
with bucket outcome probability `1 - p`, `p(θ) = Σ_measured P_{j,l}(θ)`.

This is exactly the remainder channel already used in the **detection** section
("we implicitly include a remainder channel with probability 1 - Σ P_{j,l}"). Spec 2 ports
that pattern into the **estimation** FI. The code pattern already exists.

## FI contribution
Classical FI is additive over channels, so the bucket adds one term:
```
ΔJ_ab = (1 / (1 - p)) · (∂p/∂θ_a) · (∂p/∂θ_b),   ∂p/∂θ_a = Σ_measured ∂P_{j,l}/∂θ_a
```
Both `p` and `∂p/∂θ` come from quantities already computed.

## Expected behaviour (the result we want)
- **Monotone help:** adding a channel never lowers classical FI ⇒
  `J_trunc ≤ J_trunc+bucket ≤ J_all-resolved ≤ H` (QFI).
- **Deep sub-Rayleigh:** info sits in modes *inside* the `(2,2)` cutoff
  (`(1,0)` width; `(2,0)/(0,2)` depth), so `1-p` and `∂p` are high-order small ⇒ bucket
  adds ~nothing. This *demonstrates* truncation at `(2,2)` costs ~no information
  sub-Rayleigh.
- **Larger δx, δz:** real power leaks out, MS currently falls away from QFI; the bucket
  recovers part of it and **extends the useful range** of MS.

## Important caveat
The bucket carries only the **aggregate (zeroth-moment)** information of the tail (total
leaked power), not the distribution across higher modes. So
```
ΔJ_bucket  ≤  (J_all-resolved − J_trunc)
```
i.e. `ΔJ_bucket` is a **lower bound** on the full higher-order-mode information — not the
entire gap to QFI. State it this way; do not equate bucket-info with "all info in higher
modes". (It is the right experimental object: count the dump port, no need to resolve
high modes.)

## Numerical gotcha
`ΔJ ~ (∂p)² / (1 - p)` is a `0/0` as δ→0 (both vanish; finite analytically). Guard the
δ→0 limit — we have already hit a divide-by-zero edge case here (commit `5d4d629`).

## Deliverable
- Bucket term wired into the estimation FI assembler; regenerate Fig. (a)/(b) MS curves
  (or a separate panel) showing the improvement.
- Optionally overlay `ΔJ_bucket` to visualize recovered tail information.

## Open questions to pin down (BEFORE coding)
1. **Does protocol A currently include any remainder/renormalization?** If estimation
   sums the 16 modes *without* a bucket, adding it is a strict change; if it renormalizes,
   keep it consistent. → check the FI assembler first (read-only).
2. Cutoff shape (shared with Spec 1): box `j,l ≤ 2` vs total-order `j+l ≤ 2`.
3. Presentation: extra curve on existing Fig. (a)/(b) vs separate panel.

## Referee relevance
- With Spec 1, answers Referee A3 (mode truncation) head-on.
- Strengthens B5 (experimental realism): modeling the dump port makes protocol A a
  faithful model of a real device, not just an add-on.

## Dependencies / sequencing
- Reuses the detection-side remainder code and the same `p, ∂p` as Spec 1.
- Independent of the unequal-brightness work — safe to land before it.
- Do **after** Spec 1.

## Effort
Small — wire the complement term into the estimation FI assembler + regenerate one
figure. ~Half a day, mostly plotting.

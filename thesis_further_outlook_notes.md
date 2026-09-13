# Further outlook notes — nuisance intensities, compatibility, and adapted measurements

Notes from the discussion following Spec 6 (`SPEC_6_effective_sld_adapted_povm.md`).
Material for the thesis outlook chapter: why the unknown-brightness problem has the
structure it has, what is fundamental vs fixable, and how far the closed-form
machinery stretches.

## 1. Why the D-form QFIM carries no compatibility information

Two equivalent-looking expressions for the QFIM are not equally informative.
Substituting the SLD equation `2∂_b ρ = ρΛ_b + Λ_bρ` into the D-form and using
cyclicity of the trace:

```
Tr(Λ_a ∂_b ρ) = (1/2) Tr(ρ {Λ_a, Λ_b}),
```

i.e. the D-form is identically the **anticommutator half** — real by construction.
The LL-form splits as

```
Tr(ρ Λ_a Λ_b) = (1/2) Tr(ρ {Λ_a, Λ_b})   [real, symmetric in a,b  → QFIM]
             + (1/2) Tr(ρ [Λ_a, Λ_b])   [imaginary, antisymmetric → compatibility D_ab].
```

(Conjugation swaps `a ↔ b`: the symmetric combination is self-conjugate hence real;
the antisymmetric one flips sign hence imaginary.) The commutator piece — the
obstruction to jointly attaining the bounds for `θ_a` and `θ_b` — cancels
*identically* in `Tr(Λ_a ∂_b ρ)`, because `∂_b ρ` is built from the anticommutator and
only ever probes the symmetric part of the score algebra. It is not lost to numerics;
it is projected out by construction.

Consequence for the code: the complex output of the D-form node is
`QFIM + i·(pinv noise)`. The compatibility matrix requires the genuinely different
contraction `Im Tr(R G L_a G L_b G)` (Spec 6, Part 1).

## 2. The general recipe: Gram–Schmidt on score operators

Everything in the nuisance treatment is orthogonalisation of the score operators in
the inner product `⟨A, B⟩_ρ = Re Tr(ρ A B)`:

- effective SLD `L_a^eff = L_a − (H_aη H_ηη^{-1}) L_η` = residual of the projection of
  the interest score off the nuisance scores (the quantum efficient influence);
- marginal QFIM (Schur complement) = Gram matrix of the residuals.

This is quantum semiparametric estimation (Suzuki's nuisance-parameter QCRB;
Tsang–Albarelli–Datta). Three caveats keep it from being a universal improve-stuff
button:

1. **Accounting, not elimination.** The Schur complement is always ≤ the plain block:
   orthogonalising never *gains* information; it computes the correct, smaller amount
   that survives not knowing the nuisance. Adapted measurements improve *attainment*
   of that bound, never the bound. (Likewise, diagonalising the FIM by
   reparametrisation is cosmetic — no CRB for the original parameters changes.)
2. **Per-parameter, not joint.** Measuring the eigenbasis of `L_a^eff` attains the
   marginal bound for that one parameter. Effective scores of different parameters
   are each orthogonal to the nuisance score but generally not to each other, and
   `[L_a^eff, L_b^eff]` need not vanish on supp(ρ) — compatibility gates joint
   attainment (hence Spec 6 Part 1 computes `c = ‖iH⁻¹D‖`; Holevo lies between
   SLD-CRB and `(1+c)·`SLD-CRB).
3. **Local.** Scores depend on the true parameter point, so the adapted basis does
   too → two-stage/adaptive protocol; a fixed device realises the gain only near its
   design point.

The genuinely quantum part: classically the statistician can only reparametrise — the
experiment is fixed. Quantum mechanically the measurement is a design variable, so
orthogonalising scores translates into a physically different POVM. It also explains
why nobody noticed the issue in the known-brightness case: sub-Rayleigh, the HG modes
*are* the eigenmodes of the plain SLDs. Unknown brightness is the first time the basis
and the scores stop coinciding.

## 3. Reparametrisation invariance and the three escape hatches

The QCRB for a fixed *function* of the parameters is reparametrisation-invariant:
estimating an exotic `φ = f(δx, δz, r_c)` and inferring depth through a deterministic
map hands back exactly the original depth bound (`H̃ = B H Bᵀ`, `∇gᵀ H⁻¹ ∇g`).
Renaming coordinates never moves information. Only three things do:

1. **Different question.** Sub-Rayleigh the identifiable combinations are the moments
   `m_x = I_out δx²`, `m_z = I_mid δz²` — these are estimable at the quantum limit
   *without* brightness knowledge (they are what the channels measure:
   `P_{2,0} ∝ I_mid δz²`). The flat direction is precisely "trade brightness against
   depth² at fixed product"; no parametrisation repairs it because it is a genuine
   flat direction of the statistical model. If the task tolerates a
   brightness-weighted answer — detection does: `m_z ≠ 0` is the alarm — the intensity
   problem evaporates. Clean framing for the appendix: **MS estimates the moments
   `I_v δ²` at the quantum limit regardless of brightness knowledge; converting
   moments to geometry requires brightness information, and that conversion cost is
   fundamental (it inflates the QFI bound too, not just MS).**
2. **Different model.** A physical link `I_mid(δz)` (emission model) or a prior on
   `r_c` (calibration with error bars → van Trees) shrinks the parameter space and
   genuinely improves the bound: the constraint removes the flat direction rather
   than renaming it. Practically relevant: nobody has zero brightness knowledge.
3. **Different physics.** Illumination control, a reference beam, or an independent
   brightness monitor (bucket/defocused tap) breaks the degeneracy at the source.

## 4. The closed-form machinery survives hatches 1 and 2

Worry: constrained models `I(δz)` or moment coordinates entangle intensity with
position, so the intensity "cannot be taken out of the integral" and the Gaussian
overlap closed forms break. False — intensities were never inside any integral. The
non-orthogonal-basis formalism separates:

- **geometry**: states `|ψ_v⟩` and Gram matrix `G` (closed-form overlaps; depend only
  on displacements);
- **statistics**: coefficient objects `R_Δ = diag(I_v)`, derivative matrices `D_a`,
  Jacobian `B` (all intensity dependence lives here as plain numbers).

Hatch 1 (moments) only changes `B` (chain rule over existing coordinate + intensity
derivative matrices). Hatch 2 (constrained model) makes the depth score
`D_{δz} + (dI_mid/dδz)·∂R/∂I_mid` — a sum of two coefficient matrices that already
exist. Same classically: `P_j = Σ_v I_v |⟨φ_j|ψ_v⟩|²` has `I_v` outside the integral
already. Only hatch 3, in its aggressive variants (structured illumination replacing
the Gaussian PSF), sends us back to numerical overlap integrals.

Caveat on moment coordinates: `m ∝ δ²` makes the map singular at `δ = 0`
(`∂m/∂δ → 0`). Not a breakdown — the same physics restated (moments are the good
coordinates *because* δ degenerates) — but do the analysis in moment coordinates near
the origin rather than converting bounds back and forth.

## 5. Product rule, the budget constraint, and sparsity

For `ρ = Σ_v I_v(θ) |ψ_v(θ)⟩⟨ψ_v(θ)|`, each derivative matrix has two pieces with the
two known sparsity patterns: a coordinate piece (γ-block entries of the displaced
source) and an intensity piece (Δ-diagonal entries). The `Σ_v I_v = 1` constraint is
never imposed separately — build it into the parametrisation `I_v(η)` and it lives in
the Jacobian row `dI_v/dη`, which sums to zero automatically. (That trace-free
property is exactly why the zeroth-order channel terms cancel in the degeneracy
identity of §3.)

Refinement on "derivatives isolate individual sources": the normalisation coupling
*does* leak into the derivative matrices. If `I_mid` depends on `δz`, the fixed budget
forces `∂I_out/∂δz ≠ 0`, so the intensity piece of the `δz` direction has Δ-diagonal
entries for **all** sources (same shape as the `r_c` direction). What stays
single-source is the γ-block part: displacing source v never generates tangent
entries for source u.

Implementation bonus: the Lyapunov equation `2D = LGR + RGL` is **linear** in
`(D, L)`. For a mixed direction `D = D_coord + (dI/dδz)·D_intensity` the SLD is the
same linear combination of the separately solved SLDs — no mixed-direction solver
needed. Chain rule commutes with the entire pipeline: overlaps untouched → D-matrices
linear combos → SLDs same combos → FIM Jacobian sandwich.

## 6. Empirical anchors (from the Spec 5 saved arrays, δx ≈ δz ≈ 0.3, r_c = 1)

- Brightness score collinearity with geometry scores: MS `R² ≈ 0.985–0.993`;
  QFI `R² ≈ 0.53–0.71`.
- Known → marginal CRB: QFI `3.0 → 5.4` (width), `3.1 → 8.6` (depth);
  MS `3.1 → 67`, `3.2 → 147`.
- Inflation vs separation is non-monotonic and **physical**: the confounding
  coefficient of the width score is `∝ I_out δx/2`, so width inflation → small as
  `δx → 0` (odd-mode intuition holds in the limit), peaks near `δx ≈ 0.4`, decays
  super-Rayleigh. The Spec 5 working point sits in the maximally confounded region.
- Leading-order degeneracy identity, any intensity measurement (MS or DI):
  `∂P_j/∂r_c = −(I_out δx/2) ∂P_j/∂δx + (I_out²/I_mid) δz ∂P_j/∂δz + O(δ⁴)`.

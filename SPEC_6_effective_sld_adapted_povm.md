# Spec 6 — Compatibility matrix + effective-SLD-adapted mode sorting

## Goal
Two follow-ups to the Spec 5 result (unknown brightness is expensive sub-Rayleigh and
the 4-mode MS bank sits far above the marginal quantum bound):

1. **Split the QFIM and the compatibility matrix** and save both, so we can decide
   whether the nuisance-adjusted SLD bound is attainable at all — i.e. how much of the
   Fig 2/4 gap is MS being suboptimal vs the benchmark being unreachable.
2. **Build a nuisance-adapted POVM from the effective SLDs**: extract a unitary rotation
   of the truncated HG mode space from the SLD eigenbases (mechanically like the
   cross-talk rotation in `app:ms_errors`, but derived from the SLDs instead of a
   beam-splitter chain) and measure whether it recovers the marginal quantum limit.

## Background / what Spec 5 established
Sub-Rayleigh there is an exact leading-order degeneracy for *any* intensity
measurement: for every channel,

```
∂P_j/∂r_c = −(I_out δx / 2) ∂P_j/∂δx + (I_out²/I_mid) δz ∂P_j/∂δz + O(δ⁴),
```

so the brightness score is ~98–99% collinear with the geometry scores in the MS
channels (measured from saved FIMs), vs ~53–71% for the QFIM. The marginal bounds are
built from the residual after projecting out the nuisance score — the efficient
influence — which is second-order structure the 4-mode bank truncates. Known-intensity
optimality (Spec 4) tests the leading scores; marginal optimality tests the residual.
No bug: SLD solve for the intensity direction is exact, `(J⁻¹)_θθ` = Schur to 1e-13.

## Part 1 — compatibility matrix

### The trap (why this is not "just stop taking np.real")
`fisher_information_matrix_for_entire_distribution` computes the **D-form**

```
Tr(L_a G D_b G) = Tr(Λ_a ∂_b ρ),
```

which is identically real for Hermitian `Λ_a`, `∂_b ρ` (up to pinv kernel noise).
Its imaginary part carries **no** compatibility information — the docstring's
"imaginary part is the compatibility matrix" claim is wrong for this form and should
be fixed. The compatibility matrix requires the **LL-form** of
Eq.~(eq:qfi_nonorthogonal):

```
C_ab = Tr(R G L_a G L_b G) = Tr(ρ Λ_a Λ_b),
Re C = QFIM (must reproduce the D-form — free consistency check),
D_ab = Im C_ab = (1/2i) Tr(ρ [Λ_a, Λ_b])   (antisymmetric, the mean Uhlmann curvature).
```

### Code changes — `quantum_fisher_information/quantum_fisher_information.py`
1. New node `sld_covariance_matrix_for_entire_distribution`: batched
   `Tr(R G L_a G L_b G)` over parameter pairs (same flattened-trace trick as the
   existing node; one extra `G L_b G` product). Apply the same coordinate Jacobian:
   both `H` and `D` transform as `B · Bᵀ`.
2. Emit `quantum_fisher_information` (= Re, cross-check against the D-form node to
   ~1e-12) and `compatibility_matrix` (= Im) as separate outputs; driver saves both
   (`qfi_*.npy` as now, plus `compat_*.npy`).
3. Scalar incompatibility per distribution:
   `c = ‖ i H⁻¹ D ‖_∞  ∈ [0, 1]` (largest |eigenvalue|). Sandwich: the Holevo bound
   lies between the SLD-CRB and `(1+c)`·SLD-CRB, so `c ≈ 0` ⇒ marginal SLD bound is
   an honest benchmark; `c = O(1)` ⇒ part of the Fig 2/4 gap is unattainable by any
   measurement.
4. Do this for (a) the full `(δx, δz, r_c)` set and (b) the **effective scores**
   (below) — the marginal-bound attainability question is about
   `Im Tr(ρ L_a^eff L_b^eff)` on the θ block.

## Part 2 — effective SLDs and the adapted POVM

### Effective scores
For nuisance `r` (single knob), the efficient influence operators are

```
L_a^eff = L_a − (H_ar / H_rr) L_r ,   a ∈ {δx, δz},
```

and the marginal QFIM is exactly the Gram matrix of the effective scores,
`H^marg_ab = Re Tr(ρ L_a^eff L_b^eff)` — coefficient-space scalars `H_ar/H_rr` from the
already-computed QFIM, no new solves. (For the general N-intensity case the scalar
becomes `H_aη H_ηη⁻¹` row-times-block; keep the single-knob version first.)

### From SLD coefficients to a mode-space unitary
The SLDs live in the non-orthogonal 4N-basis coefficient representation. To get a
POVM in HG space:

1. **Amplitude matrix `S`**: `S_{m,k} = ⟨φ_m | b_k⟩` for the 4N basis vectors
   `b_k ∈ {ψ_v, ∂ψ_v}` and HG modes `φ_m` up to `mode_degree`. Both the overlaps and
   their derivatives already exist in closed form in the moment machinery
   (`incoherent_mode_sorting` / `hermite_gaussian_basis`) — the ∂ψ columns are the
   derivative amplitudes. Truncation is safe: leakage out of low orders is O(δ⁴).
2. **HG-space operator**: `Λ^HG = S L_coef Sᴴ` (Hermitian, small: M×M with M = number
   of truncated modes).
3. **Eigendecomposition** of `Λ^HG` → orthonormal eigenmodes = columns of a unitary
   `V` on the truncated mode space. This `V` is the "rotation from the SLDs".
4. **Slot into the cross-talk path**: the cross-talk code already rotates the
   measurement basis before computing probabilities. Same insertion point, but note
   one difference: `U(ε)⊗U(ε)` is separable per axis, while `V` mixes x/y orders
   (e.g. `(2,0) ± (0,2)`). That is fine **as long as the rotation acts on the
   per-source amplitude vectors**, not on the polynomial coefficients: with
   `A_{v,m} = ⟨φ_m|ψ_v⟩` already computed, the rotated channel is
   `P_k = Σ_v I_v |(Vᴴ A_v)_k|²` and derivatives chain through `Vᴴ ∂A_v`. Do NOT try
   to push `V` into the separable `g_j(x) f_l(y)` closed form — it won't fit.
5. Keep the remainder channel: `V` unitary on the truncated space + unchanged
   remainder keeps the POVM complete.

### Which operator to rotate by
- **Start per-parameter**: eigenbasis of `L_a^eff` for each of δx, δz separately —
  each should attain its own marginal diagonal (single-parameter reduction of the
  semiparametric bound), modulo the compatibility caveat from Part 1. This is the
  clean validation that the machinery works.
- **Then one fixed POVM**: take the union of the leading eigenmodes of
  `L_δx^eff, L_δz^eff` (rank by |eigenvalue| × mode weight), orthonormalise, feed as a
  single `V`; compute the full 3-parameter FIM + Schur marginal + efficiency. The
  interesting question is how few extra modes close most of the gap.
- The SLDs are point-dependent, so the adapted POVM is too (adaptive-measurement
  flavour). For figures: adapt at one representative sub-Rayleigh working point
  (e.g. δx = δz = 0.3, r_c = 1) and plot the efficiency of that *fixed* POVM across
  the (separation, contrast) neighbourhood — robustness is part of the result.

## Deliverables (figures / numbers)
1. **Incompatibility `c` vs separation and contrast**, full set and effective-score
   set. Decides the benchmark question; also a nice physics statement on its own
   (compatible ⇒ the marginal bound is jointly reachable).
2. **Efficiency of the adapted POVM** (known and marginal) vs the naive 4-mode bank —
   the Spec 5 Fig 2 axes with a third curve. Claim to test: adapted modes recover
   marginal efficiency ≈ 1 sub-Rayleigh (or up to `1/(1+c)` if incompatible).
3. **Mode content of the adapted modes**: bar chart of the dominant `|V|` columns —
   interpretability deliverable, e.g. "the brightness-resolving mode is
   (0,0) − α[(2,0)+(0,2)] + …", the even-mode contrast that separates intensity
   reweighting from displacement².
4. (Diagnostic) `Re` of the LL-form vs the D-form QFIM — regression guard for the new
   node.

## Open questions to pin down before starting
1. Eigenmode truncation: how many eigenmodes of `L^eff` to keep per parameter (start:
   all with |eigenvalue| above, say, 1% of the max within the truncated space).
2. `mode_degree` for the truncated space: 4 (current) vs 5–6 — check the adapted-mode
   content is converged in the truncation.
3. Whether to adapt at the true `r_c` (oracle) or at a nominal `r_c = 1` — the honest
   protocol adapts at the nominal value; quantify the penalty.
4. Multiparameter simultaneity of the per-parameter optimal POVMs — if the two
   effective eigenbases clash, the single-POVM compromise (union-of-modes) is the
   realistic object; Part 1's `c` tells us how much must be lost.

## Effort
Small-to-moderate. Part 1 is one batched matrix-product node + driver plumbing +
docstring fix. Part 2's ingredients all exist (amplitudes, derivative amplitudes,
basis-rotation insertion point from cross-talk, Schur nodes from Spec 5); new work is
the `S`-matrix assembly, an eigendecomposition, and the amplitude-vector rotation path
(replacing the separable-basis assumption at that one point). No new sweeps beyond
re-running the Spec 5 grid with the extra final vars.

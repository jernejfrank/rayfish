All audits are in. Here is the consolidated report.

---
Verdict

The codebase is mathematically solid — every numerical result in the paper is reproducible and correct to machine precision. The most serious finding is in the manuscript itself: the paper's headline closed-form QFIM equation is wrong as printed (the code does not use it, so no figure is affected — but it must be fixed before resubmission). The repo needs a handful of real code fixes and packaging cleanups before the PR, and the unknown-intensities example is mathematically sound and merge-ready.

Verification method: every claim below marked "verified" was checked against independently built ground truth (sympy symbolic integration, brute-force SLD eigendecomposition on a sqrt-Gram embedding, direct 2D quadrature, finite differences) — not against the code's own output. The two central findings were each confirmed by two agents using different methods.

---
✅ What is solid

Core QFIM machinery (src/rayfish/quantum_fisher_information/)
- All 10 closed-form Gaussian overlaps in gaussian_overlaps.py are exact vs sympy ground truth (≤1.6e-15), including all k·z_R terms.
- Package QFIM ≡ brute-force QFIM to ≤6.4e-14 relative error — for equal and unequal intensities, the full 12-parameter matrix including the intensity block, and the Jacobian collapse to (δx, δz). k·z_R phase invariance holds as it must.
- Lyapunov solver exactly encodes 2D = LGR + RGL (residual ≤3.5e-13); the (N+1)-dim subspace reduction and kernel-ambiguity claims in the docstrings are valid; SLD comes out Hermitian.

Classical FI machinery (src/rayfish/classical_fisher_information/)
- Closed-form mode probabilities and all three derivative families match direct 2D quadrature to ~1e-10 (probabilities to 3.4e-13).
- Remainder channel, noise-floor formula τ²∂P∂P/(τP+b), MS FIM assembly with chain-rule Jacobians, and the DI FIM all verified independently. The x-only basis is bit-identical to the 2D basis (j,0) modes.
- The paper's leading-order expansions (P₀₀ ≈ 1−δx²/12−δz²/12, P₁₀ ≈ δx²/12, P₂₀ = P₀₂ ≈ δz²/24) are exactly right, verified symbolically.

Examples used in the paper — all quantitative claims hold
- Main + SM-S1 figures: reproduce pixel-identically from committed data; MS/QFI CRB ratio is 1.008 (δx) and 1.006 (δz) at the smallest separations; DI/MS CRB ratio ~76× (width) and ~149× (depth) sub-Rayleigh; DI depth-FI peaks at 1.15 z_R.
- bucket_vs_mode_cutoff: regenerates all 18 arrays to ≤1e-12; leaked fraction log-log slope 3.86 ≈ 4 in δz (and actually O(δx⁶) in width — the paper's O(δ⁴) is conservative); the 4-mode "Dominant" bank saturates the QCRB sub-Rayleigh (within 0.4% of the order≤5 bank).
- mixing_modes / background_noise: implementation matches Appendix E exactly — rotation chain order, U⊗U (no shortcut; verified the product structure is exact), dB convention, remainder after mixing, noise on all channels including remainder. Committed data reproduces to 1.7e-13; theory_paper PDFs are byte-identical to example outputs. The "−20 dB → 15–25% FI loss" claim is accurate; "interior modes leak ~2×" confirmed numerically.
- ideal_unequal_intensities (Appendix F): weight map I_mid = r_c/(r_c+2) exact; QFIM and MS FIM verified against two independent brute forces; all four figure claims hold (monotone reallocation, efficiency flat within [0.946, 0.991] sub-Rayleigh, H−J PSD at all 52,500 points with zero violations, and the dashed "pure reallocation" line was proven to be the exact zero-overlap limit).
- hello_world: QFI exactly 0.5 constant (the Tsang result in these units); MS saturates to ≥0.9963.

Unknown-intensities example (your specific question): correct and merge-ready.
- The subtle points are handled properly: the intensity derivative D_I has Δ-block support and the code extends the SLD solve accordingly (verified analytically that the Δ-restricted Sylvester solution is exact); the contrast Jacobian row is trace-preserving (dI's sum to zero); nuisance bounds use the Schur complement — i.e. invert-then-take-block, the correct order (confirmed equal to full-inverse-block to 2e-12).
- Full 3×3 QFIM matches brute force to ≤2.4e-13. Zero violations in 105,000 ordering checks each of MS ≥ QCRB and CRB_unknown ≥ CRB_known. All 190 tracked arrays reproduce from current code. One genuinely interesting physics output worth writing down if you ever publish it: the marginal MS efficiency collapses to 0.02–0.32 deep sub-Rayleigh (vs 0.95–0.99 known-intensity) — the 4-mode sorter is a poor intensity estimator.

Tests / CI: 102 pass, 0 warnings, pyright clean, formatting clean, CI green. The QFI core including the new intensity path is at ~100% coverage; the Gramian fixtures trace to genuine independent ground truth (Tsang's analytic two-source result).

---
❌ Problems and proposed fixes

A. Paper — must fix before resubmission

1. Eq. eq:qfim_final (tex 201–205) is not the QFIM. Confirmed independently by two agents. Two defects: (i) the derivation silently drops the two G_τ traces between tex 496–499 and 500–505 — the missing term 4 δ_{u_a,u_b} Re(G_τ)_{a,b} is the dominant contribution (in the orthogonal-source limit the printed formula gives H=0; for V=1 it even goes negative); (ii) the overall 1/V (generally I_v) prefactor is inconsistent. Corrected, machine-precision-verified form:
H_ab = (1/V)[ 4 δ_{u_a,u_b} Re(G_τ)_{a,b} + 2 Re⟨γᵇ|G_γ† L̃ᵃ_Δ G_Δ|γᵇ⟩ + 4 Re⟨γᵃ|G_γ†|γᵇ⟩⟨γᵇ|G_γ† G_Δ⁻¹|γᵃ⟩ ]
(with 1/V → I_v appropriately for unequal intensities). The code computes the exact expression, so no figure changes.
2. Factor-V inconsistency in Eq. eq:L_delta_sylvester (tex 207–210): written with D (entries 1/V) but only correct with the appendix's D̃ = V·D (tex 465). As printed, L_Δ is a factor V too small. Fix by stating both display equations in one normalization.
3. Appendix F claim (tex 702) that the closed form "carries over with 1/V → I_v" doesn't hold literally: diag(I) no longer cancels, and the reduced equation becomes a weighted Sylvester equation (verified form available). Conclusions unaffected — the code solves the full SLD equation directly.
4. Compile is fatally broken: unescaped & in ref.bib entry bearne2026ModeSorting ("Optics & Laser Technology"). One-character fix; after it, zero undefined refs/citations, all 18 figures present.
5. Appendix A mislabels (tex 410–413): the three single-derivative overlap rows are labelled ⟨∂ψ_v|ψ_u⟩ but evaluate ⟨ψ_v|∂ψ_u⟩ (exact negatives). Relabel to match the Grammian definition at tex 398. Also: ψ₀ parenthesization typo (tex 404); the factor (−1+iδz_u)³/(i+δz_u)³ at tex 416/419 is identically −i (unsimplified CAS export — simplify); ψ_i → ψ_v at tex 419. All double-derivative formulas including the big τ_pp are exact as printed.
6. Response-letter promises not in the manuscript — a referee will check these:
  - Surface roughness "flagged as future work in the Conclusions" (claimed twice) — the word "roughness" appears nowhere in the tex.
  - Continuous emitter distributions (Ref. 2 Q5) — absent.
  - The three-part novelty statement (Ref. 1 P2): the Chernoff/hypothesis-testing contribution (a) is missing from the red Intro sentence (tex 100).
  - Response says results hold for "known, unknown, or fluctuating" intensities; Appendix F explicitly assumes known brightness and the unknown-intensity analysis is not in the paper. Either soften the response wording or add a sentence (you have the verified unknown_intensities results if you want to cite "not shown" or include them).
  - Appendix F conclusion (tex 747) claims MS "retains its detection advantage" under unequal brightness — no detection analysis under unequal brightness exists anywhere in the paper. Reword to estimation-only or add a line of justification.
7. SM vs End Matter identity crisis: the same material is simultaneously \section*{SUPPLEMENTAL MATERIAL}, "Appendix~\ref{}", and \cite{supposta} ("See Supplemental Material…"), including a figure referenced through a citation into the same document (tex 247). Pick one: End Matter → delete supposta and retitle; separate SM → all cross-boundary \refs break and appendix equations need S-numbering.
8. Smaller items: Appendix C typos — missing i in the exponentials at tex 582–583 (code has it right), σ vs σ² notation inconsistency (tex 557 vs 568–585, code uses variance, correct); block dimensions swapped at tex 195 (X_γ is V×3V); \pdv missing second argument at tex 516; garbled "As these modes are complete" at tex 607; stray \ at tex 226; commented-out obsolete conclusion paragraph (tex 368) and duplicate section line (tex 639) to delete. One descoped-but-found Chernoff note, for George: the "min at s=0" step at tex 338 is degenerate once the remainder channel is included (the sum is exactly 1 at s=0); the δz²/12 scaling survives but the minimization wording needs care.
9. Wording nuances the data mildly contradicts: "MS remains above DI throughout the sub-Rayleigh regime" under −20 dB cross-talk fails for depth in the 0.85 ≲ δz ≲ 1 band (85% of points pass; defensible under the existing Rayleigh caveat, but be aware); in Appendix F, "departures smallest in the sub-Rayleigh regime" holds for QFI but reverses for MS depth (−21% at the most sub-Rayleigh sweep).

B. Code — fix before the PR

1. incoherent_di.py is incompatible with the current parameter catalogue (crashes with a matmul shape error): it builds 3N derivative rows while the catalogue Jacobians are now 4N columns after the intensity refactor (incoherent_di.py:100-151, 233 vs distance_3_sources_xz.py:88-91). This is why the paper's DI curves currently rely on legacy committed fi_di.npy that cannot be regenerated at HEAD. Fix: add the N intensity-derivative rows (the unweighted single-source probabilities) mirroring incoherent_mode_sorting.py:137, and add an e2e test wiring DI + catalogue (the existing test hand-rolls a 9-column Jacobian, masking the bug).
2. examples/3_point_sources/outdated_ideal_incl_qfi_di/ is the sole generator of the paper's main figures and is broken at HEAD (imports the deleted incoherent_sources module). Do not delete it in the cleanup — fix the imports (a verified working patch exists: swap to quantum_fisher_information.{common,operator_matrices,lyapunov_operator,...} as hello_world does) and rename the directory (e.g. ideal_main_figures). After patching, it reproduces the committed QFI to 1.5e-13 and regenerates all six main/SM figures pixel-identically. check_qfi.ipynb was a one-off migration check and can then be deleted.
3. Complex Schur complement in the nuisance covariance (quantum_fisher_information.py:175-209): the Schur complement is computed on the complex FIM and silently cast to real (the ComplexWarning you may have seen). Harmless at the tested geometry (Im ≈ 1e-14) but wrong in general — take np.real of the full FIM first.ideal_main_figures). After patching, it reproduces the committed QFI to 1.5e-13 and regenerates all six main/SM figures pixel-identically. check_qfi.ipynb was a one-off migration check and can then be deleted.
3. Complex Schur complement in the nuisance covariance (quantum_fisher_information.py:175-209): the Schur complement is computed on the complex FIM and silently cast to real (the ComplexWarning you may have seen). Harmless at the tested geometry (Im ≈ 1e-14) but wrong in general — take np.real of the full FIM first.
4. Missing np.conj on mode coefficients in the three derivative closed forms (gaussian_overlaps.py:730, 776-786, 833-834) — the probability path conjugates correctly, the derivative paths don't. Errors up to 645% for complex-coefficient bases; all shipped HG bases are real so nothing published is affected. One-line fixes.
5. Off-by-one in coordinate_jacobian__with_intensity (distance_3_sources_xz.py:119-123): vacuously-true bounds check → IndexError when the target source is last. Not triggered by current examples.
6. Smaller: dark-mode mask P > 1e-16 (incoherent_mode_sorting.py:268) discontinuously discards the finite dP²/P limit within ~3.5e-8 of the origin — clamp instead of drop; _select_mode_set ignores the caller's mode ordering (hermite_gaussian_basis.py:87-92) — the mixing and noise examples end up with different channel orderings (both internally consistent, but it's a silent-misread trap; sort by the requested order); CWD-relative np.save("coordinates.npy", ...) in mixing_modes/simulate_bounds.py:101-104; docstring errors — gamma_* conjugation-side claims (gaussian_overlaps.py:117-197, mirror of paper item A5), stale "δx := (δx₃−δx₂)/2" (distance_3_sources_xz.py:82,105 — extra /2, also makes the notebooks' sqrt(2)·x_range axis labels a factor-2 trap), stale "1D only" claim (incoherent_mode_sorting.py:59), PSF sign in quantum_fisher_information.py:15-17.

C. Repo hygiene — must-fix before the PR

1. Clean pip install is broken: rayfish/__init__.py reads a VERSION file the wheel doesn't ship (confirmed by building the wheel — import rayfish raises FileNotFoundError). Use importlib.metadata.version() or add package-data. This also removes the Path/f namespace leaks.
2. Version chaos: pyproject 0.2.0, VERSION file 0.4.0, CHANGELOG 0.4.0 — and the CHANGELOG 0.3.0/0.4.0 entries describe features that don't exist in this repo (streamlit dashboard, sqlite, pydantic app — apparently pasted from another project). Align all three and rewrite the CHANGELOG to cover the actual new work (remainder channel, noise floor, cross-talk, unequal/nuisance intensities).
3. ~73 MB of binaries in the diff (353 .npy + 89 pdf/eps). Drop at minimum: theory_paper/{old_old_figures, old_figure, figures_v0, misc_old} and the duplicate main_paper_V0.tex; consider gitignoring examples/**/*.npy (everything regenerates from the scripts, now verified) or LFS.
4. [tool.ruff] fix = true silently mutates files on plain ruff check and makes CI lint unfailable — remove it, fix the 3 real B905 findings, drop the conflicting ISC001 rule.
5. Zero test coverage on incoherent_mode_sorting.py (the headline new module — 0/113 statements) and the covariance/Schur functions. The external audit has now effectively tested them, but at least one regression test pinning the 2-source MS FI against Tsang's analytic result would protect the remainder-index gymnastics at lines 259–322. Also 0% on the parameter catalogues (the old test_jacobians.py was deleted this branch without replacement — worth restoring).
6. Tracked .DS_Store files (5); src/visualize_helpers.py ships as a top-level module in the wheel and imports matplotlib (a dev-only dep) — move into the package behind a [viz] extra or into examples/; notebook in runtime deps; pandas declared but never imported in src; README badge says Python 3.9–3.12 (requires 3.13) and references a nonexistent Makefile; dead docs links in pyproject.

---
Suggested order of attack: paper items A1–A4 and the response-letter gaps (A6) before resubmission; code items B1–B2 first since the PR's cleanup would otherwise delete the only main-figure generator; then C1–C4 to make the PR installable and honest, with C5's regression tests as the safety net. Every fix above has a concrete verified patch direction, and the verification scripts (brute-force QFIM/FIM, quadrature checks, corrected-formula demos) are preserved in the session scratchpad if you want to reuse them as test fixtures. Happy to start applying any subset of these.

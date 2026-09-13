# C++/CUDA rewrite of rayfish — assessment and plan

## Verdict
Yes, this is a well-chosen learning project — with one reframe. As pure software
engineering the rewrite has weak ROI (the Python is ~4k lines, tested, BLAS-backed,
and works). As a *learning vehicle with a real scientific payoff attached* it is
close to ideal:

- You know the problem inside out and have golden reference outputs (`.npy` archives
  and Mathematica cross-checks) — the hardest part of any rewrite, a trustworthy
  oracle, already exists.
- The workload is embarrassingly parallel at two levels (geometries × contrasts) with
  small, dense, closed-form kernels — the canonical shape for learning CUDA.
- There is a genuine need: Spec 5/6 want dense log-spaced sub-Rayleigh sweeps (the
  inflation-figure choppiness) and adapted-POVM robustness maps. At Python speeds
  (~2 h for 21 contrasts × 50×50) those figures are painful; a native version makes
  them interactive. The rewrite has a deliverable beyond pedagogy.

C++ over Go/Rust is the right call for *this* project: CUDA is C++-first, the
numerics ecosystem (Eigen, cuBLAS/cuSOLVER batched APIs, Thrust) is mature, and
complex arithmetic + operator overloading matter for readability of the math. Go's
GC and lack of operator overloading fight the domain; Rust's CUDA story is still
rough. Save those for the side-projects that fit them.

## What the computation actually is (inventory before porting)

Per task = one (geometry grid point, contrast) pair; ~52.5k tasks per sweep today:

**Quantum path**
1. Closed-form Gaussian overlaps → Gram matrix `G` (4N×4N complex, N=3 → 12×12).
2. Sparse derivative matrices `D_a`: γ-block entries (coordinates) + Δ-diagonal
   (intensity); P = 3N+1 directions after the contrast Jacobian.
3. SLD solve per direction: Kronecker-vectorised Sylvester system, currently
   min-norm `pinv` of (N+1)²×(N+1)² = 16×16 (coordinate dirs) and N²×N² = 9×9
   (intensity dir) complex matrices.
4. QFIM assembly `Re Tr(L G D G)` (+ LL-form for compatibility, Spec 6), Jacobian
   sandwich `B H Bᵀ`, block inverse / Schur complement of 3×3.

**Classical path**
1. Recursive Gaussian moments `E[x^ζ]` per source, mode amplitudes
   `⟨φ_jl|ψ_v⟩` and their closed-form derivatives (small polynomial recursions,
   complex).
2. Channel probabilities + remainder, scores (coordinate + intensity), FIM over
   channels, same Jacobian/Schur post-processing.
3. (Spec 6) amplitude-vector rotation by a mode-space unitary before |·|².

Every task is a few thousand flops on matrices ≤ 16×16 — the **"many tiny problems"**
regime. No single kernel is interesting; throughput comes from batching. That is a
feature for learning: the design problem is data layout and batching strategy, not
one clever kernel.

## Architecture guidance

- **Do not port Hamilton.** The DAG framework earns its keep in Python (config
  switching, notebook introspection); in C++ it dissolves into plain functions
  composed in `main`/driver code. Keep the *module boundaries* it gave you —
  `parameter_catalogue` / `overlaps` / `qfi` / `cfi` — as namespaces or libraries;
  drop the framework.
- **Config switching (`config.when`) → runtime strategy or template parameter.** Start with
  runtime `if`; templatize only if a profile says so. Resist the C++ urge to
  metaprogram on day one.
- **Structure-of-arrays from the start.** One big batch of tasks with contiguous
  per-field arrays (all Gram matrices, all D-matrices, …), not a vector of task
  objects. This is the single decision that makes the later CUDA port mechanical.
- **Keep `.npy` as the interchange format** (`cnpy` or `libnpy`). Notebooks, plotting
  (stays Python/matplotlib — do not rewrite), and regression baselines all keep
  working. The C++ binary is a drop-in producer of the same files the notebooks read.
- Libraries: Eigen (CPU linear algebra, header-only), Catch2 or GTest, CMake +
  presets. CUDA phase: cuBLAS batched GEMM, cuSOLVER batched SVD *or* a hand-rolled
  solve (below), Thrust for reductions. `std::complex<double>` ↔ `cuDoubleComplex`
  at the boundary.

## Numerical landmines (know these before writing a line)

1. **FP64 on consumer GPUs is crippled** — 1/32 to 1/64 of FP32 throughput on
   GeForce cards. This problem *needs* double: you mask on condition numbers up to
   1e12–1e16 and take Schur complements of nearly-degenerate matrices (the whole
   Spec 5 story lives in the last few digits). Consequences: (a) a well-parallelised
   CPU version with OpenMP may beat a consumer-GPU FP64 version — do CPU first and
   measure; (b) the GPU win is still real vs Python, just not the headline FP32
   TFLOPs; (c) if a data-center card (A100/H100) is reachable, FP64 is a non-issue.
2. **`pinv` min-norm semantics.** The SLD kernel component is non-unique (the QFIM is
   invariant, individual `L` entries are not). Element-wise comparison of `L` against
   Python will fail; compare QFIMs and bounds, not intermediates — or better, replace
   the 16×16 pinv with the structured reduced solve (the (N+1)-subsystem is
   non-singular for coordinate directions once reduced properly; the Δ-block solve for
   intensity directions is plainly invertible). Batched SVD on GPU is painful; a
   structured direct solve is both faster and a better learning exercise.
3. **Condition-number masks and `inf` sentinels** must reproduce Python's behaviour
   exactly (`cond < 1e16` vs `1e12` differs between nodes — match per node) or the
   figures change at the singular edges.
4. **Complex-Hermitian discipline**: enforce Hermiticity after assembly the way numpy
   implicitly does via `np.real` at fixed places; drift here shows up as spurious
   imaginary parts exactly where Spec 6 goes looking for the compatibility matrix.

## Validation strategy (the part that makes this safe)

- **Golden regression**: the existing `.npy` sweeps are frozen fixtures. A pytest
  harness runs the C++ binary, loads both outputs, asserts `allclose` with per-array
  tolerances (bounds: 1e-10 relative; masked/singular entries: same mask).
- **Property tests** (independent of Python): Schur marginal ≡ θθ-block of full
  inverse; MS CRB ≥ QFI CRB; marginal ≥ known; FIM symmetry/PSD; known-intensity
  efficiency ≈ 1 sub-Rayleigh (Spec 4's result as a test).
- **Freeze the Python implementation during paper revisions.** Two live
  implementations during a referee round is how errors get published. The C++ tree
  is a sibling (`rayfish-native/` or a `native/` subdir), Python stays the reference
  until the native version has survived a full figure-regeneration cycle.

## Staged plan (each stage ships something)

0. **Skeleton** — CMake, Catch2, cnpy round-trip, `coordinates` grid generation.
   Diff coordinates.npy against Python. *(Learn: modern CMake, project hygiene.)*
1. **Gaussian overlaps + Gram** — closed forms from `app:gaussian_overlaps`.
   *(Learn: `std::complex`, Eigen, unit testing against Mathematica values.)*
2. **Classical FI path** — moments recursion → amplitudes → probabilities → FIM →
   Jacobian/Schur. First full figure parity (fi_ms files). *(Learn: numerics
   organisation, SoA layout.)*
3. **Quantum path** — D-matrices, structured Sylvester solve (replacing pinv), QFIM
   both forms (D-form + LL-form ⇒ Spec 6 Part 1 lands natively for free),
   compatibility matrix. *(Learn: dense linear algebra beyond library calls.)*
4. **CPU parallelism** — OpenMP over the task batch; measure scaling; likely
   100–500× over the current Python sweep already. Regenerate Spec 5 figures on a
   dense log-spaced grid — first scientific payoff.
5. **CUDA port** — kernel-per-stage over the SoA batch; batched cuBLAS where natural,
   hand-rolled 12×12/16×16 device routines where not (one warp or block per task;
   these matrices fit in shared memory/registers comfortably). *(Learn: occupancy,
   memory coalescing, the batched-small-matrix regime, profiling with Nsight.)*
6. **Perf study + writeup** — CPU-parallel vs GPU, FP64 reality check, roofline.
   This is the "learning parallel computing" deliverable; the numbers are honest
   because both versions produce identical physics.

Suggested order rationale: classical before quantum (fewer numerical landmines,
faster gratification), CPU-parallel before CUDA (isolates "is it parallel correctly"
from "is it GPU correctly"), and the Spec 6 compatibility node lands as part of
stage 3 rather than being ported later.

## Risks

- **Scope creep**: plotting, notebooks, Hamilton-style config all stay in Python.
  The C++ surface is exactly "read inputs → write `.npy`".
- **Drift**: any physics change during referee rounds goes into Python first
  (reference), C++ second, with the regression harness as the gate.
- **Motivation cliff at stage 3**: the quantum path is where C++ friction (complex
  Eigen expressions, ownership of batched buffers) peaks. Stages 0–2 are deliberately
  front-loaded with quick wins.
- **CUDA debugging** is a genuinely worse experience than anything in Python;
  budget for it, keep the CPU path as the differential-testing twin of every kernel.

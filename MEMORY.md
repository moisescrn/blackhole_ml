# MEMORY.md — Black Hole Machine Learning

Project memory across sessions. Maximum 50-70 lines.

## Current State
- Spec `001-ForwardModel` is approved and fully IMPLEMENTED (8/8 tasks): Step 2 forward model, parameters → shadow.
- `src/knn_shadow/metric.py`: `delta`, `deltaPrime`, `sigma`, `maxSpin`, `outerHorizon`, `validateParameters`, `validateParametersBatch`; constants `R_O_FACTOR = 5.0`, `THETA_0 = π/2`.
- `src/knn_shadow/shadow.py`: `sinTheta`, `sinPsi`, `photonResidual`, `solvePhotonRadii`, `solvePhotonRadiiBatch`, `stereographicProjection`, `shadowBoundary` → single `(2N_r−2, 5)`, `shadowBoundaryBatch` → `(B, 2N_r−2, 5)`.
- 104 tests green (`test_metric.py`, `test_shadow.py`, `test_benchmark.py`); `ruff check` and `ruff format --check` clean; no new dependencies.

## Decisions (and Why)
- `m` is a free parameter; output `(x, y)` scale linearly with `m` (all lengths in units of `m`, `r_O = 5m`).
- Sampling: only "uniform in `r_p`" (dropped "uniform in sky angle ψ"). Default `N_r = 128`, endpoints included; open sequence = `2·N_r − 2` points (endpoints shared between mirror halves).
- ψ branch rule (locked): upper `ψ = arcsin(sin ψ)` (`cos ψ > 0`), lower `ψ = π − arcsin(sin ψ)` (`cos ψ < 0`), joined at `r_{p±}`, negative ψ wrapped `+2π` into `[0, 2π)`. Canonical order: `r_{p+}` → ascend upper half → `r_{p-}` → descend lower half (no repeated closing point). Both endpoint ψ are hardcoded exact (`π/2`, `3π/2`), since `arcsin(clip(±1))` can be ~1e-6 off for near-static spin.
- Batch root solve raises `RuntimeError` naming the offending rows when the 8-step bracket expansion leaves `f(hi) ≥ 0` (mirrors the scalar path).
- Schwarzschild anchor is finite-distance, NOT `3√3 m`: at `r_O=5m` the circle has `sin θ = 12√45/100 ≈ 0.804984` (plane radius `2 tan(θ/2) ≈ 1.0105 m`). `3√3 m` is the infinite-distance impact parameter.
- Error convention: invalid params → `ValueError` (FR-2); root-solve failure for valid params → `RuntimeError` (FR-4).
- Tolerance split (NFR-1): `r_{p±}` verified via residual + bracketing/monotonicity; angles/projection rel. `1e-9` (absolute near `|sin ψ|=1`). Clamp `sin θ`/`sin ψ` to `[−1,1]` before `arcsin`.
- Root solve: single set via scalar `scipy.optimize.brentq` (robust, tight tolerance); batch via vectorized fixed-count (60) bisection, because `brentq` is scalar-only and forbids per-set loops.
- No numba added (NFR-6 gate): boundary eval is ≥50% of runtime, but the JIT-able part yields only ~1.3–1.6× overall, below the 2× threshold.
- `ScalarOrArray` alias in `shadow.py` lets `sinTheta`/`sinPsi`/`photonResidual` broadcast over `(B,1)` parameter columns instead of duplicating batch formulas.
- Benchmark baseline: ~0.27 ms/call (`N_r=128`), ~4.9 ms/batch (`B=100`).

## Lessons Learned and Mistakes to Avoid
- Do NOT read `PROJECT_DESCRIPTION.md` unless explicitly told (AGENTS.md forbids it; contents outdated/wrong). Authoritative physics = `docs/physical_framework.md`.
- `3√3 m` is a different quantity (infinite-distance impact parameter); don't use it as the forward-model anchor.
- Don't confuse spacetime `ϑ` with celestial-sphere `θ`.
- For this analytic/vectorized model numba is NOT worth it: the bulk is NumPy ufunc work on large arrays (not JIT-able), and `brentq` callbacks cannot be JIT-compiled.
- The NUT parameter `l` triggers ruff E741; keep the physics name and annotate `# noqa: E741`.
- Scalar helpers casting with `float(...)` blocked broadcasting; return arrays / use `np.asarray` in the inner path.

## Next Steps
- Step 3 validation (limit tests + geodesic cross-check) or Step 4 data generation.
- Update `src/knn_shadow/__init__.py` public exports if desired.

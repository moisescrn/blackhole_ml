# Tasks 001 — Forward Model

Dependency order. Tests-first where stated (write the failing test, implement,
run `uv run pytest`, then green). Each task ≈ 20–30 min. 8 tasks; no split
needed.

- [x] **T1. Metric helpers and constants in `metric.py` + tests.** FR-3
  - Done when: `delta`, `deltaPrime`, `sigma`, `maxSpin`, `outerHorizon` are
    implemented with type hints and `tests/test_metric.py` has green tests that
    check them against hand-computed values (Schwarzschild `m=1,Q=0,a=0.001,l=0`),
    `outerHorizon == m + √(a_max² − a²)`, and the constants `r_O = 5·m`,
    `ϑ_O = π/2` are defined and correct.

- [x] **T2. `validateParameters` / `validateParametersBatch` + tests.** FR-1, FR-2
  - Done when: `validateParameters` raises `ValueError` for `m ≤ 0`, `a_max² ≤ 0`,
    `a² > a_max²`, `a == 0`, `|a| == a_max`, and `Δ(r_O) ≤ 0`, passes a known-valid
    set, and `validateParametersBatch` (vectorized over `(B,4)`) raises with the
    offending index when any set is invalid — all covered by green tests in
    `tests/test_metric.py`.

- [x] **T3. `sinTheta` / `sinPsi` formulas + tests.** FR-6
  - Done when: `sinTheta`/`sinPsi` in `shadow.py` compute the raw FR-6 formulas
    (equatorial `Σ = r²+l²`) and `tests/test_shadow.py` verifies them against the
    analytic formulas to 1e-9 on sample radii, including the Schwarzschild anchor
    value `sin θ ≈ 0.804984`, and that clamping to `[−1,1]` (applied at the
    `arcsin` call site) handles a `sin ψ = 1 + 4.4e-16` endpoint.

- [x] **T4. Root solve `solvePhotonRadii` (bracketing + `brentq`) + tests.** FR-4
  - Done when: `photonResidual(r, …, sign)` and `solvePhotonRadii` are implemented
    (bracket `[r_+ + δ, 4·(a_max+|l|+m)]` with doubling fallback; `brentq` per
    sign) and `tests/test_shadow.py` verifies, for several valid sets (`a>0`,
    `a<0`, large `|l|`): normalized residual ≤ 1e-12 at each root, `f(lo) > 0`,
    `f(hi) < 0`, monotonicity on the bracket, `sin ψ(r_{p+}) ≈ +1` and
    `sin ψ(r_{p-}) ≈ −1`, and that a forced root failure raises `RuntimeError`.

- [x] **T5. `shadowBoundary`: sampling, ψ reconstruction, projection, ordering + tests.** FR-5, FR-7, FR-8, FR-9, NFR-3
  - Done when: `shadowBoundary(m, Q, a, l, nR=128)` returns a `(2·nR−2, 5)` array
    `(x,y,θ,ψ,r_p)` in the canonical FR-9 order (first `ψ = π/2`, index `nR−1`
    `ψ = 3π/2`, interiors doubled with the two branch ψ values), raises `ValueError`
    for `nR < 3`, and `tests/test_shadow.py` confirms: shape/order, `θ = arcsin(sin θ)`,
    projection `(x,y) = −2·tan(θ/2)(sin ψ, cos ψ)`, the Schwarzschild circle anchor
    (`sin θ ≈ 0.804984`, plane radius ≈ 1.0105, both ±1e-3), and bit-identical
    output across two identical calls (determinism).

- [x] **T6. `shadowBoundaryBatch` with vectorized bisection + tests.** FR-10, NFR-5
  - Done when: `solvePhotonRadiiBatch` (vectorized bisection, fixed 60 steps) and
    `shadowBoundaryBatch` are implemented with no per-set Python loop, and
    `tests/test_shadow.py` verifies the `(B, 2·nR−2, 5)` shape, the `(m,Q,a,l)`
    column order, and that `batch[i]` equals `shadowBoundary(arrParams[i])` within
    `atol=1e-6` for a mixed valid batch (incl. `a>0` and `a<0`).

- [x] **T7. Benchmark report + profiling gate.** NFR-6
  - Done when: `tests/test_benchmark.py` measures and prints the median wall-clock
    time per single call (`N_r=128`) and per batch (`B=100`) over `R=10`
    repetitions, and a manual `python -m cProfile` run records whether the
    boundary evaluation is ≥50% of runtime. If (and only if) it is AND a numba
    version shows ≥2× speedup, a follow-up task is proposed; otherwise no JIT is
    added.
  - Result: median 0.27 ms/call (single, `N_r=128`) and 4.9 ms/batch
    (`B=100`, `N_r=128`); boundary evaluation is ~97% of runtime, but >45% of it
    is large-array NumPy ufunc work that JIT cannot accelerate, so the ≥2×
    criterion is not plausibly met → **no numba/JIT added**.

- [x] **T8. Full verification, lint, cleanup, memory.** NFR-2, NFR-3
  - Done when: `uv run pytest` passes all tests, `uv run check .` is clean (ruff,
    line-length 85, double quotes), no new dependency was added, the placeholder
    `tests/test_sample.py` is removed, and `MEMORY.md` is updated with the current
    state and any decisions/lessons.
  - Result: 94 tests pass; `ruff check .` and `ruff format --check .` are clean
    (there is no `check` console script, so the two explicit ruff commands are
    used); `pyproject.toml` unchanged; `tests/test_sample.py` removed;
    `MEMORY.md` updated.

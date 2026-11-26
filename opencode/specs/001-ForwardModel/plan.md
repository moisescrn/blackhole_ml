# Plan 001 — Forward Model

Status: implemented

Covers FR-1 … FR-10 and NFR-1 … NFR-6 of `spec.md`. Only the WHAT/HOW of the
forward model; noise, dataset generation, inference and the Step-3 validation
suite are out of scope.

Conventions that apply throughout (from `AGENTS.md` + `pyproject.toml`):
`camelCase` functions, `PascalCase` classes, Hungarian notation for variables,
type hints on every argument and `->` return, ruff with line-length 85 and
double quotes, short precise comments with no spec/FR/task references, English
only. First implementation is pure NumPy/vectorized; numba is a *conditional*
later step (NFR-6), so no `@jit`/`@njit` decorator is applied now even where
`AGENTS.md`'s "≥3 uses" rule would suggest it.

## 1. Files and responsibilities

Mapping to the EXISTING modules (no new source file is required):

- `src/knn_shadow/metric.py` — parameters, constants and validity.
  - `delta` (`Δ`), `deltaPrime` (`Δ'`), `sigma` (`Σ`), `maxSpin` (`a_max`),
    `outerHorizon` (`r_+`). Supports FR-3, FR-4, FR-6 and the FR-2 checks.
  - Constants: observer radius factor `5.0` (so `r_O = 5·m`) and
    `ϑ_O = π/2` (equatorial observer) — FR-3.
  - `validateParameters` / `validateParametersBatch`: FR-1 (`m ≤ 0`) and FR-2
    (spin validity `a² ≤ a_max²`, `a = 0`, `|a| = a_max`, `Δ(r_O) ≤ 0`),
    raising `ValueError`. Covers FR-1, FR-2.
- `src/knn_shadow/shadow.py` — the forward model.
  - `sinTheta`, `sinPsi` (FR-6), `photonResidual` (FR-4), root solving
    `solvePhotonRadii` (FR-4, single set, `brentq`) and
    `solvePhotonRadiiBatch` (FR-4/FR-10, vectorized bisection),
    `stereographicProjection` (FR-8), sampling + ψ-reconstruction + canonical
    ordering inside `shadowBoundary` (FR-5, FR-7, FR-9) and `shadowBoundaryBatch`
    (FR-10).
- `src/knn_shadow/geodesics.py`, `noise.py`, `data_generation.py` — NOT touched
  (out of scope: validation integrator, noise, dataset).

NEW files proposed (test files only, allowed by NFR-2 — no source/script file,
so no approval needed for the source tree):

- `tests/test_metric.py` — FR-1, FR-2, FR-3 (validation + metric helpers).
- `tests/test_shadow.py` — FR-4 … FR-10, NFR-1a/1b, NFR-3, NFR-5.
- `tests/test_benchmark.py` — NFR-6 (median timings; profiling stays a manual
  `python -m cProfile` step, no committed script).
- `tests/test_sample.py` — placeholder sample (`reverse_text`), unrelated to the
  spec; recommend deleting it (test file, no approval required).

## 2. Pure functions

All arrays are `numpy.typing.NDArray[np.float64]`; scalars are `float`. Every
function below is **pure** (no global state, no I/O, deterministic — NFR-3)
except the validators and solvers, which only *raise* on bad input.

`metric.py`
```python
def delta(r, m: float, Q: float, a: float, l: float) -> NDArray       # Δ = r²−2mr+a²−l²+Q²      [pure, numba-candidate]
def deltaPrime(r, m: float) -> NDArray                                # Δ' = 2r−2m              [pure, numba-candidate]
def sigma(r, cosTheta: float, a: float, l: float) -> NDArray          # Σ = r²+(l+a·cosθ)²      [pure]
def maxSpin(m: float, Q: float, l: float) -> float                    # a_max = √(m²+l²−Q²)     [pure]
def outerHorizon(m: float, Q: float, a: float, l: float) -> float     # r_+ = m+√(a_max²−a²)    [pure]
def validateParameters(m: float, Q: float, a: float, l: float) -> None            # raises ValueError (FR-1/2)
def validateParametersBatch(arrParams: NDArray) -> None                           # vectorized, raises ValueError
```
`shadow.py`
```python
def sinTheta(r, m, Q, a, l) -> NDArray     # FR-6 sin θ          [pure, numba-candidate]
def sinPsi(r, m, Q, a, l) -> NDArray       # FR-6 sin ψ          [pure, numba-candidate]
def photonResidual(r, m, Q, a, l, sign: float) -> NDArray  # FR-4 f_s(r) [pure, numba-candidate]
def solvePhotonRadii(m, Q, a, l) -> tuple[float, float]          # (rPPlus, rPMinus); RuntimeError on failure
def solvePhotonRadiiBatch(arrParams) -> tuple[NDArray, NDArray]  # (B,) each; vectorized bisection
def stereographicProjection(th, psi) -> tuple[NDArray, NDArray]  # (x, y)   [pure]
def shadowBoundary(m, Q, a, l, nR: int = 128) -> NDArray         # (2·nR−2, 5)
def shadowBoundaryBatch(arrParams, nR: int = 128) -> NDArray     # (B, 2·nR−2, 5)
```

`r` (or `arrParams`) is always passed as an array in the inner path; scalar
callers use `np.asarray`. `sinTheta`/`sinPsi` return the RAW values; clamping to
`[-1, 1]` (FR-6) is applied where `arcsin` is taken, not inside them, so the raw
residual stays inspectable in tests.

**Numba note (NFR-6):** `delta`, `deltaPrime`, `sinTheta`, `sinPsi`,
`photonResidual` are the only functions evaluated ≥3 times in the inner path and
are the natural `@njit` candidates. They are intentionally left pure NumPy for
this plan; JIT is added only after the benchmark shows the boundary evaluation
is ≥50% of runtime AND a measured ≥2× speedup (see §5 and task T7).

## 3. Algorithm in pseudocode

Notation: `ϑ_O = π/2`, `sinϑ_O = 1`, `cosϑ_O = 0`, so `Σ(r,ϑ_O) = r² + l²`
and `√(Δ(r))` is real and positive on the admissible interval (`r > r_+`).
Equatorial coincidence used in FR-6: `(l + a·cos²ϑ_O)² = l² = (l + a·cosϑ_O)²`,
so the `sin ψ` numerator equals `Σ(r,ϑ_O)·Δ'(r) − 4r·Δ(r)`. **This must be
re-derived from the thesis before any future free-inclination work.**

### 3.1 Residual and root isolation (FR-4)
```
photonResidual(r, m, Q, a, l, sign):            # sign = −1 → r_{p+}, +1 → r_{p−}
    return sigma(r, cosTheta0, a, l)*deltaPrime(r, m) − 4*r*delta(r, m, Q, a, l)
           + sign * 4*a*r*sqrt(delta(r, m, Q, a, l)) * sinTheta0

rPlus = outerHorizon(...)                        # largest real root of Δ(r)=0
lo = rPlus + δ                                   # δ = 1e-9 · max(1, rPlus); f(lo) > 0
hi = 4 * (maxSpin(...) + |l| + m)                # parameter-scaled; f(hi) < 0
while f(hi) >= 0: hi *= 2                        # safety doubling; cap → RuntimeError
```
`f(lo) > 0` because at `Δ → 0` the `±4ar√Δ` term vanishes and `Σ·Δ' > 0`;
`f(hi) < 0` because the leading term at large `r` is `−2r³`. With `sign = −1`
(`r_{p+}`) and `sign = +1` (`r_{p-}`) the two equations are solved independently
and labeled by the equation, not by magnitude (§5, decision D2).

### 3.2 Single-set root solve (FR-4) — `brentq`
```
for sign in (−1, +1):
    lo, hi = bracket(sign)                       # as in 3.1
    root = scipy.optimize.brentq(lambda r: photonResidual(r, ..., sign), lo, hi)
    if not finite/positive/Δ≥0: raise RuntimeError(f"root failed for (m,Q,a,l)")
rPPlus, rPMinus = root(sign=−1), root(sign=+1)
```

### 3.3 Batch root solve (FR-4 + FR-10) — vectorized bisection
Brentq is scalar-only, so the batch uses a single vectorized bisection over all
`B × 2` brackets; the loop is over *iterations*, never over parameter sets
(NFR-5). Fixed iteration counts keep it deterministic (NFR-3).
```
rPlus = outerHorizon(arrParams)                  # (B,)
lo = rPlus + δ                                   # (B,)
hi = 4*(maxSpin + |l| + m)                       # (B,)
for _ in range(8):                               # expand until f(hi)<0 for every set
    fHi = photonResidual(hi, sign=±, batched)    # (B,)
    if (fHi < 0).all(): break
    hi = np.where(fHi >= 0, 2*hi, hi)
for _ in range(60):                              # bisection, fixed count
    mid = 0.5*(lo+hi); fMid = photonResidual(mid)
    lo = np.where(fMid > 0, mid, lo)             # f(lo)>0 invariant
    hi = np.where(fMid <= 0, mid, hi)
root = 0.5*(lo+hi)
if any(non-finite | root<=0 | Δ(root)<0): raise RuntimeError(...)
```
Run once per `sign` (or jointly by stacking a `(B,)` value and a `(1,2)` sign
axis). `60` bisection steps give interval width `≈ hi/2⁶⁰`, far below the
`1e-12` normalized-residual target of NFR-1a for any realistic `l`.

### 3.4 Uniform sampling (FR-5)
```
rArr = linspace(rPPlus, rPMinus, nR)             # nR points, both endpoints included
```
`linspace` handles `rPPlus > rPMinus` (descending) correctly; FR-9 "in order"
means "in `linspace` order", whatever that direction is.

### 3.5 `sin θ` / `sin ψ` with clamping (FR-6)
```
sinTh = clip(sinTheta(rArr, ...), -1, 1)
sinPs = clip(sinPsi(rArr, ...),  -1, 1)          # guards 1+ε endpoints (edge case)
th    = arcsin(sinTh)                            # θ ∈ [0, π/2]
```

### 3.6 Mirror-branch ψ reconstruction (FR-7)
```
psiUpper = arcsin(sinPs)                         # cos ψ > 0 branch, ∈ [−π/2, π/2]
psiUpper = where(psiUpper < 0, psiUpper + 2π, psiUpper)   # wrap into [0, 2π)
psiLower = π − arcsin(sinPs)                     # cos ψ < 0 branch, ∈ [π/2, 3π/2]
```

### 3.7 Canonical `2·nR − 2` ordering (FR-9)
```
idxUpper = arange(0, nR−1)                       # [0 .. nR−2]: r_{p+} + interior (upper)
idxLower = arange(nR−2, 0, −1)                   # [nR−2 .. 1]: interior descending (lower)
rSeq   = concat([rArr[idxUpper],  [rArr[nR−1]],  rArr[idxLower]])      # (2nR−2,)
thSeq  = concat([th[idxUpper],     [th[nR−1]],    th[idxLower]])
psiSeq = concat([psiUpper[idxUpper], [3π/2],     psiLower[idxLower]])  # r_{p−} → 3π/2 once
```
Endpoint `r_{p+}` (index 0) is emitted once with `ψ = π/2` (from `arcsin(1)`);
endpoint `r_{p-}` (index `nR−1`) is emitted once with `ψ = 3π/2`; every interior
radius appears twice, once per branch. Sequence length `(nR−1) + 1 + (nR−2) =
2·nR − 2`. The returned `ψ` has one wrap discontinuity and is non-monotonic,
while `(x, y)` forms a continuous closed curve (edge case).

### 3.8 Stereographic projection (FR-8)
```
x = −2 * tan(thSeq/2) * sin(psiSeq)
y = −2 * tan(thSeq/2) * cos(psiSeq)
```
`th/2 ∈ [0, π/4]`, so `tan` is finite; no singularity.

### 3.9 Batched assembly (FR-10)
```
rArr = rPPlus[:,None] + (rPMinus − rPPlus)[:,None] * t[None,:]   # (B, nR), t=linspace(0,1,nR)
# sinTh/sinPs/th/psiUpper/psiLower all broadcast to (B, nR)
# concatenate along axis=1 with the same idxUpper/[nR−1]/idxLower index arrays → (B, 2nR−2)
return stack([x, y, thSeq, psiSeq, rSeq], axis=−1)               # (B, 2nR−2, 5)
```
No Python loop over `r_p` (NFR-4) and no Python loop over parameter sets (NFR-5).

## 4. Interface

```python
# single set  -> (2·nR − 2, 5), last axis ordered (x, y, θ, ψ, r_p)
shadowBoundary(m: float, Q: float, a: float, l: float, nR: int = 128) -> NDArray

# batch       -> (B, 2·nR − 2, 5), same last-axis order
shadowBoundaryBatch(arrParams: NDArray, nR: int = 128) -> NDArray
# arrParams shape (B, 4), columns ordered (m, Q, a, l)   [decision D6, deferred to plan]
```
- `nR` default 128; `nR < 3` → `ValueError` (FR-5).
- `x`, `y` are dimensionless (the stereographic projection of the sky angles;
  they depend only on the ratios `a/m, Q/m, l/m` and are m-independent).
  `θ`, `ψ` are in radians. `r_p` is a length in units of `m` (it scales linearly
  with `m`). The output is NOT rescaled by `m` (§5, decision D4).
- Error contract:
  - `ValueError`: `m ≤ 0` (FR-1); invalid spin `a² > a_max²`, `a = 0`,
    `|a| = a_max`, `Δ(r_O) ≤ 0` (FR-2); `nR < 3` (FR-5). Batch validation raises
    with the offending index/indices.
  - `RuntimeError`: root-solve failure for an otherwise valid set — no real
    positive root with `Δ(r_p) ≥ 0`, non-convergence, or a non-finite root
    (FR-4), message identifies the parameter set / batch index.

## 5. Decisions, justified with rejected alternatives

- **D1 — Scalar root finder: `scipy.optimize.brentq` (single) + vectorized
  bisection (batch).** Brentq is robust, derivative-free and converges to ~1e-15
  on a sign-changing bracket, but it is strictly scalar, so it cannot satisfy
  NFR-5 (no per-set loop) for the batch. Rejected: `scipy.optimize.newton`
  (needs a derivative or a good initial guess; not guaranteed to converge over
  the whole parameter range); a "vectorized brentq" (no such primitive exists in
  scipy — `np.vectorize` just hides the loop). Bisection is chosen for the batch
  because it is guaranteed and deterministic given the bracket, and fixed
  iteration counts make it trivially vectorizable.
- **D2 — Root labeling by equation (sin ψ endpoint), not by magnitude.** `r_{p+}`
  is the root of the `−` equation and `r_{p-}` the root of the `+` equation,
  exactly as FR-4 defines them, and we *verify* the labeling afterwards via
  `sin ψ(r_{p+}) = +1`, `sin ψ(r_{p-}) = −1` (NFR-1a). Rejected: labeling by
  `r_{p+} < r_{p-}` ordering, which is wrong for `a < 0` (the two roots swap
  order) and would contradict the spec's explicit "labeled by sky angle".
- **D3 — Mirror-branch ψ reconstruction** follows FR-7 verbatim: upper branch
  `ψ = arcsin(sin ψ)` wrapped by `+2π`, lower branch `ψ = π − arcsin(sin ψ)`,
  joined at the shared endpoints (`π/2` at `r_{p+}`, `3π/2` at `r_{p-}`).
  Rejected: reconstructing a single branch and assuming `ψ → 2π − ψ` reflection,
  or reconstructing `ψ` from `cos ψ` via `arccos` (no `cos ψ` formula is given;
  the mirror rule is the only unambiguous choice). The mirror symmetry
  `ψ → π − ψ` is itself a Step-3 validation item (`l ≠ 0`), out of scope here.
- **D4 — Output quantities and their m-dependence.** `(x, y)` are dimensionless
  and m-independent: they depend only on the ratios `a/m, Q/m, l/m`, so scaling
  all four parameters by a common factor leaves `(x, y)` bit-identical. `θ`, `ψ`
  are in radians; `r_p` is a length in units of `m` and scales linearly with `m`.
  `m` remains an input because the observer distance `r_O = 5·m`, the validity
  checks (`Δ(r_O) > 0`, the spin bound), and the returned `r_p` all involve `m`.
  Rejected: forcibly normalizing everything to `m = 1` — it would strip `r_p`'s
  physical scale and complicate the downstream noise model and dataset generator.
- **D5 — Benchmark lives in a test file.** `tests/test_benchmark.py` measures and
  prints the median timings (NFR-6); it is a test file, so it is permitted by
  NFR-2 without a new-source-file approval. Profiling (`cProfile`) is a manual
  `uv run python -m cProfile …` step during implementation, not a committed
  script. Rejected: a `scripts/benchmark.py` or `benchmarks/` source file
  (violates constitution principle 1 "no new files without approval"); a
  notebook (also a new file needing approval).
- **D6 — Batch input column order `(m, Q, a, l)`.** Matches the single-set
  argument order of US-1/FR-1 and `shadowBoundary(m, Q, a, l, …)`, minimizing
  confusion. This was explicitly deferred to the plan (spec open question
  "None" but FR-10 leaves the order unstated).

## 6. Test strategy (pytest, `uv run pytest`)

Tests-first per task (write the failing test, implement, then green). Column in
brackets = the FR/NFR each test covers.

`tests/test_metric.py`
- `delta`/`deltaPrime`/`sigma` vs hand-computed values on the Schwarzschild case
  `(m=1, Q=0, a=0.001, l=0)`. [FR-3, supports FR-4/FR-6]
- `maxSpin`/`outerHorizon` positivity and `outerHorizon = m + √(a_max²−a²)`.
  [FR-2, FR-4]
- `validateParameters`: `m ≤ 0` → `ValueError` [FR-1]; each of `a_max² ≤ 0`,
  `a² > a_max²`, `a = 0`, `|a| = a_max`, `Δ(r_O) ≤ 0` → `ValueError` [FR-2];
  a known-valid set passes. `validateParametersBatch` raises for a batch
  containing one invalid set and reports its index. [FR-1/FR-2]

`tests/test_shadow.py`
- Root residual: for several valid `(m,Q,a,l)` (Schwarzschild-like, `a>0`, `a<0`,
  large `|l|`), normalized residual `|f| / max(|Σ·Δ'|,|4r·Δ|,|4ar√Δ·sinϑ_O|)`
  at each root ≤ 1e-12, plus `f(lo) > 0`, `f(hi) < 0` and monotonicity of `f` on
  the bracket. [FR-4, NFR-1a]
- Root labeling: `sin ψ(r_{p+}) ≈ +1`, `sin ψ(r_{p-}) ≈ −1`. [FR-4, NFR-1a]
- Root failure → `RuntimeError` (monkeypatch/forced non-convergence or a
  deliberately unresolvable set). [FR-4]
- `nR < 3` → `ValueError`. [FR-5]
- Angles: `sin θ`, `sin ψ` agree with the FR-6 formulas to 1e-9; reconstructed
  `θ`, `ψ` agree with `arcsin(sin θ)`, `arcsin(sin ψ)` to 1e-9 rad where
  `|sin ψ| ≤ 1−1e-6` and 1e-6 rad near the endpoints. [FR-6, NFR-1b]
- Projection: `(x, y)` agree with `−2 tan(θ/2)(sin ψ, cos ψ)`. [FR-8, NFR-1b]
- Shape and order: output shape `(2·nR−2, 5)`, columns `(x,y,θ,ψ,r_p)`, first
  point `ψ = π/2`, point at index `nR−1` is `ψ = 3π/2`, interior radii appear
  twice with the two branch `ψ` values. [FR-9]
- Schwarzschild finite-distance anchor: `(m=1,Q=0,l=0,a=a_max/1000)`,
  `r_O = 5`, `sin θ ≈ 0.804984` (±1e-3) and every plane radius `√(x²+y²) ≈
  1.0105` (±1e-3). [FR-6/FR-8, completion criterion]
- Determinism: two calls with identical inputs are bit-identical. [NFR-3]
- Batch: `shadowBoundaryBatch` returns `(B, 2·nR−2, 5)`; `batch[i]` equals
  `shadowBoundary(arrParams[i])` within `atol=1e-6`; mixed valid batch and a
  batch mixing `a>0`/`a<0` all match individual results. [FR-10, NFR-5]
- Column order: swapping `Q` and `a` in the input changes the output
  accordingly (guards the `(m,Q,a,l)` contract). [FR-10, D6]

`tests/test_benchmark.py`
- Runs `shadowBoundary` at `N_r=128` and `shadowBoundaryBatch` at `B=100`, over
  `R=10` repetitions, and prints the median wall-clock time per single call and
  per batch (no speed assertion; the output is the report). [NFR-6]

Not automated here: the `cProfile` profiling and the conditional numba step are
a manual decision gate after T7 (see tasks).

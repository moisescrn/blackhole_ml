# Spec 001 — Forward Model

Status: implemented

## Context and objective

This project infers the parameters of a Kerr-Newman-NUT (KNN) black hole from its shadow. Before any inference can happen, we need the **forward problem**: given the black-hole parameters, compute the shape of the shadow as seen by a distant observer.

The shadow is the boundary (a one-dimensional closed curve) of the black region that the black hole produces on the observer's sky (the *celestial sphere*). Its shape is determined analytically by the photon region of the spacetime.

This spec covers **Step 2 ("Forward model") in full**, i.e. all four of its points:

1. the core function: parameters → shadow boundary as sampled points on the observer's sky, projected to the plane;
2. vectorized evaluation over the orbit radius and batching across parameter sets (no Python loops);
3. robust determination of the two photon-orbit radii `r_{p±}`;
4. benchmarking and profiling, with JIT acceleration introduced only where profiling justifies it.

The objective is a coherent, verifiable unit that turns parameters into sampled shadow boundaries. Everything downstream (dataset generation, inference) is out of scope here.

## Users

- **The inference pipeline developer**, who will feed the sampled boundary into a noise model, an MCMC sampler, and a neural network.
- **The physicist / author**, who needs a correct, transparent implementation of the analytic photon-region equations (with all conventions fixed).
- **The dataset generator**, who needs to compute many shadows (batches of parameter sets) efficiently to build training data.

## User stories

- US-1. As the inference pipeline developer, I want a function that turns a parameter set `(m, Q, a, l)` into sampled points of the shadow boundary on the plane, so that I can later add noise and train/validate estimators on it.
- US-2. As the physicist, I want the returned curve to reproduce the analytic shadow equations exactly (with all conventions fixed), so that I can trust the physics before any inference is built on top.
- US-3. As the dataset generator, I want to compute shadows for a batch of parameter sets in a single vectorized call, so that I can produce large training datasets efficiently and without Python loops.

## Definitions

- **Black-hole parameters.** The four relevant parameters are the mass `m` (`m > 0`), the electric charge `Q`, the spin `a`, and the NUT charge `l`. The Manko-Ruiz parameter `C` does not appear in the shadow parametrization and is ignored. `a` may be positive or negative (the sign encodes the spin direction).
- **Spin validity.** The spin is mathematically valid when `a² ≤ m² + l² − Q²`. The maximal spin is `a_max = √(m² + l² − Q²)`. Because the parametrization is numerically ill-behaved exactly at `a = 0` and `a = a_max`, those exact values are excluded (FR-2) and values very near them are used instead (e.g. `a = a_max/1000` or `a = 999·a_max/1000`).
- **Units and conventions.** `G = c = 1`; all lengths are measured in units of the mass `m`. The observer is fixed at distance `r_O = 5 m` and lies on the equatorial plane `ϑ_O = π/2` (angle between the observer and the rotation axis).
- **Celestial sphere vs. spacetime coordinates.** The spacetime coordinates are `(t, r, ϑ, φ)` with `ϑ ∈ [0, π)` and `φ ∈ [0, 2π)`. The observer's sky is a separate two-dimensional manifold described by the angles `(θ, ψ)`, with `θ ∈ [0, π)` and `ψ ∈ [0, 2π)`. `ϑ` and `θ` are different quantities and must never be confused. The observer position is written in spacetime coordinates (`r_O`, `ϑ_O`); the shadow is parametrized on the sky angles `(θ, ψ)`.
- **Auxiliary quantities.** For the metric we define `Δ(r) = r² − 2mr + a² − l² + Q²`, `Δ'(r) = 2r − 2m`, and `Σ(r, ϑ) = r² + (l + a·cos ϑ)²`.
- **Photon-orbit radii.** The shadow parameter `r_p` runs between the two roots `r_{p+}` and `r_{p-}` of
  `Σ(r_{p±}, ϑ_O)·Δ'(r_{p±}) − 4·r_{p±}·Δ(r_{p±}) ∓ 4·a·r_{p±}·√(Δ(r_{p±}))·sin ϑ_O = 0`,
  where the upper sign (minus) corresponds to `r_{p+}` and the lower sign (plus) to `r_{p-}`. The roots are labeled by their sky angle, not by their ordering: `r_{p+}` is the root where `sin ψ = +1` and `r_{p-}` is the root where `sin ψ = −1` (this holds for either sign of `a`).
- **Sampled points.** `N_r` radii `r_p` are chosen deterministically, covering the interval between `r_{p+}` and `r_{p-}`, inclusive of both endpoints. Each interior `r_p` yields one sky angle `θ` and two `ψ` values (one per mirror branch, FR-7); each endpoint yields a single `ψ` because `sin ψ(r_{p+}) = +1` and `sin ψ(r_{p-}) = −1` (at `ϑ_O = π/2` the `sin ψ` numerator equals `Σ·Δ' − 4·r·Δ`, which at the roots `r_{p±}` equals `±4·a·r·√Δ·sin ϑ_O` per FR-4, making `sin ψ = ±1`). Hence the returned sequence has `2·N_r − 2` points. Each point carries its plane coordinates `(x, y)`, its sky angles `(θ, ψ)`, and the radius `r_p` that produced it. The returned sequence is **open** (the closing point is not duplicated): it traces the closed boundary in order, so the curve closes implicitly between the last and the first point.
- **Batch.** A collection of `B` parameter sets evaluated together in one call. Each set solves its own `r_{p±}`, while a single `N_r` applies to the whole batch.

## Functional requirements

- FR-1. THE SYSTEM shall accept a black-hole parameter set given by the four values `(m, Q, a, l)`. IF `m ≤ 0`, THEN THE SYSTEM shall raise a `ValueError`.
- FR-2. THE SYSTEM shall raise a `ValueError` and produce no shadow in each of the following cases:
  - the spin violates `a² ≤ m² + l² − Q²` (which also covers `m² + l² − Q² < 0`, where no spin is valid);
  - `a = 0` exactly or `|a| = a_max` exactly (numerically singular; near values are used instead);
  - `Δ(r_O) ≤ 0` (the observer is inside or on the horizon, e.g. for large `|l|`).
- FR-3. THE SYSTEM shall use the fixed observer position `r_O = 5 m` and `ϑ_O = π/2` for the shadow parametrization.
- FR-4. THE SYSTEM shall determine the two photon-orbit radii `r_{p+}` and `r_{p-}` by solving, for `r_{p±}`:
  `Σ(r_{p±}, ϑ_O)·Δ'(r_{p±}) − 4·r_{p±}·Δ(r_{p±}) ∓ 4·a·r_{p±}·√(Δ(r_{p±}))·sin ϑ_O = 0`,
  where the upper sign (minus) maps to `r_{p+}` (the root with `sin ψ = +1`) and the lower sign (plus) to `r_{p-}` (the root with `sin ψ = −1`). An admissible root is real and positive, lies outside the outer horizon (i.e. `Δ(r_p) ≥ 0`), and each defining equation has exactly one admissible root. IF no admissible root exists, the root does not converge, or the root is complex, THEN THE SYSTEM shall raise a `RuntimeError` with a clear message identifying the parameter set.
- FR-5. THE SYSTEM shall sample `r_p` uniformly on the interval between `r_{p+}` and `r_{p-}`, including both endpoints, with a caller-selectable number of points `N_r` defaulting to `N_r = 128`. IF `N_r < 3`, THEN THE SYSTEM shall raise a `ValueError`.
- FR-6. For each sampled `r_p`, THE SYSTEM shall compute the celestial-sphere angles via
  `sin θ(r_p) = 4·r_p·√(Δ(r_O)·Δ(r_p)) / [ Δ'(r_p)·(r_O² − r_p²) + 4·r_p·Δ(r_p) ]`
  and
  `sin ψ(r_p) = [ Δ'(r_p)·(r_p² + (l + a·cos²ϑ_O)²) − 4·r_p·Δ(r_p) ] / [ 4·r_p·a·√(Δ(r_p))·sin ϑ_O ]`,
  with both `sin θ` and `sin ψ` clamped to `[−1, 1]` before any `arcsin`. The `sin ψ` numerator uses `(l + a·cos²ϑ_O)²` exactly as written; at the fixed `ϑ_O = π/2` this reduces to `l²`, and this form must be re-confirmed against the thesis before any future free-inclination work.
- FR-7. THE SYSTEM shall reconstruct the full sky angle `ψ` from `sin ψ` using the mirror-branch rule: on the `cos ψ > 0` branch, `ψ = arcsin(sin ψ) ∈ [−π/2, π/2]`; on the `cos ψ < 0` branch, `ψ = π − arcsin(sin ψ)` (the mirror across the x-axis in the plane); the two branches are joined at the endpoints `r_{p±}`; negative `ψ` values are wrapped by `+2π` into `[0, 2π)`. At each shared endpoint a single `ψ` is emitted: `ψ = π/2` at `r_{p+}` (`sin ψ = +1`) and `ψ = 3π/2` at `r_{p-}` (`sin ψ = −1`).
- FR-8. THE SYSTEM shall reconstruct `θ = arcsin(sin θ)`, with `θ ∈ [0, π/2]` (`sin θ` can reach exactly `1` when a sampled `r_p` equals `r_O`; `arcsin` is unambiguous because `sin θ ≤ 1` always), and map each `(θ, ψ)` pair to plane coordinates using the stereographic projection
  `(x, y) = −2·tan(θ/2)·(sin ψ, cos ψ)`.
- FR-9. THE SYSTEM shall return the shadow boundary as an ordered, open sequence of `2·N_r − 2` points — each carrying `(x, y)` together with its intermediate values `(θ, ψ, r_p)` — tracing the closed boundary in one continuous direction. Canonical order: begin at the endpoint `r_{p+}` (`sin ψ = +1`, `ψ = π/2`), traverse the `cos ψ > 0` branch through the sampled radii in order to the endpoint `r_{p-}` (`sin ψ = −1`, `ψ = 3π/2`), then traverse the `cos ψ < 0` branch through the remaining interior radii back to `r_{p+}`, which is not repeated.
- FR-10. THE SYSTEM shall evaluate the shadow boundary vectorized over `r_p`, and shall compute shadows for a batch of `B` parameter sets at once: given an array of `B` parameter sets, it shall return the corresponding `B` boundaries, without a per-parameter-set Python loop. Each set in the batch solves its own `r_{p±}`; the same `N_r` applies to every set. The result is an array of shape `(B, 2·N_r − 2, 5)` whose last axis is ordered `(x, y, θ, ψ, r_p)`.

## Non-functional requirements

- NFR-1 (Physics correctness, split). THE SYSTEM shall be verified as follows:
  - (a) `r_{p±}`: since there is no closed-form reference, the two roots are verified by the residual of their defining equation (FR-4), normalized by the operator's characteristic scale `max(|Σ·Δ'|, |4·r·Δ|, |4·a·r·√Δ·sin ϑ_O|)` evaluated at the root, being `≤ 1e-12`, together with correct bracketing and monotonicity of the residual around each root.
  - (b) angles and projection (FR-6, FR-8): the computed `sin θ`, `sin ψ`, and `(x, y)` agree with the analytic formulas to an absolute tolerance of `1e-9` (dimensionless); the reconstructed angles `θ` and `ψ` agree with `arcsin(sin θ)` and `arcsin(sin ψ)` to an absolute tolerance of `1e-9` rad where `|sin ψ| ≤ 1 − 1e-6`, and `1e-6` rad where `|sin ψ| > 1 − 1e-6` (near the endpoints, where `arcsin` accuracy degrades like the square root).
- NFR-2 (Stack simplicity). No new dependency or package beyond the existing stack shall be introduced (test files required by this spec are allowed).
- NFR-3 (Determinism). WHEN the same valid parameter set is provided, THE SYSTEM shall return bit-identical sampled points; the function is pure (no global state, no hidden randomness).
- NFR-4 (Inner path efficiency). THE SYSTEM shall evaluate the boundary over the `N_r` sampled radii using vectorized array operations, with no Python `for` loop over points in the inner path.
- NFR-5 (Batch efficiency). THE SYSTEM shall compute a batch of parameter sets without a Python `for` loop over parameter sets.
- NFR-6 (Benchmark before acceleration). THE SYSTEM shall be benchmarked and profiled at `N_r = 128`, batch size `B = 100`, over `R = 10` repetitions, reporting the median wall-clock time per single call and per batch. JIT acceleration (e.g. Numba) shall be introduced only if profiling shows the boundary evaluation is the dominant cost (≥ 50% of runtime) AND the measured JIT speedup is ≥ 2×.

## Edge cases

- **Static limit `a → 0`.** The `sin ψ` formula has `a` in the denominator; exactly `a = 0` raises a `ValueError` (FR-2). The function is required to be correct only for `a ≠ 0` (values very near zero, e.g. `a_max/1000`, are allowed) and must not divide by zero.
- **Extremal limit `|a| = a_max`.** `√(Δ(r_p))` may vanish at the horizon; exactly `|a| = a_max` raises a `ValueError` (FR-2) and near-extremal values (e.g. `a = ±999·a_max/1000`) are used.
- **Negative spin.** `a < 0` (retrograde spin) is a valid input and must be handled.
- **No valid spin range.** If `m² + l² − Q² < 0`, no `a` satisfies the validity condition; the parameter set is invalid (FR-2).
- **Observer at or below the horizon.** If `Δ(r_O) ≤ 0` (observer inside or on the horizon, e.g. for large `|l|`), the system raises a `ValueError` (FR-2).
- **Interval collapse.** As `a → 0` the interval between `r_{p+}` and `r_{p-}` shrinks toward a single radius and the shadow approaches a circle; the function must remain well-defined for the near-zero `a` case.
- **Angular branch of `ψ`.** The analytic formula gives only `sin ψ`; the mirror-branch rule (FR-7) reconstructs the two branches and joins them at the endpoints `r_{p±}` to form the single closed curve, with negative `ψ` wrapped into `[0, 2π)`.
- **`sin` slightly out of range.** At a numerically computed endpoint the formulas can return `sin ψ = 1 + ε` (e.g. `1 + 4.4e-16`), which would make `arcsin` raise; both `sin θ` and `sin ψ` are therefore clamped to `[−1, 1]` before `arcsin` (FR-6). The interval endpoints are included in the `N_r` samples.
- **θ branch.** `θ = arcsin(sin θ)` with `θ ∈ [0, π/2]`; `sin θ` reaches exactly `1` only when a sampled `r_p` equals `r_O`, and `arcsin` remains unambiguous because `sin θ ≤ 1` always (FR-8).
- **ψ discontinuity.** The returned `ψ` array has one wrap discontinuity (from `arcsin` to `+2π`) and is not monotonic, whereas the plane coordinates `(x, y)` form a continuous closed curve.
- **Open-curve representation.** The returned sequence has exactly `2·N_r − 2` points (the closing point is not repeated); the boundary is closed geometrically but the sequence is open, closing implicitly from the last point back to the first.
- **Root failure.** For an otherwise valid parameter set, a failed root solve (no real positive root with `Δ(r_p) ≥ 0`, non-convergence, or a complex root) raises a `RuntimeError` (FR-4).

## Out of scope

- **Full physical validation** (Step 3): the systematic limit-test suite (KNN → Kerr-Newman → Kerr → Schwarzschild) and the independent numerical geodesic (ray-tracing) cross-check. The finite-distance Schwarzschild anchor below is a correctness anchor for this spec, not the full validation. In particular, the mirror symmetry `ψ → π − ψ` assumed by FR-7 must be validated in Step 3 for `l ≠ 0`.
- **Noise models and dataset generation** (Step 4).
- **Free observer position / inclination** as parameters (future work).
- **Inference** (MCMC baseline and neural-network estimator, Steps 5-6).

## Completion criteria

- A valid parameter set `(m, Q, a, l)` is mapped to an ordered, open sequence of `2·N_r − 2` sampled points `(x, y)` (together with `(θ, ψ, r_p)`) in the canonical order of FR-9, using uniform sampling between `r_{p+}` and `r_{p-}` (FR-5).
- The two roots `r_{p±}` satisfy their defining equation with normalized residual `≤ 1e-12` (NFR-1a), with correct bracketing and monotonicity, and are labeled correctly: `r_{p+}` has `sin ψ = +1`, `r_{p-}` has `sin ψ = −1`.
- The angles and plane points agree with the analytic formulas (FR-6, FR-8) to the absolute tolerances stated in NFR-1b.
- Schwarzschild finite-distance anchor: for `m = 1`, `Q = 0`, `l = 0`, `a = a_max/1000` (`a → 0`) at `r_O = 5`, the shadow is a circle with `sin θ = 12·√45/100 ≈ 0.804984` (equivalently `θ ≈ 0.93565 rad`, plane radius `2·tan(θ/2) ≈ 1.0105`), checked with tolerance `1e-3` on `sin θ` (the deviation is `O(a)`); each point's plane radius is within `1e-3` of `1.0105`.
- A batch of `B` parameter sets (FR-10) returns an array of shape `(B, 2·N_r − 2, 5)` that matches the per-set results of individual evaluation, with no per-parameter-set Python loop (NFR-5).
- The benchmark reports the median timing per call and per batch at `N_r = 128`, `B = 100`, `R = 10`, and JIT acceleration is introduced only with a measured speedup (NFR-6).
- Invalid inputs behave as specified: `m ≤ 0`, invalid spin, `a = 0` or `|a| = a_max` exactly, `Δ(r_O) ≤ 0`, and `N_r < 3` raise a `ValueError` (FR-1, FR-2, FR-5); a failed root solve raises a `RuntimeError` (FR-4).
- Determinism holds: the same valid input yields bit-identical output (NFR-3); no new dependency or package is introduced (NFR-2).
- The behavior described by FR-1 through FR-10 is covered by automated tests, and `uv run pytest` passes (constitution principle 3).

## Open questions

None.

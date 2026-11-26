# Constitution — Black Hole Machine Learning

Non-negotiable principles. Every spec, plan, and task must comply with them.

1. **Stack Simplicity**: Only the existing stack and structure may be used. No new dependencies, packages, or files without prior explicit approval.
2. **The Spec Is Authoritative**: Nothing is implemented unless it is in the active spec (`opencode/specs/NNN-*/spec.md`). If a decision is missing, stop and ask.
3. **Tests as a Gate**: Logic must be tested using `uv run pytest`, and check that all tests are bypassed
4. **User Data Is Sacred**: Never modify, delete, or leak user files or data; never store secrets in code, specs, or `MEMORY.md`.
5. **Minimize computational cost**: If it is possible avoid always Python loops in the inner path (per-point or per-sample loops). Vectorize first, profile (cProfile, line_profiler, timeit) and then consider numba

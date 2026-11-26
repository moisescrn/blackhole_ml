"""Wall-clock benchmark of the single and batched shadow evaluation."""

import time
from collections.abc import Callable

import numpy as np

from knn_shadow.shadow import shadowBoundary, shadowBoundaryBatch

N_R = 128
B_SIZE = 100
R_REPS = 10


def medianCallTime(call: Callable[[], object], nReps: int) -> float:
    """Median wall-clock seconds of nReps consecutive calls of `call`."""
    fSamples: list[float] = []
    for _ in range(nReps):
        fStart = time.perf_counter()
        call()
        fSamples.append(time.perf_counter() - fStart)
    return float(np.median(fSamples))


def makeParams(nSets: int) -> np.ndarray:
    """Deterministic valid batch: mixed spin signs, small charge and NUT charge."""
    fT = np.linspace(0.0, 1.0, nSets)
    fA = np.linspace(-0.9, 0.9, nSets)
    fA = np.where(np.abs(fA) < 1e-2, 1e-2, fA)
    fM = np.ones(nSets)
    fQ = 0.3 * fT
    fL = 0.4 * fT
    return np.column_stack([fM, fQ, fA, fL])


def testBenchmarkSingleShadowBoundary() -> None:
    fM, fQ, fA, fL = 1.0, 0.3, 0.4, 0.2
    shadowBoundary(fM, fQ, fA, fL, N_R)  # warm-up
    fMedian = medianCallTime(lambda: shadowBoundary(fM, fQ, fA, fL, N_R), R_REPS)
    print(
        f"\n[benchmark] shadowBoundary nR={N_R}, R={R_REPS}: "
        f"median {fMedian * 1e3:.3f} ms/call"
    )
    arrOut = shadowBoundary(fM, fQ, fA, fL, N_R)
    assert arrOut.shape == (2 * N_R - 2, 5)
    assert np.all(np.isfinite(arrOut))


def testBenchmarkShadowBoundaryBatch() -> None:
    arrParams = makeParams(B_SIZE)
    shadowBoundaryBatch(arrParams, N_R)  # warm-up
    fMedian = medianCallTime(lambda: shadowBoundaryBatch(arrParams, N_R), R_REPS)
    print(
        f"\n[benchmark] shadowBoundaryBatch B={B_SIZE}, nR={N_R}, R={R_REPS}: "
        f"median {fMedian * 1e3:.3f} ms/batch"
    )
    arrOut = shadowBoundaryBatch(arrParams, N_R)
    assert arrOut.shape == (B_SIZE, 2 * N_R - 2, 5)
    assert np.all(np.isfinite(arrOut))

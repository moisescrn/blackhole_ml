import numpy as np
import pytest
from scipy import optimize

from knn_shadow import shadow
from knn_shadow.metric import delta, deltaPrime, maxSpin, outerHorizon
from knn_shadow.shadow import (
    photonResidual,
    shadowBoundary,
    shadowBoundaryBatch,
    sinPsi,
    sinTheta,
    solvePhotonRadii,
    solvePhotonRadiiBatch,
    stereographicProjection,
)


def refSinTheta(
    rArr: np.ndarray,
    m: float,
    Q: float,
    a: float,
    l: float,  # noqa: E741
) -> np.ndarray:
    """Independent hand-written copy of the sin(theta) formula."""
    fRO = 5.0 * m
    fDeltaO = fRO**2 - 2.0 * m * fRO + a**2 - l**2 + Q**2
    fDeltaR = rArr**2 - 2.0 * m * rArr + a**2 - l**2 + Q**2
    fDeltaPrime = 2.0 * rArr - 2.0 * m
    return (
        4.0
        * rArr
        * np.sqrt(fDeltaO * fDeltaR)
        / (fDeltaPrime * (fRO**2 - rArr**2) + 4.0 * rArr * fDeltaR)
    )


def refSinPsi(
    rArr: np.ndarray,
    m: float,
    Q: float,
    a: float,
    l: float,  # noqa: E741
) -> np.ndarray:
    """Independent hand-written copy of the sin(psi) formula at vartheta_O = pi/2."""
    fDeltaR = rArr**2 - 2.0 * m * rArr + a**2 - l**2 + Q**2
    fDeltaPrime = 2.0 * rArr - 2.0 * m
    return (fDeltaPrime * (rArr**2 + l**2) - 4.0 * rArr * fDeltaR) / (
        4.0 * rArr * a * np.sqrt(fDeltaR)
    )


PARAM_SETS = [
    (1.0, 0.0, 1e-3, 0.0),
    (1.0, 0.0, 0.5, 0.0),
    (1.0, 0.0, -0.5, 0.0),
    (1.0, 0.2, 0.4, 0.3),
    (2.0, 0.0, -1.5, 0.0),
]

RADII = [
    np.array([3.0, 4.0, 5.0, 7.5]),
    np.array([2.5, 3.5, 6.0]),
    np.array([3.0, 4.5, 8.0]),
    np.array([2.5, 4.0, 9.0]),
    np.array([4.0, 5.5, 9.0]),
]


@pytest.mark.parametrize("params, rArr", list(zip(PARAM_SETS, RADII, strict=True)))
def testSinThetaMatchesFormula(
    params: tuple[float, float, float, float], rArr: np.ndarray
) -> None:
    fM, fQ, fA, fL = params
    fExpected = refSinTheta(rArr, fM, fQ, fA, fL)
    fActual = sinTheta(rArr, fM, fQ, fA, fL)
    assert fActual.shape == rArr.shape
    assert fActual == pytest.approx(fExpected, abs=1e-9)


@pytest.mark.parametrize("params, rArr", list(zip(PARAM_SETS, RADII, strict=True)))
def testSinPsiMatchesFormula(
    params: tuple[float, float, float, float], rArr: np.ndarray
) -> None:
    fM, fQ, fA, fL = params
    fExpected = refSinPsi(rArr, fM, fQ, fA, fL)
    fActual = sinPsi(rArr, fM, fQ, fA, fL)
    assert fActual.shape == rArr.shape
    assert fActual == pytest.approx(fExpected, abs=1e-9)


def testSinThetaScalarInput() -> None:
    assert sinTheta(3.0, 1.0, 0.0, 1e-3, 0.0) == pytest.approx(
        refSinTheta(np.array(3.0), 1.0, 0.0, 1e-3, 0.0), abs=1e-9
    )


def testSinThetaSchwarzschildAnchor() -> None:
    fExpected = 12.0 * np.sqrt(45.0) / 100.0
    assert fExpected == pytest.approx(0.80498447, abs=1e-8)
    assert sinTheta(3.0, 1.0, 0.0, 1e-3, 0.0) == pytest.approx(fExpected, abs=1e-3)


def testSinThetaSchwarzschildHandValue() -> None:
    fValue = sinTheta(np.array([3.0]), 1.0, 0.0, 1e-3, 0.0)
    assert fValue[0] == pytest.approx(0.80498447, abs=1e-3)


def testSinPsiReturnsRawValueWithoutClamping() -> None:
    # r = 2 with (m=1, Q=0, a=0.5, l=0) gives a raw sin(psi) of 3, far above 1.
    fRaw = sinPsi(np.array([2.0]), 1.0, 0.0, 0.5, 0.0)
    assert fRaw[0] == pytest.approx(3.0, abs=1e-9)
    fClipped = np.clip(fRaw, -1.0, 1.0)
    assert fClipped[0] == 1.0
    assert np.isfinite(np.arcsin(fClipped[0]))


def testClipGuardHandlesEndpointOvershoot() -> None:
    fRaw = 1.0 + 4.4e-16
    fClipped = np.clip(fRaw, -1.0, 1.0)
    assert fClipped <= 1.0
    assert np.arcsin(fClipped) == pytest.approx(np.pi / 2.0)


ROOT_SETS = [
    (1.0, 0.0, 0.5, 0.0),
    (1.0, 0.3, -0.4, 0.2),
    (1.0, 0.0, 0.5, 1.5),
]

SIGNS = (-1.0, 1.0)


def refPhotonResidual(
    rArr: np.ndarray,
    m: float,
    Q: float,
    a: float,
    l: float,  # noqa: E741
    fSign: float,
) -> np.ndarray:
    """Hand-written copy of the photon-orbit residual at vartheta_O = pi/2."""
    fDeltaR = rArr**2 - 2.0 * m * rArr + a**2 - l**2 + Q**2
    fDeltaPrime = 2.0 * rArr - 2.0 * m
    return (
        (rArr**2 + l**2) * fDeltaPrime
        - 4.0 * rArr * fDeltaR
        + fSign * 4.0 * a * rArr * np.sqrt(fDeltaR)
    )


def refScale(
    fR: float,
    m: float,
    Q: float,
    a: float,
    l: float,  # noqa: E741
) -> float:
    """Characteristic scale of the residual operator at one radius."""
    rArr = np.array([fR])
    fDeltaR = float(delta(rArr, m, Q, a, l)[0])
    fTermSigma = (fR**2 + l**2) * float(deltaPrime(rArr, m)[0])
    fTermFour = 4.0 * fR * fDeltaR
    fTermSpin = 4.0 * abs(a) * fR * np.sqrt(fDeltaR)
    return max(abs(fTermSigma), abs(fTermFour), abs(fTermSpin))


def bracketOf(
    m: float,
    Q: float,
    a: float,
    l: float,  # noqa: E741
) -> tuple[float, float]:
    """Documented bracket [r_+ + delta, 4 (a_max + |l| + m)]."""
    fRPlus = outerHorizon(m, Q, a, l)
    fDelta = 1e-9 * max(1.0, fRPlus)
    return fRPlus + fDelta, 4.0 * (maxSpin(m, Q, l) + abs(l) + m)


@pytest.mark.parametrize("params", ROOT_SETS)
def testRootParameterSetsAreValid(params: tuple[float, float, float, float]) -> None:
    fM, fQ, fA, fL = params
    fAMax = maxSpin(fM, fQ, fL)
    assert fA != 0.0
    assert fA**2 < fAMax**2
    assert float(delta(np.array([5.0 * fM]), fM, fQ, fA, fL)[0]) > 0.0


@pytest.mark.parametrize("params", ROOT_SETS)
@pytest.mark.parametrize("fSign", SIGNS)
def testPhotonResidualBracketSigns(
    params: tuple[float, float, float, float], fSign: float
) -> None:
    fM, fQ, fA, fL = params
    fLo, fHi = bracketOf(fM, fQ, fA, fL)
    fLoVal = float(photonResidual(fLo, fM, fQ, fA, fL, fSign))
    fHiVal = float(photonResidual(fHi, fM, fQ, fA, fL, fSign))
    assert fLoVal > 0.0
    assert fHiVal < 0.0


@pytest.mark.parametrize("params", ROOT_SETS)
def testPhotonResidualMatchesFormula(
    params: tuple[float, float, float, float],
) -> None:
    fM, fQ, fA, fL = params
    fRArr = np.array([3.0, 4.0, 5.5, 8.0])
    for fSign in SIGNS:
        fExpected = refPhotonResidual(fRArr, fM, fQ, fA, fL, fSign)
        fActual = photonResidual(fRArr, fM, fQ, fA, fL, fSign)
        assert fActual.shape == fRArr.shape
        assert fActual == pytest.approx(fExpected, abs=1e-12)


@pytest.mark.parametrize("params", ROOT_SETS)
@pytest.mark.parametrize("fSign", SIGNS)
def testPhotonRootResidualIsSmall(
    params: tuple[float, float, float, float], fSign: float
) -> None:
    fM, fQ, fA, fL = params
    fRPlus, fRMinus = solvePhotonRadii(fM, fQ, fA, fL)
    fRoot = fRPlus if fSign < 0.0 else fRMinus
    fResidual = float(refPhotonResidual(np.array([fRoot]), fM, fQ, fA, fL, fSign)[0])
    assert abs(fResidual) / refScale(fRoot, fM, fQ, fA, fL) <= 1e-12


@pytest.mark.parametrize("params", ROOT_SETS)
@pytest.mark.parametrize("fSign", SIGNS)
def testPhotonRootIsUniqueAndMonotoneAroundRoot(
    params: tuple[float, float, float, float], fSign: float
) -> None:
    fM, fQ, fA, fL = params
    fLo, fHi = bracketOf(fM, fQ, fA, fL)
    fRPlus, fRMinus = solvePhotonRadii(fM, fQ, fA, fL)
    fRoot = fRPlus if fSign < 0.0 else fRMinus
    # One sign change only: f(lo) > 0 and f(r) < 0 for every r beyond the root.
    fScan = np.linspace(fLo, fHi, 20001)
    fVals = refPhotonResidual(fScan, fM, fQ, fA, fL, fSign)
    assert np.count_nonzero(np.diff(np.sign(fVals)) != 0) == 1
    # Strictly decreasing on a window around the root.
    fWindow = np.linspace(fRoot * (1.0 - 1e-2), fRoot * (1.0 + 1e-2), 201)
    fWinVals = photonResidual(fWindow, fM, fQ, fA, fL, fSign)
    assert np.all(np.diff(fWinVals) < 0.0)


@pytest.mark.parametrize("params", ROOT_SETS)
def testPhotonRootsAreLabeledBySinPsi(
    params: tuple[float, float, float, float],
) -> None:
    fM, fQ, fA, fL = params
    fRPlus, fRMinus = solvePhotonRadii(fM, fQ, fA, fL)
    assert float(sinPsi(fRPlus, fM, fQ, fA, fL)) == pytest.approx(1.0, abs=1e-3)
    assert float(sinPsi(fRMinus, fM, fQ, fA, fL)) == pytest.approx(-1.0, abs=1e-3)


def testSolvePhotonRadiiIsDeterministic() -> None:
    fFirst = solvePhotonRadii(1.0, 0.3, -0.4, 0.2)
    fSecond = solvePhotonRadii(1.0, 0.3, -0.4, 0.2)
    assert fFirst == fSecond


def testSolvePhotonRadiiRaisesWhenBracketNeverCloses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No sign change is ever reached: the upper bound must grow without bound.
    def fAlwaysPositive(
        rArr: np.ndarray,
        fM: float,
        fQ: float,
        fA: float,
        fL: float,
        fSign: float,
    ) -> np.ndarray:
        return np.ones_like(np.asarray(rArr, dtype=np.float64))

    monkeypatch.setattr(shadow, "photonResidual", fAlwaysPositive)
    with pytest.raises(RuntimeError, match="Photon-orbit root solve failed"):
        solvePhotonRadii(1.0, 0.0, 0.5, 0.0)


def testSolvePhotonRadiiRaisesWhenBrentqFails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-convergent brentq must surface as RuntimeError, not ValueError.
    def fFailingBrentq(*args: object, **kwargs: object) -> float:
        raise ValueError("no sign change over the bracket")

    monkeypatch.setattr(optimize, "brentq", fFailingBrentq)
    with pytest.raises(RuntimeError, match="Photon-orbit root solve failed"):
        solvePhotonRadii(1.0, 0.0, 0.5, 0.0)


def testSolvePhotonRadiiRaisesWhenRootIsNotAdmissible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A negative root violates the positivity requirement of FR-4.
    monkeypatch.setattr(optimize, "brentq", lambda *a, **k: -1.0)
    with pytest.raises(RuntimeError, match="Photon-orbit root solve failed"):
        solvePhotonRadii(1.0, 0.0, 0.5, 0.0)


def refProjection(
    thArr: np.ndarray, psiArr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Hand-written copy of the stereographic projection."""
    fScale = 2.0 * np.tan(thArr / 2.0)
    return -fScale * np.sin(psiArr), -fScale * np.cos(psiArr)


def testStereographicProjectionMatchesFormula() -> None:
    fTh = np.array([0.0, 0.3, np.pi / 2.0])
    fPsi = np.array([0.0, np.pi / 2.0, 4.0])
    fX, fY = stereographicProjection(fTh, fPsi)
    fXRef, fYRef = refProjection(fTh, fPsi)
    assert fX == pytest.approx(fXRef, abs=1e-12)
    assert fY == pytest.approx(fYRef, abs=1e-12)


def testShadowBoundaryShapeAndCanonicalOrder() -> None:
    fM, fQ, fA, fL, fN = 1.0, 0.0, 0.5, 0.0, 16
    arr = shadowBoundary(fM, fQ, fA, fL, fN)
    fRPlus, fRMinus = solvePhotonRadii(fM, fQ, fA, fL)
    assert arr.shape == (2 * fN - 2, 5)
    # First point: r_p+ with psi = pi/2.
    assert arr[0, 4] == fRPlus
    assert arr[0, 3] == pytest.approx(np.pi / 2.0, abs=1e-12)
    # Index nR-1: r_p- with psi = 3pi/2.
    assert arr[fN - 1, 4] == fRMinus
    assert arr[fN - 1, 3] == pytest.approx(3.0 * np.pi / 2.0, abs=1e-12)
    # Interior radii appear twice, in the canonical traversal order.
    fUpperRadii = arr[1 : fN - 1, 4]
    fLowerRadii = arr[fN:, 4]
    assert np.array_equal(fLowerRadii[::-1], fUpperRadii)


def testShadowBoundaryInteriorPointsShareSinPsi() -> None:
    fM, fQ, fA, fL, fN = 1.0, 0.0, 0.5, 0.0, 16
    arr = shadowBoundary(fM, fQ, fA, fL, fN)
    # Both branch psi values reproduce the same sampled sin(psi).
    for fSlice in (slice(1, fN - 1), slice(fN, 2 * fN - 2)):
        fRadii = arr[fSlice, 4]
        fSinPs = np.clip(sinPsi(fRadii, fM, fQ, fA, fL), -1.0, 1.0)
        assert np.sin(arr[fSlice, 3]) == pytest.approx(fSinPs, abs=1e-12)


def testShadowBoundaryThetaIsArcsinOfClampedSinTheta() -> None:
    fM, fQ, fA, fL, fN = 1.0, 0.3, 0.4, 0.2, 12
    arr = shadowBoundary(fM, fQ, fA, fL, fN)
    fSinTh = np.clip(sinTheta(arr[:, 4], fM, fQ, fA, fL), -1.0, 1.0)
    assert arr[:, 2] == pytest.approx(np.arcsin(fSinTh), abs=1e-12)
    assert np.all(arr[:, 2] >= 0.0) and np.all(arr[:, 2] <= np.pi / 2.0)


def testShadowBoundaryProjectionMatchesAngles() -> None:
    fM, fQ, fA, fL, fN = 1.0, 0.0, -0.5, 0.0, 12
    arr = shadowBoundary(fM, fQ, fA, fL, fN)
    fXRef, fYRef = refProjection(arr[:, 2], arr[:, 3])
    assert arr[:, 0] == pytest.approx(fXRef, abs=1e-12)
    assert arr[:, 1] == pytest.approx(fYRef, abs=1e-12)


def testShadowBoundaryRaisesForTooFewPoints() -> None:
    with pytest.raises(ValueError, match="nR"):
        shadowBoundary(1.0, 0.0, 0.5, 0.0, 2)
    with pytest.raises(ValueError, match="nR"):
        shadowBoundary(1.0, 0.0, 0.5, 0.0, 1)


@pytest.mark.parametrize(
    "params",
    [
        (0.0, 0.0, 0.5, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 2.0, 0.0),
    ],
)
def testShadowBoundaryDelegatesToValidation(
    params: tuple[float, float, float, float],
) -> None:
    fM, fQ, fA, fL = params
    with pytest.raises(ValueError):
        shadowBoundary(fM, fQ, fA, fL, 16)


def testShadowBoundarySchwarzschildFiniteDistanceAnchor() -> None:
    # a -> 0: constant circle sin(theta) = 12 sqrt(45) / 100 at r_O = 5 m.
    fM, fQ, fA, fL, fN = 1.0, 0.0, 1e-3, 0.0, 16
    arr = shadowBoundary(fM, fQ, fA, fL, fN)
    assert np.sin(arr[:, 2]) == pytest.approx(12.0 * np.sqrt(45.0) / 100.0, abs=1e-3)
    fRadius = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
    assert fRadius == pytest.approx(1.0105, abs=1e-3)


def testShadowBoundaryIsDeterministic() -> None:
    fFirst = shadowBoundary(1.0, 0.3, 0.4, 0.2, 16)
    fSecond = shadowBoundary(1.0, 0.3, 0.4, 0.2, 16)
    assert np.array_equal(fFirst, fSecond)


BATCH_SETS = [
    (1.0, 0.0, 0.5, 0.0),
    (1.0, 0.0, -0.5, 0.0),
    (1.0, 0.3, -0.4, 0.2),
    (1.0, 0.0, 0.5, 1.5),
]


def testSolvePhotonRadiiBatchMatchesSingle() -> None:
    arrParams = np.array(BATCH_SETS, dtype=np.float64)
    fPlus, fMinus = solvePhotonRadiiBatch(arrParams)
    assert fPlus.shape == (len(BATCH_SETS),)
    assert fMinus.shape == (len(BATCH_SETS),)
    for i, params in enumerate(BATCH_SETS):
        fRefPlus, fRefMinus = solvePhotonRadii(*params)
        assert fPlus[i] == pytest.approx(fRefPlus, rel=1e-9, abs=1e-12)
        assert fMinus[i] == pytest.approx(fRefMinus, rel=1e-9, abs=1e-12)


def testSolvePhotonRadiiBatchAcceptsNearExtremalRoots() -> None:
    # Near-extremal spin: 0 < delta(r_p+) < 1, still an admissible root.
    arrParams = np.array([[1.0, 0.3, 0.9, 0.4]], dtype=np.float64)
    fPlus, fMinus = solvePhotonRadiiBatch(arrParams)
    fRefPlus, fRefMinus = solvePhotonRadii(1.0, 0.3, 0.9, 0.4)
    assert float(delta(fPlus, 1.0, 0.3, 0.9, 0.4)[0]) < 1.0
    assert fPlus[0] == pytest.approx(fRefPlus, rel=1e-9, abs=1e-12)
    assert fMinus[0] == pytest.approx(fRefMinus, rel=1e-9, abs=1e-12)


def testShadowBoundaryBatchShape() -> None:
    arrParams = np.array(BATCH_SETS, dtype=np.float64)
    fN = 16
    arr = shadowBoundaryBatch(arrParams, fN)
    assert arr.shape == (len(BATCH_SETS), 2 * fN - 2, 5)


def testShadowBoundaryBatchColumnOrder() -> None:
    # Swapping Q and a must be read as (m, Q, a, l), not (m, a, Q, l).
    arrParams = np.array([[1.0, 0.3, 0.4, 0.2], [1.0, 0.4, 0.3, 0.2]])
    arr = shadowBoundaryBatch(arrParams, 16)
    for i, params in enumerate(arrParams):
        assert arr[i] == pytest.approx(shadowBoundary(*params, 16), abs=1e-6)
    assert not np.allclose(arr[0], arr[1])


def testShadowBoundaryBatchMatchesIndividual() -> None:
    arrParams = np.array(BATCH_SETS, dtype=np.float64)
    fN = 16
    arr = shadowBoundaryBatch(arrParams, fN)
    for i, params in enumerate(BATCH_SETS):
        assert arr[i] == pytest.approx(shadowBoundary(*params, fN), abs=1e-6)


def testShadowBoundaryBatchRaisesForInvalidRow() -> None:
    arrParams = np.array(
        [[1.0, 0.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, -0.5, 0.0]]
    )
    with pytest.raises(ValueError, match="row"):
        shadowBoundaryBatch(arrParams, 16)


def testShadowBoundaryBatchRaisesForTooFewPoints() -> None:
    arrParams = np.array([[1.0, 0.0, 0.5, 0.0]])
    with pytest.raises(ValueError, match="nR"):
        shadowBoundaryBatch(arrParams, 2)


@pytest.mark.parametrize("params", [*ROOT_SETS, (1.0, 0.0, 1e-3, 0.0)])
def testShadowBoundarySamplesPhotonRadiiUniformly(
    params: tuple[float, float, float, float],
) -> None:
    fM, fQ, fA, fL = params
    fN = 12
    arr = shadowBoundary(fM, fQ, fA, fL, fN)
    fRPlus, fRMinus = solvePhotonRadii(fM, fQ, fA, fL)
    # Uniform in r_p with both endpoints included; interiors visited twice.
    fLinspace = np.linspace(fRPlus, fRMinus, fN)
    fExpected = np.concatenate(
        [fLinspace[0 : fN - 1], [fLinspace[fN - 1]], fLinspace[fN - 2 : 0 : -1]]
    )
    np.testing.assert_array_equal(arr[:, 4], fExpected)
    assert np.allclose(np.diff(fLinspace), np.diff(fLinspace)[0], rtol=1e-12)


def testShadowBoundaryDefaultSamplingCount() -> None:
    arr = shadowBoundary(1.0, 0.0, 0.5, 0.0)
    assert arr.shape == (254, 5)


def testShadowBoundaryBatchDefaultSamplingCount() -> None:
    arrParams = np.array([[1.0, 0.0, 0.5, 0.0], [1.0, 0.0, -0.5, 0.0]])
    arr = shadowBoundaryBatch(arrParams)
    assert arr.shape == (2, 254, 5)


@pytest.mark.parametrize("params", [(1.0, 0.0, 1e-3, 0.0), (1.0, 0.3, 0.4, 0.2)])
def testShadowBoundaryMirrorBranchesAreDistinct(
    params: tuple[float, float, float, float],
) -> None:
    fM, fQ, fA, fL = params
    fN = 12
    arr = shadowBoundary(fM, fQ, fA, fL, fN)
    # Shared endpoints are exact and emitted once each.
    assert arr[0, 3] == 0.5 * np.pi
    assert arr[fN - 1, 3] == 1.5 * np.pi

    fUpper = arr[1 : fN - 1]
    fLower = arr[fN : 2 * fN - 2]
    fSinUpper = np.clip(sinPsi(fUpper[:, 4], fM, fQ, fA, fL), -1.0, 1.0)
    fSinLower = np.clip(sinPsi(fLower[:, 4], fM, fQ, fA, fL), -1.0, 1.0)

    fExpectedUpper = np.arcsin(fSinUpper)
    fExpectedUpper = np.where(
        fExpectedUpper < 0.0, fExpectedUpper + 2.0 * np.pi, fExpectedUpper
    )
    fExpectedLower = np.pi - np.arcsin(fSinLower)

    assert fUpper[:, 3] == pytest.approx(fExpectedUpper, abs=1e-9)
    assert fLower[:, 3] == pytest.approx(fExpectedLower, abs=1e-9)
    # Upper branch: cos(psi) > 0 (plane y < 0); lower: cos(psi) < 0 (plane y > 0).
    assert np.all(np.cos(fUpper[:, 3]) > 0.0)
    assert np.all(np.cos(fLower[:, 3]) < 0.0)
    assert np.all(fUpper[:, 1] < 0.0)
    assert np.all(fLower[:, 1] > 0.0)
    # The branches are mirror images sharing sin(psi), not the same angle.
    assert np.sin(fUpper[:, 3]) == pytest.approx(np.sin(fLower[::-1, 3]), abs=1e-9)
    assert np.all(np.abs(fUpper[:, 3] - fLower[::-1, 3]) > 1e-3)


def testSolvePhotonRadiiBatchRaisesWhenBracketNeverCloses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One row never reaches a sign change: the expansion cap must raise.
    def fFailsSecondRow(
        rArr: np.ndarray,
        fM: np.ndarray,
        fQ: np.ndarray,
        fA: np.ndarray,
        fL: np.ndarray,
        fSign: float,
    ) -> np.ndarray:
        fOut = -np.ones_like(np.asarray(rArr, dtype=np.float64))
        fOut[1] = 1.0
        return fOut

    monkeypatch.setattr(shadow, "photonResidual", fFailsSecondRow)
    arrParams = np.array(BATCH_SETS, dtype=np.float64)
    with pytest.raises(RuntimeError, match=r"row\(s\) \[1\]"):
        solvePhotonRadiiBatch(arrParams)


def testShadowBoundaryBatchRaisesWhenBracketNeverCloses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fAlwaysPositive(
        rArr: np.ndarray,
        fM: np.ndarray,
        fQ: np.ndarray,
        fA: np.ndarray,
        fL: np.ndarray,
        fSign: float,
    ) -> np.ndarray:
        return np.ones_like(np.asarray(rArr, dtype=np.float64))

    monkeypatch.setattr(shadow, "photonResidual", fAlwaysPositive)
    arrParams = np.array([[1.0, 0.0, 0.5, 0.0], [1.0, 0.0, -0.5, 0.0]])
    with pytest.raises(RuntimeError, match=r"row\(s\) \[0, 1\]"):
        shadowBoundaryBatch(arrParams, 8)

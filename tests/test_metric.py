import numpy as np
import pytest

from knn_shadow.metric import (
    R_O_FACTOR,
    THETA_O,
    delta,
    deltaPrime,
    maxSpin,
    outerHorizon,
    sigma,
    validateParameters,
    validateParametersBatch,
)


def testDeltaMatchesHandComputedValue() -> None:
    rArr = np.array([5.0])
    assert delta(rArr, 1.0, 0.0, 0.001, 0.0) == pytest.approx(25.0 - 10.0 + 1e-6)


def testDeltaIsVectorized() -> None:
    rArr = np.array([0.0, 2.0, 5.0])
    resArr = delta(rArr, 1.0, 0.3, 0.5, 0.2)
    assert resArr.shape == rArr.shape
    assert resArr.tolist() == pytest.approx(
        [
            0.0 + 0.25 - 0.04 + 0.09,
            4.0 - 4.0 + 0.25 - 0.04 + 0.09,
            25.0 - 10.0 + 0.30,
        ]
    )


def testDeltaPrimeMatchesHandComputedValue() -> None:
    assert deltaPrime(np.array([5.0]), 1.0) == pytest.approx(8.0)


def testDeltaPrimeIsVectorized() -> None:
    rArr = np.array([0.0, 2.0, 5.0])
    assert deltaPrime(rArr, 1.0).tolist() == pytest.approx([-2.0, 2.0, 8.0])


def testSigmaAtCosThetaOne() -> None:
    assert sigma(np.array([5.0]), 1.0, 0.001, 0.0) == pytest.approx(25.0 + 1e-6)


def testSigmaAtEquatorialObserver() -> None:
    assert sigma(np.array([5.0]), 0.0, 0.001, 0.0) == pytest.approx(25.0)


def testSigmaGeneralValue() -> None:
    assert sigma(np.array([3.0]), 0.5, 0.4, 0.2) == pytest.approx(9.0 + 0.16)


def testMaxSpinIsPositiveAndCorrect() -> None:
    assert maxSpin(1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert maxSpin(1.0, 0.3, 0.2) == pytest.approx(np.sqrt(0.95))
    assert maxSpin(1.0, 0.3, 0.2) > 0.0


def testOuterHorizonMatchesDefinition() -> None:
    for fM, fQ, fA, fL in [
        (1.0, 0.0, 0.001, 0.0),
        (1.0, 0.3, 0.5, 0.2),
        (2.0, 0.0, -1.5, 0.0),
    ]:
        fAMax = maxSpin(fM, fQ, fL)
        assert outerHorizon(fM, fQ, fA, fL) == pytest.approx(
            fM + np.sqrt(fAMax**2 - fA**2)
        )


def testOuterHorizonSchwarzschildLike() -> None:
    fHorizon = outerHorizon(1.0, 0.0, 0.001, 0.0)
    assert fHorizon == pytest.approx(1.0 + np.sqrt(1.0 - 1e-6))
    assert fHorizon > 1.0


def testOuterHorizonIsPositiveAndOutsideHorizon() -> None:
    assert outerHorizon(1.0, 0.3, 0.5, 0.2) > 0.0
    assert outerHorizon(1.0, 0.3, -0.5, 0.2) == pytest.approx(
        outerHorizon(1.0, 0.3, 0.5, 0.2)
    )


def testObserverConstants() -> None:
    assert R_O_FACTOR == 5.0
    assert THETA_O == pytest.approx(np.pi / 2.0)


@pytest.mark.parametrize(
    "fM, fQ, fA, fL",
    [
        (0.0, 0.0, 0.5, 0.0),
        (-1.0, 0.0, 0.5, 0.0),
    ],
)
def testValidateParametersRejectsNonPositiveMass(
    fM: float, fQ: float, fA: float, fL: float
) -> None:
    with pytest.raises(ValueError):
        validateParameters(fM, fQ, fA, fL)


@pytest.mark.parametrize(
    "fM, fQ, fA, fL",
    [
        (1.0, 2.0, 0.5, 0.0),
        (1.0, 1.0, 0.001, 0.0),
    ],
)
def testValidateParametersRejectsEmptySpinRange(
    fM: float, fQ: float, fA: float, fL: float
) -> None:
    with pytest.raises(ValueError):
        validateParameters(fM, fQ, fA, fL)


def testValidateParametersRejectsSpinAboveMaximum() -> None:
    with pytest.raises(ValueError):
        validateParameters(1.0, 0.0, 2.0, 0.0)


@pytest.mark.parametrize("fA", [0.0, -0.0])
def testValidateParametersRejectsZeroSpin(fA: float) -> None:
    with pytest.raises(ValueError):
        validateParameters(1.0, 0.0, fA, 0.0)


@pytest.mark.parametrize("fA", [1.0, -1.0])
def testValidateParametersRejectsExtremalSpin(fA: float) -> None:
    with pytest.raises(ValueError):
        validateParameters(1.0, 0.0, fA, 0.0)


def testValidateParametersRejectsObserverInsideHorizon() -> None:
    assert delta(np.array([R_O_FACTOR]), 1.0, 0.0, 0.001, 5.0)[0] <= 0.0
    with pytest.raises(ValueError):
        validateParameters(1.0, 0.0, 0.001, 5.0)


@pytest.mark.parametrize(
    "fM, fQ, fA, fL",
    [
        (1.0, 0.0, 0.5, 0.0),
        (1.0, 0.0, -0.5, 0.0),
        (1.0, 0.3, 0.5, 0.2),
        (1.0, 0.0, 0.999, 0.0),
    ],
)
def testValidateParametersAcceptsValidSets(
    fM: float, fQ: float, fA: float, fL: float
) -> None:
    assert validateParameters(fM, fQ, fA, fL) is None


def testValidateParametersBatchAcceptsValidBatch() -> None:
    arrParams = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [1.0, 0.0, -0.5, 0.0],
            [1.0, 0.3, 0.5, 0.2],
            [2.0, 0.0, 0.001, 0.0],
        ]
    )
    assert validateParametersBatch(arrParams) is None


def testValidateParametersBatchReportsOffendingIndex() -> None:
    arrParams = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [1.0, 2.0, 0.5, 0.0],
            [1.0, 0.0, 0.4, 0.1],
        ]
    )
    with pytest.raises(ValueError, match=r"\[1\]"):
        validateParametersBatch(arrParams)


def testValidateParametersBatchReportsAllOffendingIndices() -> None:
    arrParams = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.4, 0.1],
            [0.0, 0.0, 0.5, 0.0],
        ]
    )
    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        validateParametersBatch(arrParams)

#!/usr/bin/env python3
#
# Photon region, shadow boundary, stereographic projection

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from knn_shadow.metric import (
    R_O_FACTOR,
    delta,
    deltaPrime,
    fCosThO,
    fSinThO,
    maxSpin,
    outerHorizon,
    sigma,
    validateParameters,
    validateParametersBatch,
)

# Metric parameters arrive as scalars (single set) or (B, 1) columns (batch).
ScalarOrArray = float | NDArray[np.float64]


# Shadow parametrization
def sinTheta(
    r: NDArray[np.float64] | float,
    m: ScalarOrArray,
    Q: ScalarOrArray,
    a: ScalarOrArray,
    l: ScalarOrArray, # noqa: E741
) -> NDArray[np.float64]:
    """Sine of the celestial polar angle, raw value (no clamping)."""
    rArr = np.asarray(r, dtype=np.float64)
    fRO = R_O_FACTOR * m
    fDeltaO = delta(fRO, m, Q, a, l)
    fDeltaR = delta(rArr, m, Q, a, l)
    fNumer = 4.0 * rArr * np.sqrt(fDeltaO * fDeltaR)
    fDenom = deltaPrime(rArr, m) * (fRO**2 - rArr**2) + 4.0 * rArr * fDeltaR
    return fNumer / fDenom

def sinPsi(
    r: NDArray[np.float64] | float,
    m: ScalarOrArray,
    Q: ScalarOrArray,
    a: ScalarOrArray,
    l: ScalarOrArray, # noqa: E741
) -> NDArray[np.float64]:
    """Sine of the celestial azimuth, raw value (no clamping)."""
    rArr = np.asarray(r, dtype=np.float64)
    fDeltaR = delta(rArr, m, Q, a, l)
    fNumer = deltaPrime(rArr, m) * (rArr**2 + (l+a*fCosThO**2)**2) - 4.0*rArr*fDeltaR
    fDenom = 4.0 * rArr * a * np.sqrt(fDeltaR) * fSinThO
    return fNumer / fDenom


def photonResidual(
    r: NDArray[np.float64] | float,
    m: ScalarOrArray,
    Q: ScalarOrArray,
    a: ScalarOrArray,
    l: ScalarOrArray, # noqa: E741
    sign: float,
) -> NDArray[np.float64]:
    """Residual of the photon-orbit equation; sign=-1 for r_p+, sign=+1 for r_p-."""
    rArr = np.asarray(r, dtype=np.float64)
    fDelta = delta(rArr, m, Q, a, l)
    fSigma = sigma(rArr, fCosThO, a, l)
    return (
        fSigma * deltaPrime(rArr, m)
        - 4.0 * rArr * fDelta
        + sign * 4.0 * a * rArr * np.sqrt(fDelta) * fSinThO
    )


def solvePhotonRadii(
    m: float,
    Q: float,
    a: float,
    l: float, # noqa: E741
) -> tuple[float, float]:
    """Solve both photon-orbit radii, returning (r_p+, r_p-) labeled by equation."""
    fRPlus = outerHorizon(m, Q, a, l)
    # Little numerical preparation for the brent algorithm
    fDelta = 1e-9 * max(1.0, fRPlus)
    fLo = fRPlus + fDelta
    fHiCap = 1e12 * max(1.0, fRPlus)
    fRoots: list[float] = []
    for fSign in (-1.0, 1.0):
        fHi = 4.0 * (maxSpin(m, Q, l) + abs(l) + m)
        # photonResdiual(fLo)>0; expand the upper bound until photonResidual(fHi)<0
        while float(photonResidual(fHi, m, Q, a, l, fSign)) >= 0.0:
            fHi *= 2.0
            if fHi > fHiCap:
                raise RuntimeError(
                    f"Photon-orbit root solve failed for (m={m}, Q={Q}, a={a}, "
                    f"l={l}): no sign change above r={fLo}."
                )
        try:
            fRoot = float(
                optimize.brentq(
                    lambda fR, fS=fSign: float(photonResidual(fR, m, Q, a, l, fS)),
                    fLo,
                    fHi,
                    xtol=1e-14,
                    rtol=1e-15,
                ) # find root (in the variable fR) in the interval [fLo, fHi]
            )
        except ValueError as fErr:
            raise RuntimeError(
                f"Photon-orbit root solve failed for (m={m}, Q={Q}, a={a}, "
                f"l={l}): {fErr}"
            ) from fErr
        if (
            not np.isfinite(fRoot)
            or fRoot <= 0.0
            or float(delta(fRoot, m, Q, a, l)) < 0.0
        ):
            raise RuntimeError(
                f"Photon-orbit root solve failed for (m={m}, Q={Q}, a={a}, "
                f"l={l}): inadmissible root r={fRoot}."
            )
        fRoots.append(fRoot)
    return fRoots[0], fRoots[1]


def solvePhotonRadiiBatch(
    arrParams: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve both photon-orbit radii for every (m, Q, a, l) row by bisection."""
    validateParametersBatch(arrParams)
    arrP = np.asarray(arrParams, dtype=np.float64)
    mArr = arrP[:, 0]
    QArr = arrP[:, 1]
    aArr = arrP[:, 2]
    lArr = arrP[:, 3]

    # maxSpin/outerHorizon coerce to float, so the row-wise horizon is built here.
    fAMaxSqArr = mArr**2 + lArr**2 - QArr**2
    rPlusArr = mArr + np.sqrt(fAMaxSqArr - aArr**2)
    fLoArr = rPlusArr + 1e-9 * np.maximum(1.0, rPlusArr)
    fHiArr = 4.0 * (np.sqrt(fAMaxSqArr) + np.abs(lArr) + mArr)

    fRoots: list[NDArray[np.float64]] = []
    for fSign in (-1.0, 1.0):
        fHiCur = fHiArr.copy()
        # Expand the upper bound until photonResidual(fHi) < 0 for every row.
        for _ in range(8):
            fHiVal = photonResidual(fHiCur, mArr, QArr, aArr, lArr, fSign)
            if np.all(fHiVal < 0.0):
                break
            fHiCur = np.where(fHiVal >= 0.0, 2.0 * fHiCur, fHiCur)
        fHiVal = photonResidual(fHiCur, mArr, QArr, aArr, lArr, fSign)
        fBadHi = fHiVal >= 0.0
        if np.any(fBadHi):
            raise RuntimeError(
                "Photon-orbit root solve failed for row(s) "
                f"{np.where(fBadHi)[0].tolist()}: no sign change above the "
                "expanded upper bound."
            )
        fLoCur = fLoArr.copy()
        for _ in range(60):
            fMid = 0.5 * (fLoCur + fHiCur)
            fMidVal = photonResidual(fMid, mArr, QArr, aArr, lArr, fSign)
            fLoCur = np.where(fMidVal > 0.0, fMid, fLoCur)
            fHiCur = np.where(fMidVal <= 0.0, fMid, fHiCur)
        fRoot = 0.5 * (fLoCur + fHiCur)
        fBad = ~(
            np.isfinite(fRoot)
            & (fRoot > 0.0)
            & (delta(fRoot, mArr, QArr, aArr, lArr) >= 1.0)
        )
        if np.any(fBad):
            raise RuntimeError(
                "Photon-orbit batch root solve failed for row(s) "
                f"{np.where(fBad)[0].tolist()}."
            )
        fRoots.append(fRoot)
    return fRoots[0], fRoots[1]


def stereographicProjection(
    th: NDArray[np.float64],
    psi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map sky angles to plane coordinates via the stereographic projection."""
    thArr = np.asarray(th, dtype=np.float64)
    psiArr = np.asarray(psi, dtype=np.float64)
    fScale = 2.0 * np.tan(thArr / 2.0)
    return -fScale * np.sin(psiArr), -fScale * np.cos(psiArr)


def shadowBoundary(
    m: float,
    Q: float,
    a: float,
    l: float,  # noqa: E741
    nR: int = 128,
) -> NDArray[np.float64]:
    """Sample the shadow boundary as (2*nR-2, 5) rows (x, y, theta, psi, r_p)."""
    validateParameters(m, Q, a, l)
    if nR < 3:
        raise ValueError(f"nR must be at least 3, got nR={nR}.")
    fRPlus, fRMinus = solvePhotonRadii(m, Q, a, l)

    rArr = np.linspace(fRPlus, fRMinus, nR)
    sinTh = np.clip(sinTheta(rArr, m, Q, a, l), -1.0, 1.0)
    sinPs = np.clip(sinPsi(rArr, m, Q, a, l), -1.0, 1.0)
    thArr = np.arcsin(sinTh)

    # Upper branch: cos(psi) >= 0; lower branch mirrors it across the x-axis.
    psiUpper = np.arcsin(sinPs)
    psiUpper = np.where(psiUpper < 0.0, psiUpper + 2.0 * np.pi, psiUpper)
    psiLower = np.pi - np.arcsin(sinPs)

    # Create indices for upper and lower branches
    idxUpper = np.arange(0, nR - 1)
    idxLower = np.arange(nR - 2, 0, -1)

    # Both shared endpoints are exact: psi = pi/2 at r_p+ and 3pi/2 at r_p-.
    psiSeqUpper = psiUpper[idxUpper]
    psiSeqUpper[0] = 0.5 * np.pi

    rSeq = np.concatenate([rArr[idxUpper], rArr[[nR - 1]], rArr[idxLower]])
    thSeq = np.concatenate([thArr[idxUpper], thArr[[nR - 1]], thArr[idxLower]])
    psiSeq = np.concatenate([psiSeqUpper, [1.5 * np.pi], psiLower[idxLower]])

    fX, fY = stereographicProjection(thSeq, psiSeq)
    return np.stack([fX, fY, thSeq, psiSeq, rSeq], axis=-1)


def shadowBoundaryBatch(
    arrParams: NDArray[np.float64],
    nR: int = 128,
) -> NDArray[np.float64]:
    """Sample every parameter row as (B, 2*nR-2, 5) rows (x, y, theta, psi, r_p)."""
    validateParametersBatch(arrParams)
    if nR < 3:
        raise ValueError(f"nR must be at least 3, got nR={nR}.")
    arrP = np.asarray(arrParams, dtype=np.float64)
    fRPlus, fRMinus = solvePhotonRadiiBatch(arrP)

    tArr = np.linspace(0.0, 1.0, nR)
    rArr = fRPlus[:, None] + (fRMinus - fRPlus)[:, None] * tArr[None, :]

    mCol = arrP[:, 0:1]
    QCol = arrP[:, 1:2]
    aCol = arrP[:, 2:3]
    lCol = arrP[:, 3:4]
    sinTh = np.clip(sinTheta(rArr, mCol, QCol, aCol, lCol), -1.0, 1.0)
    sinPs = np.clip(sinPsi(rArr, mCol, QCol, aCol, lCol), -1.0, 1.0)
    thArr = np.arcsin(sinTh)

    # Upper branch: cos(psi) >= 0; lower branch mirrors it across the x-axis.
    psiUpper = np.arcsin(sinPs)
    psiUpper = np.where(psiUpper < 0.0, psiUpper + 2.0 * np.pi, psiUpper)
    psiLower = np.pi - np.arcsin(sinPs)

    idxUpper = np.arange(0, nR - 1)
    idxLower = np.arange(nR - 2, 0, -1)

    # Both shared endpoints are exact: psi = pi/2 at r_p+ and 3pi/2 at r_p-.
    psiSeqUpper = psiUpper[:, idxUpper]
    psiSeqUpper[:, 0] = 0.5 * np.pi

    rSeq = np.concatenate(
        [rArr[:, idxUpper], rArr[:, [nR - 1]], rArr[:, idxLower]], axis=1
    )
    thSeq = np.concatenate(
        [thArr[:, idxUpper], thArr[:, [nR - 1]], thArr[:, idxLower]], axis=1
    )
    psiSeq = np.concatenate(
        [
            psiSeqUpper,
            np.full((arrP.shape[0], 1), 1.5 * np.pi),
            psiLower[:, idxLower],
        ],
        axis=1,
    )

    fX, fY = stereographicProjection(thSeq, psiSeq)
    return np.stack([fX, fY, thSeq, psiSeq, rSeq], axis=-1)

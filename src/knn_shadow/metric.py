#!/usr/bin/env python3
#
# Definition of variables and functions that define the metrix and spacetime:
# parameters, validity conditions (horizon, ranges)

import numpy as np
from numpy.typing import NDArray

# Observer fixed on the equatorial plane at r_O = 5 * m.
R_O_FACTOR: float = 5.0
THETA_O: float = np.pi / 2.0

fSinThO: float = np.sin(THETA_O)
#fCosThO = np.cos(THETA_O) gives a tiny error
fCosThO: float = 0.0


def delta(
    r: NDArray[np.float64],
    m: float,
    Q: float,
    a: float,
    l: float, # noqa: E741
) -> NDArray[np.float64]:
    """Delta function: r**2 - 2*m*r + a**2 - l**2 + Q**2."""
    rArr = np.asarray(r, dtype=np.float64)
    return rArr**2 - 2.0 * m * rArr + a**2 - l**2 + Q**2


def deltaPrime(r: NDArray[np.float64], m: float) -> NDArray[np.float64]:
    """Radial derivative of delta: 2*r - 2*m."""
    rArr = np.asarray(r, dtype=np.float64)
    return 2.0 * rArr - 2.0 * m


def sigma(
    r: NDArray[np.float64],
    cosTheta: float,
    a: float,
    l: float, # noqa: E741
) -> NDArray[np.float64]:
    """Frame factor: r**2 + (l + a*cos(theta))**2."""
    rArr = np.asarray(r, dtype=np.float64)
    return rArr**2 + (l + a * cosTheta) ** 2


def maxSpin(m: float, Q: float, l: float) -> float: # noqa: E741
    """Largest admissible spin: sqrt(m**2 + l**2 - Q**2)."""
    return float(np.sqrt(m**2 + l**2 - Q**2))


def outerHorizon(m: float, Q: float, a: float, l: float) -> float: # noqa: E741
    """Outer horizon radius: m + sqrt(a_max**2 - a**2)."""
    fAMax = maxSpin(m, Q, l)
    return float(m + np.sqrt(fAMax**2 - a**2))


def validateParameters(m: float, Q: float, a: float, l: float) -> None: # noqa: E741
    """Check one parameter set, raising ValueError when it is invalid."""
    if m <= 0.0:
        raise ValueError(f"Mass must be positive, got m={m}.")
    fAMaxSq = m**2 + l**2 - Q**2 # largest spin squared
    if fAMaxSq <= 0.0:
        raise ValueError(f"No valid spin: m^2 + l^2 - Q^2 = {fAMaxSq} <= 0.")
    fASq = a**2
    if fASq > fAMaxSq:
        raise ValueError(f"Spin out of range: a^2 = {fASq} > a_max^2 = {fAMaxSq}.")
    if a == 0.0:
        raise ValueError("Spin a = 0 is numerically singular.")
    fAMax = maxSpin(m, Q, l)
    if abs(a) == fAMax:
        raise ValueError(
            f"Extremal spin |a| = a_max = {fAMax} is numerically singular."
        )
    fRO = R_O_FACTOR * m
    if float(delta(np.array([fRO]), m, Q, a, l)[0]) <= 0.0:
        raise ValueError(
            f"Observer at r_O = {fRO} is not outside the horizon: delta(r_O) <= 0."
        )


def validateParametersBatch(arrParams: NDArray[np.float64]) -> None:
    """Check a (B, 4) array of (m, Q, a, l) sets, naming invalid rows."""
    arrP = np.asarray(arrParams, dtype=np.float64)
    if arrP.ndim != 2 or arrP.shape[1] != 4:
        raise ValueError(f"arrParams must have shape (B, 4), got {arrP.shape}.")
    mArr = arrP[:, 0]
    QArr = arrP[:, 1]
    aArr = arrP[:, 2]
    lArr = arrP[:, 3]
    fAMaxSqArr = mArr**2 + lArr**2 - QArr**2
    fAMaxArr = np.sqrt(np.maximum(fAMaxSqArr, 0.0))
    rOArr = R_O_FACTOR * mArr
    deltaOArr = delta(rOArr, mArr, QArr, aArr, lArr)  # delta(r_O) per row
    validArr = (
        (mArr > 0.0)
        & (fAMaxSqArr > 0.0)
        & (aArr**2 <= fAMaxSqArr)
        & (aArr != 0.0)
        & (np.abs(aArr) != fAMaxArr)
        & (deltaOArr > 0.0)
    )
    idxBad = np.where(~validArr)[0]
    if idxBad.size > 0:
        raise ValueError(f"Invalid parameter set(s) at row(s) {idxBad.tolist()}.")

# Physical framework

## Set up
An observer sits fix at a distance $r_O$ of a Kerr-Newmann-NUT black hole (described by the five parameters below), without any objects inbetween.
It is assumed that light comes from the infinity from all directions, so the hole sky seen by the observer (called *celestial sphere*) is white, except the black spot caused by presence of the black hole.
The contour of that black region is a one-dimensional line and is called *shadow* of the black hole. It depends on the five parameters and the equations that describe the shadow in the celestial sphere of the observer are given explicitly below.

## Parameters

| Variable name | Physical meaning | Comment | 
|---|---|----|
| $m$ | mass | |
| $Q$ | electric charge | |
| $a$ | spin | $a^2<=m^2 + l^2 - Q^2$ |
| $l$ | NUT-charge |  |
| $C$ | Manko-Ruiz parameter| Plays a physical role only if $l \neq 0$, it does not appear on the parametrization of the shadow, so it is not important for our purposes. |

Due to the characteristic of the Manko-Ruiz parameter, we assume for training the model that the black hole depends only on four parameters.

## Conventions
For simplicity the gravitational constants and speed of light are set to unity, $G = 1 = c$. This has a consequence that the many variables are measured as units of the mass $m$.
For example $a = 2m$, $Q = m/3$, ...

As first run, the observer is in the fixed location $r_O = 5m$, $\vartheta_0 = \pi/2$. (Where $\vartheta_0$ is the angle between the observer and the rotational axis of the blackhole, so the observer lies on the equatorial plane of the blackhole). In the future, a model will be trained to be able to deduce the observers position from the shadow. There is also the possibility to set the observer in motion, and its direction and velocity change the shadow.

## Help functions
To shorten notation and improve visibility we introduce the following functions for the metric.
$$
\Sigma(r,\vartheta) = r^2 + (l + a~\mathrm{cos}~\vartheta)^2 \\
\chi(\vartheta) = a~\mathrm{sin}^2\vartheta - 2l(\mathrm{cos}~\vartheta+C) \\
\Delta(r) = r^2 - 2mr + a^2 - l^2 + Q^2 \\
\Delta'(r) = 2r - 2m
$$

where $(t,r,\vartheta,\varphi)$ are the coordinates of the metric.

## Metric tensor
The chosen metric signature is $(-1,1,1,1)$ and for the spherical coordinates the metric of the Kerr-Newman-NUT space time is
$$
 g = -\frac{1}{\Sigma}\Big(\Delta - a^2 \mathrm{sin}^2\vartheta \Big)\mathrm{d}t^2 + \frac{2}{\Sigma}\Big(\Delta \chi - a\big(\Sigma + a \chi\big) \mathrm{sin}^2\vartheta\Big)\mathrm{d}t\mathrm{d}\varphi + \frac{1}{\Sigma}\Big(\big(\Sigma + a \chi\big)^2\mathrm{sin}^2 \vartheta - \Delta \chi^2\Big)\mathrm{d}\varphi^2 + \Sigma \Big(\frac{1}{\Delta}\mathrm{d}r^2 + \mathrm{d}\vartheta^2\Big)
$$

## The parametrization of the shadow
From the null geodesics and a long calculation the parametrization $(\theta(r_p), \psi(r_p))$ of the shadow in the celestial sphere of an observer located at $(r_O, \vartheta_O)$ is found to be
$$
\mathrm{sin}~\theta(r_p) = \frac{4 r_p \sqrt{\Delta(r_O) \Delta(r_p)}}{\Delta '(r_p) \big(r_O^2 - r_p^2\big) + 4r_p \Delta(r_p)} \\
 ~ \\
\mathrm{sin}~\psi(r_p) = \frac{\Delta '(r_p) \big(r_p^2 + (l + a~\mathrm{cos}^2\vartheta_O)^2\big) - 4 r_p \Delta(r_p)}{4 r_p a \sqrt{\Delta(r_p)} ~\mathrm{sin}~\vartheta_O}
$$

where the parameter $r_p$ runs inside the interval $[r_{p_+}, r_{p_-}]$, where $r_{p_\pm}$ are the solutions respectively of these equations:
$$
\Sigma(r_{p_\pm},\vartheta_O) \Delta ' (r_{p_\pm}) - 4 r_{p_\pm} \Delta(r_{p_\pm}) \mp 4ar_{p_\pm} \sqrt{\Delta(r_{p_\pm})} ~\mathrm{sin}~\vartheta_O = 0
$$

## Stereographic projection
The celestial sphere of the observer is a two dimensional manifold that is described by two angle parameters $\theta\in[0,\pi)$ and $\psi\in[0,2\pi)$ (the coordinate singularities are never reached), so the shadow is a one dimensional curve on it.

To represent the celestial sphere properly on the plane, the used solution is the stereographic projection, which maps a point $(\theta, \psi)$ on the sphere into the point $(x, y)$ on the plane following the transformation rule
$$
\begin{pmatrix}
        x\\
        y
    \end{pmatrix}
    = -2~\mathrm{tan}~\frac{\theta}{2}
    \begin{pmatrix}
        \mathrm{sin}~\psi\\
        \mathrm{cos}~\psi
    \end{pmatrix}
$$

## Clarification
To avoid future confusion it is important to clarify the angle variables. Although it should be clear from the text above, it is helpful to clarify it explicitly

The spacetime coordinates are $(t,r,\vartheta,\varphi)$, so for the angles we have the ranges $\vartheta\in[0,\pi)$ and $\varphi\in[0,2\pi)$. Since the position of the observer is given in the spacetime coordinates we write $r_O = 5m$ and $\vartheta_O = \pi / 2$.

The celestial sphere of the observer (what he sees in the sky) is a spherical two-dimensional manifold and is described by two angles $(\theta, \psi)$, with $\vartheta\in[0,\pi)$ and $\psi\in[0,2\pi)$. As the shadow depends on the observers and what he sees, its parametrization is given on this manifold, i.e.: on this two angles, which later can be projected to the plane.

There cannot be any confusion between $\vartheta$ and $\theta$, which would ruin everything.

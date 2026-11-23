# Machine Learning applied to Black Hole's shadows

A Python project for inferring black-hole parameters from simulated observations of a black-hole shadow. It brings together a fast simulation model, a classical Bayesian inference baseline, and a neural-network estimator, with the goal of comparing their accuracy, speed, and failure modes.

The project is at an early stage: the repository structure, environment, and automated checks are in place, and the core modules are under development.

## Plan

- Define the parameters and assumptions, and document the model.
- Implement a fast, vectorized simulation of shadow observations.
- Validate the simulation with tests and independent checks.
- Generate reproducible synthetic datasets with configurable noise.
- Build the Bayesian inference baseline.
- Train a neural-network estimator for fast inference on simulated data.
- Compare both methods across noise levels and study parameter degeneracies.

Possible later extensions include additional observer parameters, richer image models, and alternative inference methods.

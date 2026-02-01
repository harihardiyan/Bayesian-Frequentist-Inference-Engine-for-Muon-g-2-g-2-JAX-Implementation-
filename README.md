# Bayesian & Frequentist Inference Engine for Muon $g-2$ (JAX Implementation)
<div align="center">

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![JAX: Accelerated](https://img.shields.io/badge/Backend-JAX-orange.svg)
![Field: High Energy Physics](https://img.shields.io/badge/Field-High--Energy%20Physics-red.svg)
![MCMC: BlackJAX](https://img.shields.io/badge/MCMC-BlackJAX-purple.svg)
![Status: Research Grade](https://img.shields.io/badge/Status-Research--Grade-blueviolet.svg)
![Code Style: Monolithic](https://img.shields.io/badge/Code%20Style-Monolithic-brightgreen.svg)

</div>

---
**Author:** Hari Hardiyan (with Microsoft Copilot)  
**Contact:** lorozloraz@gmail.com  
**Date:** February 2, 2026  
**License:** MIT  

---

## Overview

This repository provides a monolithic, research-grade statistical engine (`g2_engine.py`) implementing a complete inference workflow for a two-observable model inspired by the muon $g-2$ anomaly. Built entirely using the **JAX** framework, the engine enables high-performance computation of both frequentist and Bayesian statistics, with optional **BlackJAX** support for Markov Chain Monte Carlo (MCMC) validation.

The framework is designed around four core pillars:
*   **Reproducibility**: Deterministic grid integration and explicit numerical normalization checks.
*   **Auditability**: A transparent, single-file architecture with no hidden abstractions or black-box dependencies.
*   **Physical Interpretability**: A two-observable structure specifically engineered to break the intrinsic $\mu$–$\kappa$ degeneracy.
*   **Methodological Completeness**: Integration of profile likelihoods, Bayesian evidence, Bayes factors, Wilks theorem calibration, and Brazilian bands in a unified environment.

---

## Scientific Context

In studies of the muon $g-2$ anomaly, the deviation between experimental measurement and the Standard Model prediction is typically parameterized as:

$$\Delta a_\mu^{\text{exp}} = \mu + \kappa$$

Where:
*   **$\mu$**: Represents a potential **New Physics (NP)** contribution.
*   **$\kappa$**: Represents the **Hadronic Vacuum Polarization (HVP)** shift or theoretical correction.

A single measurement of $\Delta a_\mu$ is inherently degenerate. This engine resolves the degeneracy by introducing a second observable (e.g., a direct HVP constraint from Lattice QCD or dispersive R-ratio analyses), enabling robust joint inference on both parameters.

---

## Key Features

### 1. Two-Observable Likelihood
Implements a joint Gaussian likelihood function that breaks parameter degeneracy:
*   $\Delta a_\mu \sim \mathcal{N}(\mu + \kappa, \sigma_\mu^2)$
*   $\Delta \text{HVP} \sim \mathcal{N}(\kappa, \sigma_{\text{HVP}}^2)$

### 2. Multi-Model Prior Framework
Supports explicitly normalized truncated priors for $\kappa$ to allow for rigorous theoretical sensitivity testing:
*   Truncated Gaussian Mixture
*   Truncated Uniform
*   Truncated Single Gaussian

### 3. Frequentist Profile Likelihood
Performs a full 2D grid search over $(\mu, \kappa)$, profiling over the nuisance parameter $\kappa$ to compute the test statistic:
$$q(\mu) = -2 \Delta \log L$$

### 4. Bayesian Marginalization
Executes exact numerical integration over $\kappa$ to yield the marginal posterior $p(\mu \mid \text{data})$. The engine verifies numerical stability such that:
$$\int p(\mu) \, d\mu = 1.000000 \pm \epsilon$$

### 5. Bayesian Evidence & Model Selection
Computes the marginal likelihood (evidence) for competing hypotheses:
*   **$Z_{\text{NP}}$**: Evidence for the model with $\mu$ as a free parameter.
*   **$Z_{\text{SM}}$**: Evidence for the Standard Model only ($\mu \equiv 0$).
*   **Bayes Factor ($B_{\text{NP,SM}}$)**: Provides a quantitative measure for model selection.

### 6. Wilks Theorem Calibration
Uses Monte Carlo pseudo-experiments to generate the empirical distribution of $q(\mu_{\text{true}})$, allowing for a direct comparison with the asymptotic $\chi^2_1$ distribution and extraction of empirical confidence thresholds.

### 7. Brazilian Band Construction
Aggregates $q(\mu)$ curves across multiple toy experiments to produce:
*   Median expected sensitivity.
*   $\pm 1\sigma$ (Green) and $\pm 2\sigma$ (Yellow) expected bands.

### 8. MCMC Cross-Validation
Includes optional **BlackJAX NUTS** sampling to provide a stochastic cross-check of the posterior shape, grid accuracy, and parameter convergence.

---

## Technical Architecture

### File Structure
The repository intentionally utilizes a monolithic design for portability and ease of peer review:
*   `g2_engine.py`: Contains all inference logic, priors, likelihoods, evidence calculations, toys, and plotting routines.
*   `README.md`: Documentation.

### Dependencies
*   Python 3.10+
*   JAX & JAXlib
*   NumPy
*   Matplotlib
*   BlackJAX (Optional, for MCMC validation)

### Installation
```bash
pip install jax jaxlib matplotlib blackjax
```

---

## Execution and Outputs

To run the full inference pipeline, execute:
```bash
python g2_engine.py
```

The engine will sequentially produce:
1.  **Point Estimates**: $\hat{\mu}$ (Profile), $\hat{\mu}$ (Bayesian), and $\hat{\mu}$ (MCMC).
2.  **Model Comparison**: $Z_{\text{NP}}$, $Z_{\text{SM}}$, and the resulting Bayes Factor.
3.  **Stability Checks**: Numerical normalization results for both priors and posteriors.
4.  **Statistical Calibration**: Empirical vs. Asymptotic Wilks thresholds.
5.  **Visualizations**: 2D posterior contours, Brazilian bands, and MCMC diagnostics.

---

## Reproducibility & Auditability

This engine is optimized for:
*   **Methodological HEP Research**: Transparent handling of nuisance parameters.
*   **Peer-Reviewed Publications**: Reproducible results via deterministic grid integration.
*   **Teaching**: Clear implementation of advanced statistical concepts in High-Energy Physics.

---

## Citation & License

**License:** MIT License  

If you utilize this engine in academic work, please cite this repository and acknowledge the use of the JAX and BlackJAX libraries.

**Contact:** [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)
```

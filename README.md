# Risk-Aware Configuration Selection Under Demand Uncertainty

## Overview

This project studies a **two-stage optimization framework** for selecting the best system configuration under uncertain demand.  
The uncertainty in demand is explicitly modeled, and **risk-sensitive criteria** such as **Value-at-Risk (VaR)** and **Conditional Value-at-Risk (CVaR)** are used to compare and rank candidate configurations.

The workflow separates:
1. **Structural decisions** (configurations)
2. **Operational decisions** (recourse actions under demand)
3. **Risk-aware selection** of the final configuration

---

## Problem Setting

- Demand is modeled as a **random variable** \( W \)
- Each **configuration** \( y \) induces a random cost:
  
\[
Z(y, W)
\]

- The objective is **not only** to minimize expected cost, but also to control **tail risk**



---

## Two-Stage Optimization Pipeline

### Stage 1: Configuration Optimization

- Solve a **deterministic or stochastic optimization**
- Output:
  - A set of **feasible, Pareto-optimal configurations** \( \{y_1, y_2, \dots, y_K\} \)

This stage captures **capacity and fixed-cost tradeoffs**.

---

### Stage 2: Recourse Optimization

For each configuration \( y \) and each demand realization \( w_i \):

\[
\xi^*(y, w_i) = \arg\min_\xi Z(y, \xi, w_i)
\]

- Produces a **cost sample**:
\[
\{ Z(y, w_1), Z(y, w_2), \dots, Z(y, w_N) \}
\]

---

## Demand Modeling

- Demand is assumed to follow a **known parametric distribution** (e.g., Normal):
\[
W \sim \mathcal{N}(\mu, \sigma^2)
\]

- Samples are drawn once and reused consistently across all configurations.
- The **distribution itself** (not just empirical frequencies) is used when computing quantiles.

---

## Risk Measures

### Value-at-Risk (VaR)

For confidence level \( \alpha \):

\[
\mathrm{VaR}_\alpha(Z) = \inf \{ z \in \mathbb{R} : P(Z \le z) \ge \alpha \}
\]

- VaR is **configuration-specific**
- Each \( y \) induces a different cost distribution \( Z(y, W) \)

---

### Conditional Value-at-Risk (CVaR)

\[
\mathrm{CVaR}_\alpha(Z)
= \mathbb{E}[ Z \mid Z \ge \mathrm{VaR}_\alpha(Z) ]
\]

- Measures **expected tail loss**
- More stable and informative than VaR alone

---

## Why Mean and Standard Deviation Are Computed

- Needed to **parameterize the cost distribution**
- Required when assuming:
\[
Z(y, W) \sim \mathcal{N}(\mu_y, \sigma_y^2)
\]
- Enables **analytical quantile computation** instead of purely empirical estimates

---

## Configuration Selection Criteria

Final configuration \( y^* \) is selected by:

- **Primary:** Minimize \( \mathrm{CVaR}_{0.9}(Z(y)) \)
- **Optional tie-breakers:**
  - Lower expected cost
  - Lower variance
  - Lower fixed cost

---

## Key Design Choices

- Uses **same demand distribution** throughout (no re-estimation noise)
- Separates **optimization** from **risk evaluation**
- Supports both **VaR-only** and **CVaR-based** selection

---

## How to Run

```bash
python main.py

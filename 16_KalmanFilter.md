# 16 — Kalman Filter

> **Keywords:** linear Gaussian state space model, predict-update cycle, Kalman gain, innovation, covariance propagation, RTS smoother, Kalman EM, online Bayesian regression

---

## 1. Motivation

The Kalman filter solves the problem of **estimating a continuously evolving hidden state from noisy measurements** in real time. The hidden state and measurements are related by a **linear Gaussian** model; the Kalman filter is the optimal (minimum mean-squared error) Bayesian filter for this class.

Its key property: the filtering posterior at every time step is Gaussian, so the entire probability distribution is propagated using only a mean vector and a covariance matrix.

---

## 2. Historical Notes

| Year | Event |
|------|-------|
| 1940s | Wiener develops the Wiener filter for continuous-time linear estimation (requires full signal history). |
| 1960 | **Kalman** introduces the recursive online filter in "A new approach to linear filtering and prediction problems." |
| 1961 | **Kalman & Bucy** extend to continuous-time systems. |
| 1965 | **Rauch, Tung & Striebel** derive the RTS backward smoother. |
| 1970s | Kalman filter adopted for NASA Apollo navigation; becomes central to aerospace. |
| 1994 | **Julier & Uhlmann** introduce the Unscented Kalman Filter (UKF) for nonlinear systems. |

---

## 3. Model Definition

**State transition:**

$$z_t = Az_{t-1} + B + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, Q)$$

**Observation model:**

$$x_t = Cz_t + D + \delta_t, \qquad \delta_t \sim \mathcal{N}(0, R)$$

**Initial prior:** $z_1 \sim \mathcal{N}(\mu_1, \Sigma_1)$.

| Symbol | Dimension | Role |
|--------|-----------|------|
| $z_t$ | $d_z$ | Hidden state |
| $x_t$ | $d_x$ | Observation |
| $A$ | $d_z \times d_z$ | State transition matrix |
| $B$ | $d_z$ | Bias / control input |
| $C$ | $d_x \times d_z$ | Observation matrix |
| $D$ | $d_x$ | Observation bias |
| $Q$ | $d_z \times d_z$ | Process noise covariance |
| $R$ | $d_x \times d_x$ | Measurement noise covariance |

Conditional distributions:

$$P(z_t \mid z_{t-1}) = \mathcal{N}(Az_{t-1}+B,\; Q), \qquad P(x_t \mid z_t) = \mathcal{N}(Cz_t+D,\; R)$$

The noise terms are mutually independent and independent of the initial state. This is a **linear Gaussian** analogue of the HMM.

---

## 4. Filtering Recursion

**Goal:** compute $P(z_t \mid x_{1:t})$ recursively.

By Bayes' theorem and the conditional independence structure (identical to the HMM derivation):

$$P(z_t \mid x_{1:t}) \propto P(x_t \mid z_t) \int P(z_t \mid z_{t-1})\,P(z_{t-1} \mid x_{1:t-1})\,dz_{t-1}$$

This decomposes into two steps:

1. **Predict:** propagate the previous posterior through the dynamics (the integral above).
2. **Update:** condition on the new observation $x_t$.

Because $P(z_{t-1} \mid x_{1:t-1})$ is Gaussian and all operations are linear, the filtering posterior is **always Gaussian**. We only need to track $(\mu_t, \Sigma_t)$.

---

## 5. Predict Step

Assume $P(z_{t-1} \mid x_{1:t-1}) = \mathcal{N}(\mu_{t-1}, \Sigma_{t-1})$.

Since $z_t = Az_{t-1} + B + \varepsilon_t$ is an affine function of $z_{t-1}$ plus independent Gaussian noise:

$$\mu_{t}^{*} = A\mu_{t-1} + B$$

$$\Sigma_{t}^{*} = A\Sigma_{t-1}A^\top + Q$$

$$P(z_t \mid x_{1:t-1}) = \mathcal{N}(\mu_{t}^{*}, \Sigma_{t}^{*})$$

The predicted covariance grows: the process noise $Q$ adds uncertainty that will be reduced in the update step.

---

## 6. Update Step

The state $z_t$ and observation $x_t$ are jointly Gaussian given $x_{1:t-1}$:

$$\begin{pmatrix}z_t \\ x_t\end{pmatrix} \Bigg\vert x_{1:t-1} \sim \mathcal{N}\!\left(\begin{pmatrix}\mu_{t}^{*} \\ C\mu_{t}^{*}+D\end{pmatrix},\;\begin{pmatrix}\Sigma_{t}^{*} & \Sigma_{t}^{*}C^\top \\ C\Sigma_{t}^{*} & C\Sigma_{t}^{*}C^\top+R\end{pmatrix}\right)$$

The cross-covariance $\text{Cov}(z_t, x_t) = \Sigma_{t}^{*} C^\top$ follows from $x_t = Cz_t + D + \delta_t$ with $\delta_t$ independent of $z_t$.

Applying the **Gaussian conditional formula** ($a \mid b \sim \mathcal{N}(\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(b - \mu_b), \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})$):

**Innovation:**

$$\nu_t = x_t - (C\mu_{t}^{*} + D)$$

**Innovation covariance:**

$$S_t = C\Sigma_{t}^{*}C^\top + R$$

**Kalman gain:**

$$\boxed{K_t = \Sigma_{t}^{*} C^\top S_t^{-1} = \Sigma_{t}^{*}C^\top(C\Sigma_{t}^{*}C^\top + R)^{-1}}$$

**Updated mean and covariance:**

$$\boxed{\mu_t = \mu_{t}^{*} + K_t\nu_t}$$

$$\boxed{\Sigma_t = (I - K_tC)\Sigma_{t}^{*}}$$

**Why $(I - K_tC)\Sigma_{t}^{*}$?** From the conditional Gaussian formula: $\Sigma_t = \Sigma_{t}^{*} - \Sigma_{t}^{*}C^\top S_t^{-1}C\Sigma_{t}^{*} = \Sigma_{t}^{*} - K_tC\Sigma_{t}^{*} = (I-K_tC)\Sigma_{t}^{*}$.

**Interpreting $K_t$:**
- If $R \to 0$ (exact measurements): $K_t \to C^{-1}$ — trust the observation completely.
- If $\Sigma_{t}^{*} \to 0$ (certain prediction): $K_t \to 0$ — trust the prediction completely.
- In general: $K_t$ trades off prediction uncertainty vs. measurement uncertainty.

---

## 7. Full Algorithm

```
Initialise: μ_1, Σ_1

For t = 1, 2, ..., T:

  [t > 1] PREDICT:
    μ_t* = A μ_{t-1} + B
    Σ_t* = A Σ_{t-1} Aᵀ + Q

  UPDATE:
    ν_t = x_t − C μ_t* − D          [innovation]
    S_t = C Σ_t* Cᵀ + R              [innovation covariance]
    K_t = Σ_t* Cᵀ S_t⁻¹              [Kalman gain]
    μ_t = μ_t* + K_t ν_t             [updated mean]
    Σ_t = (I − K_t C) Σ_t*           [updated covariance]
```

**Cost per step:** $O(d_z^2 d_x + d_x^3)$ dominated by $S_t^{-1}$.

---

## 8. Connection to Online Bayesian Linear Regression

Set $A=I$, $B=0$, $Q=0$ (state does not evolve), $C=x_t^\top$ (observation matrix is the feature vector), $D=0$, $R=\sigma^2$ (scalar). Then the model is:

$$z = w \text{ (fixed weights)}, \qquad x_t = w^\top \phi_t + \delta_t$$

Running the Kalman filter gives the **Bayesian linear regression posterior** updated one observation at a time. This shows that Bayesian linear regression is a static special case of the Kalman filter, and that the Kalman filter is online Bayesian inference for linear Gaussian dynamical systems.

---

## 9. Rauch-Tung-Striebel (RTS) Smoother

The Kalman filter computes **filtered** posteriors $P(z_t \mid x_{1:t})$ using only past observations. The **RTS smoother** computes **smoothed** posteriors $P(z_t \mid x_{1:T})$ using the full sequence.

**Smoother gain:**

$$G_t = \Sigma_t A^\top (\Sigma_{t+1}^*)^{-1}$$

**Backward pass** (from $t=T-1$ down to $1$; initialise at $\mu_T^{\text{sm}}=\mu_T$, $\Sigma_T^{\text{sm}}=\Sigma_T$):

$$\mu_t^{\text{sm}} = \mu_t + G_t(\mu_{t+1}^{\text{sm}} - \mu_{t+1}^*)$$

$$\Sigma_t^{\text{sm}} = \Sigma_t + G_t(\Sigma_{t+1}^{\text{sm}} - \Sigma_{t+1}^*)G_t^\top$$

---

## 10. Filtering vs. Smoothing vs. Prediction

| Task | Estimate | Algorithm | Uses |
|------|----------|-----------|------|
| **Filtering** | $P(z_t \mid x_{1:t})$ | Kalman forward pass | Past + present |
| **Smoothing** | $P(z_t \mid x_{1:T})$ | Kalman + RTS backward | Full sequence |
| **Prediction ($k$ steps)** | $P(z_{t+k} \mid x_{1:t})$ | Apply predict step $k$ times | Past only |

---

## 11. Extensions

| Model | Setting | Method |
|-------|---------|--------|
| **EKF** | Mild nonlinearity | Linearise $f$, $h$ via first-order Taylor |
| **UKF** | Moderate nonlinearity | Sigma-point propagation |
| **Particle Filter** | Arbitrary nonlinearity and noise | Sequential Monte Carlo (DL_09) |
| **Kalman EM** | Unknown $A, C, Q, R$ | EM with Kalman smoother as E-step |
| **Ensemble KF** | Very high-dimensional $z$ | Monte Carlo covariance approximation |

---

## 12. Summary

```
Model:
  z_t = A z_{t-1} + B + ε_t,   ε_t ~ N(0,Q)
  x_t = C z_t + D + δ_t,       δ_t ~ N(0,R)

Predict:
  μ_t* = A μ_{t-1} + B
  Σ_t* = A Σ_{t-1} Aᵀ + Q

Update:
  K_t  = Σ_t* Cᵀ (C Σ_t* Cᵀ + R)⁻¹    [Kalman gain]
  μ_t  = μ_t* + K_t (x_t − C μ_t* − D)  [innovation-corrected mean]
  Σ_t  = (I − K_t C) Σ_t*                [reduced covariance]

RTS smoother: backward pass gives P(z_t | x_{1:T})
Static limit (A=I, Q=0): reduces to online Bayesian linear regression
```

---

## 13. References

1. **Kalman, R. E.** (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
2. **Kalman, R. E. & Bucy, R. S.** (1961). New results in linear filtering and prediction theory. *Journal of Basic Engineering*, 83(1), 95–108.
3. **Rauch, H. E., Tung, F., & Striebel, C. T.** (1965). Maximum likelihood estimates of linear dynamic systems. *AIAA Journal*, 3(8), 1445–1450.
4. **Sarkka, S.** (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press. [Modern Bayesian treatment; connections to GP and regression.]
5. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §13.3 (Linear dynamical systems).
6. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Ch. 18 (State space models).
7. **Julier, S. J. & Uhlmann, J. K.** (1997). A new extension of the Kalman filter to nonlinear systems. *Proceedings of SPIE*, 3068, 182–193.

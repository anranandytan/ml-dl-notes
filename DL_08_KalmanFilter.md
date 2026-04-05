# Kalman Filter

> **Keywords:** linear Gaussian state space model, Kalman gain, predict-update cycle, innovation, Rauch-Tung-Striebel smoother, Bayesian filtering

---

## 1. Motivation & Intuition

Many physical systems evolve over time in a way that is **partially observable**: we receive noisy measurements of a hidden state, and we wish to estimate that state as accurately as possible.

**Examples:**
- Tracking an aircraft from radar returns: the true position/velocity is hidden; measurements are noisy radar blips.
- Estimating a robot's location from wheel odometry and GPS: both sources are noisy and complementary.
- Filtering a financial signal: the underlying trend (hidden) is observed through noisy price data.

The **Kalman Filter** (1960) is the optimal Bayesian filter for **linear** systems with **Gaussian** noise. "Optimal" means it produces the minimum mean-squared-error estimate of the hidden state given all observations so far. Its key insight is that the posterior over the state is always Gaussian — so the entire probability distribution can be propagated using just a mean vector and a covariance matrix.

**The predict-update cycle:**
1. **Predict:** Roll the current state estimate forward in time using the system dynamics.
2. **Update:** Incorporate the new observation, correcting the prediction toward the measurement.

This two-step cycle repeats at every time step and runs in $O(d_z^3 + d_x d_z^2)$ per step (where $d_z$ is the state dimension and $d_x$ the observation dimension), making it real-time capable.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1940s | Norbert Wiener develops the Wiener filter for optimal linear estimation in continuous time; requires the full signal history. |
| 1960 | **Rudolf E. Kálmán** publishes "A new approach to linear filtering and prediction problems", introducing the recursive (online) filter we now call the Kalman filter. |
| 1961 | **Kálmán & Bucy** extend the result to continuous-time systems (Kalman-Bucy filter). |
| 1965 | **Rauch, Tung & Striebel** derive the RTS smoother — the backward-pass complement to the Kalman filter that uses the full observation sequence. |
| 1970s | Kalman filter becomes central to the Apollo spacecraft navigation system; widespread adoption in aerospace. |
| 1994 | **Julier & Uhlmann** introduce the Unscented Kalman Filter (UKF) for nonlinear systems. |
| 1997 | Particle filters emerge as a Monte Carlo alternative for non-Gaussian, nonlinear systems. |
| Present | Kalman filtering (and its extensions) remains a cornerstone of robotics, signal processing, econometrics, and GPS navigation. |

---

## 3. Model Definition: Linear Gaussian State Space

The Kalman filter models a **linear dynamical system** with Gaussian noise — also called a **Linear Gaussian Model (LGM)** or **State Space Model (SSM)**.

**State transition (system model):**
$$z_t = A z_{t-1} + B + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, Q)$$

**Observation model:**
$$x_t = C z_t + D + \delta_t, \qquad \delta_t \sim \mathcal{N}(0, R)$$

**Initial state prior:**
$$z_1 \sim \mathcal{N}(\mu_1, \Sigma_1)$$

where:

| Symbol | Dimension | Role |
|--------|-----------|------|
| $z_t \in \mathbb{R}^{d_z}$ | State | Hidden state at time $t$ |
| $x_t \in \mathbb{R}^{d_x}$ | Observation | Noisy measurement at time $t$ |
| $A \in \mathbb{R}^{d_z \times d_z}$ | State transition matrix | How the state evolves |
| $B \in \mathbb{R}^{d_z}$ | Transition bias | Deterministic input/drift |
| $C \in \mathbb{R}^{d_x \times d_z}$ | Observation matrix | Maps state to observation space |
| $D \in \mathbb{R}^{d_x}$ | Observation bias | Deterministic offset in measurements |
| $Q \in \mathbb{R}^{d_z \times d_z}$ | Process noise covariance | Uncertainty in state dynamics |
| $R \in \mathbb{R}^{d_x \times d_x}$ | Measurement noise covariance | Uncertainty in observations |

These give the following conditional distributions:

$$P(z_t \mid z_{t-1}) = \mathcal{N}(z_t \mid Az_{t-1}+B,\; Q)$$
$$P(x_t \mid z_t) = \mathcal{N}(x_t \mid Cz_t+D,\; R)$$

The noise terms $\varepsilon_t$, $\delta_t$ are mutually independent and independent of the initial state. This is the **continuous-state analogue of the HMM** — replacing the discrete hidden state and categorical distributions with a continuous Gaussian state.

---

## 4. Derivation of the Filtering Recursion

**Goal:** Compute $P(z_t \mid x_1, x_2, \ldots, x_t)$ — the **filtered posterior** at each time step.

Starting from Bayes' theorem and dropping terms constant w.r.t. $z_t$:

$$P(z_t \mid x_{1:t}) \propto P(x_{1:t}, z_t)$$

Factor out $x_t$ using the chain rule:

$$= P(x_t \mid x_{1:t-1}, z_t)\cdot P(x_{1:t-1}, z_t)$$

By **output independence** — given $z_t$, the observation $x_t$ is independent of all past observations:

$$P(x_t \mid x_{1:t-1}, z_t) = P(x_t \mid z_t)$$

Factor the remaining joint:

$$P(x_{1:t-1}, z_t) = P(z_t \mid x_{1:t-1})\cdot P(x_{1:t-1})$$

Since $P(x_{1:t-1})$ does not depend on $z_t$:

$$P(z_t \mid x_{1:t}) \propto P(x_t \mid z_t)\cdot P(z_t \mid x_{1:t-1})$$

Now expand $P(z_t \mid x_{1:t-1})$ by marginalising over $z_{t-1}$:

$$P(z_t \mid x_{1:t-1}) = \int P(z_t, z_{t-1} \mid x_{1:t-1})\,dz_{t-1}$$

$$= \int P(z_t \mid z_{t-1}, x_{1:t-1})\cdot P(z_{t-1} \mid x_{1:t-1})\,dz_{t-1}$$

By the **Markov assumption** — given $z_{t-1}$, $z_t$ is independent of past observations:

$$P(z_t \mid z_{t-1}, x_{1:t-1}) = P(z_t \mid z_{t-1})$$

Therefore the filtering recursion decomposes into two steps:

$$\boxed{P(z_t \mid x_{1:t}) \propto P(x_t \mid z_t) \int P(z_t \mid z_{t-1})\cdot P(z_{t-1} \mid x_{1:t-1})\,dz_{t-1}}$$

This is the **predict-then-update** structure of the Kalman filter.

---

## 5. Predict Step

**Goal:** Compute the **prior** $P(z_t \mid x_{1:t-1})$ by propagating the previous posterior forward through the dynamics.

$$P(z_t \mid x_{1:t-1}) = \int P(z_t \mid z_{t-1})\cdot P(z_{t-1} \mid x_{1:t-1})\,dz_{t-1}$$

Assume the previous posterior is Gaussian: $P(z_{t-1} \mid x_{1:t-1}) = \mathcal{N}(z_{t-1} \mid \mu_{t-1}, \Sigma_{t-1})$.

Since $z_t = Az_{t-1} + B + \varepsilon_t$ is an **affine transformation** of $z_{t-1}$ plus independent Gaussian noise $\varepsilon_t$:

**Predicted mean:**
$$\mu_t^* = \mathbb{E}[z_t \mid x_{1:t-1}] = A\,\mathbb{E}[z_{t-1} \mid x_{1:t-1}] + B = A\mu_{t-1} + B$$

**Predicted covariance:** Using independence of $\varepsilon_t$ from $z_{t-1}$:
$$\Sigma_t^* = \text{Cov}(Az_{t-1}+\varepsilon_t \mid x_{1:t-1}) = A\,\text{Cov}(z_{t-1}\mid x_{1:t-1})\,A^\top + \text{Cov}(\varepsilon_t) = A\Sigma_{t-1}A^\top + Q$$

Therefore:

$$\boxed{P(z_t \mid x_{1:t-1}) = \mathcal{N}(z_t \mid \mu_t^*,\; \Sigma_t^*)}$$

$$\mu_t^* = A\mu_{t-1} + B, \qquad \Sigma_t^* = A\Sigma_{t-1}A^\top + Q$$

**Intuition:** The predicted covariance $\Sigma_t^*$ is larger than $\Sigma_{t-1}$ — uncertainty grows when we propagate forward without observing, because the process noise $Q$ adds additional uncertainty. The amount of growth depends on how much the state is amplified by $A$.

---

## 6. Update Step

**Goal:** Incorporate the new observation $x_t$ to obtain the **filtered posterior** $P(z_t \mid x_{1:t})$.

$$P(z_t \mid x_{1:t}) \propto P(x_t \mid z_t)\cdot P(z_t \mid x_{1:t-1})$$

We use the **joint Gaussian** approach. Given $x_{1:t-1}$, the state $z_t$ and observation $x_t$ are jointly Gaussian (since $x_t = Cz_t + D + \delta_t$ is an affine function of $z_t$ plus independent noise):

$$\begin{pmatrix} z_t \\ x_t \end{pmatrix}\Bigg|x_{1:t-1} \;\sim\; \mathcal{N}\!\left(\begin{pmatrix}\mu_t^* \\ C\mu_t^*+D\end{pmatrix},\; \begin{pmatrix}\Sigma_t^* & \Sigma_t^*C^\top \\ C\Sigma_t^* & C\Sigma_t^*C^\top+R\end{pmatrix}\right)$$

where the cross-covariance $\text{Cov}(z_t, x_t) = \text{Cov}(z_t, Cz_t+\delta_t) = \Sigma_t^*C^\top$ (since $\delta_t$ is independent of $z_t$).

Applying the **Gaussian conditional formula** — if $(a,b)$ are jointly Gaussian with cross-covariance $\Sigma_{ab}$, then $a|b \sim \mathcal{N}(\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(b-\mu_b),\; \Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})$:

**Innovation (residual):**
$$\nu_t = x_t - (C\mu_t^* + D)$$

This is the difference between the actual observation and what we predicted we would observe. It carries all new information.

**Innovation covariance:**
$$S_t = C\Sigma_t^*C^\top + R$$

This is the uncertainty in the predicted observation — combining prediction uncertainty (mapped through $C$) and measurement noise.

**Kalman gain:**
$$\boxed{K_t = \Sigma_t^* C^\top S_t^{-1} = \Sigma_t^*C^\top(C\Sigma_t^*C^\top + R)^{-1}}$$

$K_t \in \mathbb{R}^{d_z \times d_x}$ weights how much to trust the innovation relative to the prediction. If $R$ is large (noisy measurements), $K_t$ is small and we trust the prediction; if $\Sigma_t^*$ is large (uncertain prediction), $K_t$ is large and we trust the measurement.

**Updated (filtered) mean and covariance:**

$$\boxed{\mu_t = \mu_t^* + K_t\,\nu_t = \mu_t^* + K_t(x_t - C\mu_t^* - D)}$$

$$\boxed{\Sigma_t = (I - K_t C)\,\Sigma_t^*}$$

**Verification that $\Sigma_t = (I-K_tC)\Sigma_t^*$:** From the conditional Gaussian formula, the posterior covariance is:
$$\Sigma_t = \Sigma_t^* - \Sigma_t^*C^\top(C\Sigma_t^*C^\top+R)^{-1}C\Sigma_t^* = \Sigma_t^* - K_tC\Sigma_t^* = (I-K_tC)\Sigma_t^* \quad\checkmark$$

**Intuition for the update:**
- The state estimate $\mu_t$ is the prediction $\mu_t^*$ corrected by the Kalman gain times the innovation $\nu_t$.
- If $\nu_t = 0$ (observation exactly as predicted), the estimate is unchanged.
- The updated covariance $\Sigma_t = (I-K_tC)\Sigma_t^*$ is always smaller (less uncertain) than the predicted covariance $\Sigma_t^*$ — observing always reduces uncertainty.

**Summary of the predict-update cycle:**

$$\underbrace{(\mu_{t-1}, \Sigma_{t-1})}_{\text{previous posterior}} \xrightarrow{\text{Predict}} \underbrace{(\mu_t^*, \Sigma_t^*)}_{\text{prior}} \xrightarrow{\text{Update with }x_t} \underbrace{(\mu_t, \Sigma_t)}_{\text{filtered posterior}}$$

---

## 7. Full Algorithm

```
Initialise: μ_1 = μ_1,  Σ_1 = Σ_1

For t = 1, 2, ..., T:

  [if t > 1] PREDICT:
    μ_t*  =  A μ_{t-1}  +  B
    Σ_t*  =  A Σ_{t-1} Aᵀ  +  Q

  UPDATE:
    ν_t  =  x_t  -  C μ_t*  -  D            [innovation]
    S_t  =  C Σ_t* Cᵀ  +  R                 [innovation covariance]
    K_t  =  Σ_t* Cᵀ S_t⁻¹                   [Kalman gain]
    μ_t  =  μ_t*  +  K_t ν_t                 [updated mean]
    Σ_t  =  (I - K_t C) Σ_t*                 [updated covariance]
```

**Complexity per step:** $O(d_z^3)$ for the matrix inversion of $S_t$ (which is $d_x \times d_x$; more precisely $O(d_x^3 + d_x d_z^2)$).

---

## 8. Filtering vs. Smoothing vs. Prediction

These three tasks differ in which observations inform the state estimate, exactly as in HMMs:

| Task | Estimate | Uses | Algorithm |
|------|----------|------|-----------|
| **Filtering** | $P(z_t \mid x_{1:t})$ | Past + current | Kalman filter (forward pass) |
| **Smoothing** | $P(z_t \mid x_{1:T})$ | Full sequence | RTS smoother (forward + backward) |
| **Prediction** | $P(z_{t+k} \mid x_{1:t})$ | Past only | Apply predict step $k$ times |

### 8.1 Rauch-Tung-Striebel (RTS) Smoother

After the forward Kalman pass, the backward RTS pass computes the smoothed posterior. Define the smoother gain:

$$G_t = \Sigma_t\,A^\top\,(\Sigma_{t+1}^*)^{-1}$$

Backward recursion (from $t = T-1$ down to $1$):

$$\mu_t^{\text{smooth}} = \mu_t + G_t(\mu_{t+1}^{\text{smooth}} - \mu_{t+1}^*)$$
$$\Sigma_t^{\text{smooth}} = \Sigma_t + G_t(\Sigma_{t+1}^{\text{smooth}} - \Sigma_{t+1}^*)G_t^\top$$

Initialised with $\mu_T^{\text{smooth}} = \mu_T$, $\Sigma_T^{\text{smooth}} = \Sigma_T$.

---

## 9. Extensions

| Extension | Handles | Approach |
|-----------|---------|---------|
| **Extended KF (EKF)** | Mild nonlinearity in $f$ or $h$ | Linearise via first-order Taylor expansion |
| **Unscented KF (UKF)** | Stronger nonlinearity | Sigma points for accurate mean/covariance propagation |
| **Particle Filter** | Full nonlinearity, non-Gaussian | Sequential Monte Carlo |
| **Ensemble KF (EnKF)** | Very high-dimensional systems | Monte Carlo approximation of covariance |
| **Kalman EM** | Unknown parameters $A, C, Q, R$ | EM algorithm with Kalman smoother as E-step |

---

## 10. References

1. **Kalman, R. E.** (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45. [Original Kalman filter paper.]
2. **Kalman, R. E. & Bucy, R. S.** (1961). New results in linear filtering and prediction theory. *Journal of Basic Engineering*, 83(1), 95–108. [Continuous-time extension.]
3. **Rauch, H. E., Tung, F., & Striebel, C. T.** (1965). Maximum likelihood estimates of linear dynamic systems. *AIAA Journal*, 3(8), 1445–1450. [RTS smoother.]
4. **Welch, G. & Bishop, G.** (2006). An introduction to the Kalman filter. Technical Report TR 95-041, University of North Carolina at Chapel Hill. [Widely used introductory tutorial.]
5. **Sarkka, S.** (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press. [Modern, rigorous treatment of Kalman filtering and its extensions within the Bayesian framework.]
6. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §13.3 (Linear dynamical systems and Kalman filter).
7. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Ch. 18 (State space models).
8. **Julier, S. J. & Uhlmann, J. K.** (1997). A new extension of the Kalman filter to nonlinear systems. In *Proceedings of SPIE*, 3068, 182–193. [Original UKF paper.]

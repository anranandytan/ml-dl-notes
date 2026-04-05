# Particle Filter

> **Keywords:** sequential Monte Carlo, importance sampling, sequential importance sampling, SIR filter, bootstrap filter, weight degeneracy, resampling, nonlinear filtering

---

## 1. Motivation & Intuition

The Kalman filter is the optimal Bayesian filter for linear systems with Gaussian noise, but real-world systems are often **nonlinear** and driven by **non-Gaussian** noise:

- A robot navigating by sonar (highly nonlinear range measurements, non-Gaussian clutter).
- Tracking a manoeuvring aircraft (nonlinear dynamics, sudden direction changes).
- Speech recognition with heavy-tailed noise.

For such systems, the posterior $P(z_t|x_{1:t})$ is no longer Gaussian and cannot be represented exactly by a mean and covariance. Extended and Unscented Kalman filters approximate it as Gaussian, but this can break down badly.

**Particle filters** (also called Sequential Monte Carlo, SMC) represent the posterior as a **weighted set of samples** (particles):

$$P(z_t|x_{1:t}) \approx \sum_{i=1}^N \hat{w}_t^{(i)}\,\delta(z_t - z_t^{(i)})$$

Each particle $z_t^{(i)}$ is a hypothesis about the hidden state; its weight $\hat{w}_t^{(i)}$ reflects how well that hypothesis explains the observations. As $N\to\infty$, this approximation converges to the true posterior by the law of large numbers, for **any** nonlinear, non-Gaussian model.

**"Generate and Test":** The particle filter proposes candidate states (generate), weights them by how well they explain the data (test), and periodically concentrates particles in high-probability regions (resample).

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1949 | Monte Carlo methods introduced for nuclear physics simulations (Metropolis, Ulam). |
| 1993 | **Gordon, Salmond & Smith** introduce the Bootstrap Filter (SIR particle filter) — the first practical particle filter — for tracking applications. |
| 1996 | **Kitagawa** independently develops the Monte Carlo filter; **Liu & Chen** introduce the sequential importance sampling framework. |
| 1998 | **Doucet** generalises SIS to arbitrary proposals; **Pitt & Shephard** introduce the auxiliary particle filter. |
| 2001 | **Doucet, de Freitas & Gordon** edit the landmark book *Sequential Monte Carlo Methods in Practice*, consolidating the field. |
| 2002 | **Arulampalam et al.** publish a widely-read tutorial on particle filters for target tracking. |
| 2010s | Particle filters applied at scale: simultaneous localisation and mapping (SLAM), financial time series, probabilistic robotics. |

---

## 3. Nonlinear State Space Model

A particle filter applies to the **general nonlinear state space model**:

$$z_t = g(z_{t-1},\, u_t,\, \varepsilon_t) \qquad \text{(state transition)}$$
$$x_t = h(z_t,\, u_t,\, \delta_t) \qquad \text{(observation model)}$$

where $g$ and $h$ are arbitrary (nonlinear) functions, $u_t$ is a known control input, and the noise terms $\varepsilon_t \sim p_\varepsilon$, $\delta_t \sim p_\delta$ can follow any distribution (not necessarily Gaussian).

This gives the conditional distributions:

$$P(z_t \mid z_{t-1}) \quad\text{(transition density, from }g\text{)}$$
$$P(x_t \mid z_t) \quad\text{(likelihood, from }h\text{)}$$

**Comparison with Kalman filter:**

| | Kalman Filter | Particle Filter |
|--|--|--|
| Dynamics | Linear: $z_t = Az_{t-1}+B+\varepsilon$ | Nonlinear: $z_t = g(z_{t-1},\varepsilon)$ |
| Noise | Gaussian | Any distribution |
| Posterior | Exactly Gaussian: $\mathcal{N}(\mu_t,\Sigma_t)$ | Approximated by $N$ weighted particles |
| Cost per step | $O(d_z^3)$ | $O(N\cdot d_z)$ |

---

## 4. Bayesian Filtering Recursion

The filtering goal is to maintain the posterior $P(z_t|x_{1:t})$ as new observations arrive. As derived for the Kalman filter, the recursion has two steps:

**Step 1 — Predict:** Propagate the previous posterior through the dynamics:

$$P(z_t \mid x_{1:t-1}) = \int P(z_t \mid z_{t-1})\,P(z_{t-1} \mid x_{1:t-1})\,dz_{t-1}$$

**Step 2 — Update:** Incorporate the new observation $x_t$ via Bayes' theorem:

$$P(z_t \mid x_{1:t}) \propto P(x_t \mid z_t)\,P(z_t \mid x_{1:t-1})$$

For nonlinear/non-Gaussian models, both integrals are intractable — the particle filter approximates them with Monte Carlo.

---

## 5. Importance Sampling for Filtering

### 5.1 Basic Idea

We want to approximate expectations under the filtering posterior:

$$\mathbb{E}[f(z_t)] = \int f(z_t)\,p(z_t|x_{1:t})\,dz_t$$

Introduce a tractable **proposal** distribution $q(z_t|x_{1:t})$ and rewrite:

$$= \int f(z_t)\,\frac{p(z_t|x_{1:t})}{q(z_t|x_{1:t})}\,q(z_t|x_{1:t})\,dz_t = \mathbb{E}_q\!\left[f(z_t)\,w_t\right], \quad w_t = \frac{p(z_t|x_{1:t})}{q(z_t|x_{1:t})}$$

Drawing $N$ samples $z_t^{(i)} \sim q$ gives the Monte Carlo estimator with **unnormalised weights** $w_t^{(i)}$:

$$\mathbb{E}[f(z_t)] \approx \frac{\sum_{i=1}^N w_t^{(i)}\,f(z_t^{(i)})}{\sum_{i=1}^N w_t^{(i)}} = \sum_{i=1}^N \hat{w}_t^{(i)}\,f(z_t^{(i)})$$

where the **normalised weights** are $\hat{w}_t^{(i)} = w_t^{(i)}/\sum_j w_t^{(j)}$, satisfying $\sum_i\hat{w}_t^{(i)} = 1$.

Note: $w_t^{(i)} \propto p/q$ and $\hat{w}_t^{(i)}$ are different — the normalisation constant (which we cannot compute) cancels in the ratio, making self-normalised importance sampling applicable even when $p$ is known only up to a constant.

---

## 6. Sequential Importance Sampling (SIS)

### 6.1 From Filtering to Path Posterior

Instead of approximating $P(z_t|x_{1:t})$ at each $t$ independently, SIS works with the **path posterior** $P(z_{1:t}|x_{1:t})$ — the joint posterior over the entire trajectory up to time $t$. This enables a **recursive weight update**.

### 6.2 Factoring the Path Posterior

We derive a recursion for $P(z_{1:t}|x_{1:t})$, dropping normalisation constants:

$$P(z_{1:t}|x_{1:t}) \propto P(z_{1:t}, x_{1:t})$$

Factor out $x_t$:
$$= P(x_t \mid z_{1:t}, x_{1:t-1})\cdot P(z_{1:t}, x_{1:t-1})$$

Apply **output independence** ($x_t$ depends only on $z_t$):
$$= P(x_t \mid z_t)\cdot P(z_{1:t}, x_{1:t-1})$$

Factor out $z_t$ using the **Markov assumption** ($z_t$ depends only on $z_{t-1}$):
$$= P(x_t \mid z_t)\cdot P(z_t \mid z_{t-1})\cdot P(z_{1:t-1}, x_{1:t-1})$$

Factor the remaining joint into the path posterior and marginal:
$$= P(x_t \mid z_t)\cdot P(z_t \mid z_{t-1})\cdot P(z_{1:t-1} \mid x_{1:t-1})\cdot P(x_{1:t-1})$$

Dropping $P(x_{1:t-1})$ (constant w.r.t. the state path):

$$\boxed{P(z_{1:t}|x_{1:t}) \propto P(x_t|z_t)\,P(z_t|z_{t-1})\,P(z_{1:t-1}|x_{1:t-1})}$$

### 6.3 Recursive Weight Update

Assume the proposal factorises as:
$$q(z_{1:t}|x_{1:t}) = q(z_t \mid z_{1:t-1}, x_{1:t})\cdot q(z_{1:t-1}|x_{1:t-1})$$

The unnormalised importance weight at time $t$ is:

$$w_t^{(i)} \propto \frac{P(z_{1:t}|x_{1:t})}{q(z_{1:t}|x_{1:t})} \propto \frac{P(x_t|z_t)\,P(z_t|z_{t-1})\,P(z_{1:t-1}|x_{1:t-1})}{q(z_t|z_{1:t-1},x_{1:t})\,q(z_{1:t-1}|x_{1:t-1})}$$

$$= \frac{P(x_t|z_t)\,P(z_t|z_{t-1})}{q(z_t|z_{1:t-1},x_{1:t})}\cdot\underbrace{\frac{P(z_{1:t-1}|x_{1:t-1})}{q(z_{1:t-1}|x_{1:t-1})}}_{= w_{t-1}^{(i)}}$$

$$\boxed{w_t^{(i)} \propto w_{t-1}^{(i)}\cdot\frac{P(x_t \mid z_t^{(i)})\,P(z_t^{(i)} \mid z_{t-1}^{(i)})}{q(z_t^{(i)} \mid z_{1:t-1}^{(i)},\, x_{1:t})}}$$

This is the **SIS weight update**: the new weight is the old weight multiplied by the **incremental weight ratio** — the likelihood of the new observation times the prior transition, divided by the proposal density at the new particle. No access to $P(x_{1:t})$ is needed.

### 6.4 SIS Algorithm

```
Initialise (t=0):
  For i = 1, ..., N:
    Sample z_0^(i) ~ q(z_0)
    Set w_0^(i) = p(z_0^(i)) / q(z_0^(i))
  Normalise: ŵ_0^(i) = w_0^(i) / Σ_j w_0^(j)

For t = 1, 2, ...:
  [Propagate]
  For i = 1, ..., N:
    Sample z_t^(i)  ~ q(z_t | z_{t-1}^(i), x_{1:t})
    Update weight:
      w_t^(i) ∝ w_{t-1}^(i) · P(x_t | z_t^(i)) · P(z_t^(i) | z_{t-1}^(i))
                              / q(z_t^(i) | z_{t-1}^(i), x_{1:t})

  [Normalise]
    ŵ_t^(i) = w_t^(i) / Σ_j w_t^(j)

  [Estimate]
    E[f(z_t)] ≈ Σ_i ŵ_t^(i) f(z_t^(i))
```

---

## 7. Weight Degeneracy and Resampling

### 7.1 The Degeneracy Problem

A fundamental problem with SIS: after a few time steps, all but one or a few particles have negligible weight. This **weight degeneracy** means most particles contribute nothing to the estimate, wasting computation.

Formally, the variance of the weights increases monotonically over time and cannot be reduced by choice of proposal (Liu & Chen, 1995). The **effective sample size**:

$$N_{\text{eff}} = \frac{1}{\sum_{i=1}^N (\hat{w}_t^{(i)})^2} \in [1, N]$$

measures how many equally-weighted particles the current weighted set is equivalent to. When $N_{\text{eff}} \ll N$, degeneracy is severe.

### 7.2 Resampling

The standard fix: **resample** the particles whenever degeneracy becomes severe ($N_{\text{eff}} < N/2$, say). Resampling draws $N$ new particles from the current weighted set with replacement, with probabilities proportional to $\hat{w}_t^{(i)}$, then resets all weights to $1/N$.

Common resampling schemes (in order of efficiency):
1. **Multinomial resampling:** Draw $N$ samples independently from $\{\hat{w}_t^{(i)}\}$. Simple but high variance.
2. **Systematic resampling:** Use a single uniform random variable with $N$ evenly spaced offsets. Lower variance, $O(N)$.
3. **Stratified resampling:** Draw one sample uniformly from each of $N$ strata. Similar variance to systematic.

Resampling **concentrates particles in high-probability regions** but introduces **sample impoverishment**: after resampling, multiple particles may be at the same location, reducing diversity.

### 7.3 Choice of Proposal: The Bootstrap Filter (SIR)

The simplest and most widely used choice of proposal is the **prior transition**:

$$q(z_t \mid z_{t-1}^{(i)}, x_{1:t}) = P(z_t \mid z_{t-1}^{(i)})$$

With this choice, the transition density in the numerator and the proposal in the denominator cancel:

$$w_t^{(i)} \propto w_{t-1}^{(i)}\cdot\frac{P(x_t \mid z_t^{(i)})\,\cancel{P(z_t^{(i)} \mid z_{t-1}^{(i)})}}{\cancel{P(z_t^{(i)} \mid z_{t-1}^{(i)})}} = w_{t-1}^{(i)}\cdot P(x_t \mid z_t^{(i)})$$

With resampling between steps ($w_{t-1}^{(i)} = 1/N$ after each resample):

$$\hat{w}_t^{(i)} \propto P(x_t \mid z_t^{(i)})$$

The weight of each particle is simply the **likelihood of the new observation given the particle's position**. This is the **Bootstrap Filter** (Gordon et al., 1993), also called the SIR particle filter:

```
Bootstrap Filter:
  For i = 1, ..., N:
    1. [Predict]   Sample z_t^(i) ~ P(z_t | z_{t-1}^(i))   [propagate via dynamics]
    2. [Weight]    w_t^(i) = P(x_t | z_t^(i))               [likelihood weight]
  3. [Normalise]   ŵ_t^(i) = w_t^(i) / Σ_j w_t^(j)
  4. [Resample]    Draw N new particles from {z_t^(i)} with probs {ŵ_t^(i)}
                   Reset all weights to 1/N
```

**Advantage:** No knowledge of $P(x_t|z_t)$ in closed form is needed for the proposal — only the ability to evaluate it pointwise for weighting. Extremely simple to implement.

**Disadvantage:** Sampling from the prior ignores the current observation $x_t$. If $P(x_t|z_t)$ is very peaked (informative observation), most prior samples will fall in low-likelihood regions and get negligible weight, causing severe degeneracy. Better proposals incorporate $x_t$ to guide particles toward high-likelihood regions.

---

## 8. Full Algorithm Comparison

| Algorithm | Proposal | Weight | Resampling | Complexity |
|-----------|----------|--------|-----------|-----------|
| **SIS** | Any $q(z_t|z_{t-1},x_{1:t})$ | $w_t \propto w_{t-1}\cdot\frac{P(x_t|z_t)P(z_t|z_{t-1})}{q(z_t|z_{t-1},x_{1:t})}$ | No | Degenerates |
| **Bootstrap (SIR)** | $P(z_t|z_{t-1})$ | $w_t \propto P(x_t|z_t)$ | Yes | $O(N)$ per step |
| **Auxiliary PF** | $P(z_t|z_{t-1},x_t)$ approx. | Adjusted | Yes | $O(N)$ per step, better ESS |
| **Optimal proposal** | $P(z_t|z_{t-1},x_t)$ exact | $w_t \propto P(x_t|z_{t-1})$ | Yes | Minimises variance; often intractable |

---

## 9. Summary

```
State space: z_t = g(z_{t-1}, ε_t),  x_t = h(z_t, δ_t)    [nonlinear, non-Gaussian]

Represent posterior as N weighted particles:
  P(z_t | x_{1:t}) ≈ Σ_i ŵ_t^(i) δ(z_t - z_t^(i))

SIS weight recursion (key formula):
  w_t^(i) ∝ w_{t-1}^(i) · P(x_t | z_t^(i)) · P(z_t^(i) | z_{t-1}^(i))
                          / q(z_t^(i) | z_{t-1}^(i), x_{1:t})

Bootstrap filter (simplest case, q = prior):
  Propagate: z_t^(i) ~ P(z_t | z_{t-1}^(i))
  Weight:    ŵ_t^(i) ∝ P(x_t | z_t^(i))
  Resample to avoid degeneracy
```

---

## 10. References

1. **Gordon, N. J., Salmond, D. J., & Smith, A. F. M.** (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F — Radar and Signal Processing*, 140(2), 107–113. [Original Bootstrap/SIR particle filter.]
2. **Kitagawa, G.** (1996). Monte Carlo filter and smoother for non-Gaussian nonlinear state space models. *Journal of Computational and Graphical Statistics*, 5(1), 1–25. [Independent derivation of the particle filter.]
3. **Liu, J. S. & Chen, R.** (1998). Sequential Monte Carlo methods for dynamic systems. *Journal of the American Statistical Association*, 93(443), 1032–1044. [SIS framework; importance of resampling.]
4. **Arulampalam, M. S., Maskell, S., Gordon, N., & Clapp, T.** (2002). A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. *IEEE Transactions on Signal Processing*, 50(2), 174–188. [The most widely cited tutorial on particle filters.]
5. **Doucet, A., de Freitas, N., & Gordon, N.** (Eds.) (2001). *Sequential Monte Carlo Methods in Practice*. Springer. [Comprehensive edited volume covering all aspects of SMC.]
6. **Doucet, A. & Johansen, A. M.** (2009). A tutorial on particle filtering and smoothing: Fifteen years later. In Crisan, D. & Rozovskii, B. (Eds.), *The Oxford Handbook of Nonlinear Filtering*, pp. 656–704. Oxford University Press. [Modern, comprehensive tutorial.]
7. **Pitt, M. K. & Shephard, N.** (1999). Filtering via simulation: Auxiliary particle filters. *Journal of the American Statistical Association*, 94(446), 590–599. [Auxiliary particle filter — better proposal using a look-ahead step.]
8. **Thrun, S., Burgard, W., & Fox, D.** (2005). *Probabilistic Robotics*. MIT Press. Ch. 4 (Nonparametric filters). [Particle filters in robotics and SLAM.]

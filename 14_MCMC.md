# 14 — Markov Chain Monte Carlo

> **Keywords:** Monte Carlo integration, Markov chain, stationary distribution, detailed balance, Metropolis-Hastings, Gibbs sampling, burn-in, mixing, ergodicity

---

## 1. The Problem MCMC Solves

In Bayesian inference we need to compute expectations under the posterior $p(\theta \mid \mathcal{D})$:

$$\mathbb{E}_{p(\theta|\mathcal{D})}[f(\theta)] = \int f(\theta)\, p(\theta \mid \mathcal{D})\, d\theta$$

For most models this integral is analytically intractable. The posterior may be high-dimensional, multimodal, or have no closed form. **MCMC** constructs a Markov chain whose stationary distribution is exactly $p(\theta \mid \mathcal{D})$. Running the chain long enough and averaging over the samples gives Monte Carlo estimates of the integral.

---

## 2. Historical Notes

| Year | Event |
|------|-------|
| 1949 | **Metropolis et al.** invent the first MCMC algorithm for statistical mechanics simulations. |
| 1953 | **Metropolis, Rosenbluth, Rosenbluth, Teller & Teller** publish the original Metropolis algorithm. |
| 1970 | **Hastings** generalises to asymmetric proposal distributions (Metropolis-Hastings). |
| 1984 | **Geman & Geman** introduce Gibbs sampling for image analysis. |
| 1987 | **Duane et al.** propose Hamiltonian Monte Carlo (HMC). |
| 2011 | **Hoffman & Gelman** introduce the No-U-Turn Sampler (NUTS), making HMC practical and widely used (Stan, PyMC). |

---

## 3. Monte Carlo Estimation

**Monte Carlo integration** approximates an expectation using i.i.d. samples $\{\theta^{(s)}\}_{s=1}^S \sim p$:

$$\mathbb{E}_p[f(\theta)] \approx \frac{1}{S}\sum_{s=1}^S f(\theta^{(s)})$$

By the Law of Large Numbers, this converges to the true expectation as $S \to \infty$.

**The problem:** we cannot draw i.i.d. samples from a complex posterior. MCMC produces *correlated* samples that still converge to the correct distribution, at the cost of effective sample size.

---

## 4. Markov Chains

### 4.1 Definition

A **Markov chain** on state space $\mathcal{X}$ is a sequence $\theta^{(0)}, \theta^{(1)}, \theta^{(2)}, \ldots$ where each state depends only on the previous one:

$$p(\theta^{(t+1)} \mid \theta^{(t)}, \theta^{(t-1)}, \ldots, \theta^{(0)}) = T(\theta^{(t+1)} \mid \theta^{(t)})$$

$T(\theta' \mid \theta)$ is the **transition kernel**.

### 4.2 Stationary Distribution

A distribution $\pi$ is the **stationary distribution** of the chain if it is invariant under the transition:

$$\pi(\theta') = \int T(\theta' \mid \theta)\, \pi(\theta)\, d\theta$$

If the chain is at stationarity at time $t$, it remains there for all $t' > t$.

### 4.3 Detailed Balance (Reversibility)

A sufficient (but not necessary) condition for $\pi$ to be the stationary distribution is **detailed balance**:

$$\pi(\theta)\, T(\theta' \mid \theta) = \pi(\theta')\, T(\theta \mid \theta') \qquad \forall\, \theta, \theta'$$

**Why sufficient but not necessary:** detailed balance says the probability flux from $\theta$ to $\theta'$ equals the flux from $\theta'$ to $\theta$ — a stronger condition than stationarity, which only requires the *total* flux into each state to be balanced. There exist valid MCMC methods (e.g., lifted samplers, some slice samplers) that have $\pi$ as stationary distribution without satisfying detailed balance. In practice, most classical algorithms (MH, Gibbs) satisfy detailed balance.

### 4.4 Ergodicity

For MCMC to work, the chain must be **ergodic**: for any starting state $\theta^{(0)}$, the chain must converge to $\pi$ as $t \to \infty$. Ergodicity requires:

- **Irreducibility:** every state can be reached from every other state (the chain can explore all of $\mathcal{X}$).
- **Aperiodicity:** the chain does not oscillate in a periodic pattern.

Under ergodicity, the time average converges to the space average:

$$\frac{1}{T}\sum_{t=1}^T f(\theta^{(t)}) \xrightarrow{T \to \infty} \mathbb{E}_\pi[f(\theta)] \qquad \text{almost surely}$$

---

## 5. Metropolis-Hastings Algorithm

### 5.1 Setup

We want to sample from a target $\pi(\theta) \propto \tilde{\pi}(\theta)$ where $\tilde{\pi}$ is the unnormalised density (e.g., the product of likelihood and prior). We only need to evaluate $\tilde{\pi}$ pointwise — the normalising constant cancels.

Choose a **proposal distribution** $q(\theta' \mid \theta)$ that is easy to sample from (e.g., a Gaussian centred at $\theta$).

### 5.2 Algorithm

At each iteration $t$:

1. Sample a candidate $\theta^* \sim q(\theta^* \mid \theta^{(t)})$.
2. Compute the **acceptance ratio**:

$$\alpha = \min\left(1,\;\frac{\tilde{\pi}(\theta^*)\, q(\theta^{(t)} \mid \theta^*)}{\tilde{\pi}(\theta^{(t)})\, q(\theta^* \mid \theta^{(t)})}\right)$$

3. Accept: set $\theta^{(t+1)} = \theta^*$ with probability $\alpha$; otherwise set $\theta^{(t+1)} = \theta^{(t)}$.

### 5.3 Why Detailed Balance Is Satisfied

The MH transition kernel is:

$$T(\theta' \mid \theta) = q(\theta' \mid \theta)\,\alpha(\theta, \theta') + \delta_\theta(\theta')\,r(\theta)$$

where $r(\theta) = 1 - \int q(\theta' \mid \theta)\alpha(\theta,\theta')\,d\theta'$ is the rejection probability and $\delta_\theta$ is the atom at $\theta$.

For the continuous part ($\theta' \neq \theta$), verify detailed balance:

$$\pi(\theta)\, q(\theta' \mid \theta)\,\alpha(\theta, \theta') = \pi(\theta)\, q(\theta' \mid \theta)\min\left(1,\frac{\pi(\theta')\, q(\theta \mid \theta')}{\pi(\theta)\, q(\theta' \mid \theta)}\right)$$

**Case 1:** $\pi(\theta')q(\theta\mid\theta') \geq \pi(\theta)q(\theta'\mid\theta)$. Then $\alpha(\theta,\theta')=1$ and:

$$\pi(\theta)\,q(\theta'\mid\theta)\cdot 1 = \pi(\theta')\,q(\theta\mid\theta')\cdot\frac{\pi(\theta)\,q(\theta'\mid\theta)}{\pi(\theta')\,q(\theta\mid\theta')} = \pi(\theta')\,q(\theta\mid\theta')\,\alpha(\theta',\theta) \checkmark$$

**Case 2:** $\pi(\theta')q(\theta\mid\theta') < \pi(\theta)q(\theta'\mid\theta)$. Symmetric by swapping the roles of $\theta$ and $\theta'$.

### 5.4 Special Cases

**Metropolis algorithm:** symmetric proposal $q(\theta' \mid \theta) = q(\theta \mid \theta')$ (e.g., Gaussian centred at $\theta$). Then $q$ cancels and $\alpha = \min(1, \tilde{\pi}(\theta^*)/\tilde{\pi}(\theta^{(t)}))$.

**Random-walk Metropolis:** $\theta^* = \theta^{(t)} + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \Sigma_{\text{prop}})$. Step size $\Sigma_{\text{prop}}$ controls the trade-off: too small → slow exploration; too large → frequent rejection.

**Optimal acceptance rate:** for random-walk MH in high dimensions, the acceptance rate should target approximately 23.4% (Roberts, Gelman & Gilks, 1997).

### 5.5 Independence Sampler

$q(\theta' \mid \theta) = q(\theta')$ (proposal does not depend on current state). Then:

$$\alpha = \min\left(1,\; \frac{\pi(\theta^*)q(\theta^{(t)})}{\pi(\theta^{(t)})q(\theta^*)}\right) = \min\left(1,\; \frac{w(\theta^*)}{w(\theta^{(t)})}\right), \qquad w(\theta) = \frac{\pi(\theta)}{q(\theta)}$$

This is the importance weight ratio. Works well when $q$ is a good approximation to $\pi$; breaks down when $\pi/q$ has heavy tails.

---

## 6. Gibbs Sampling

### 6.1 Idea

When the target is a joint distribution $\pi(\theta_1, \ldots, \theta_p)$, Gibbs sampling cycles through variables, sampling each from its **full conditional** while holding all others fixed:

$$\theta_{j}^{(t+1)} \sim p\left(\theta_j \mid \theta_1^{(t+1)}, \ldots, \theta_{j-1}^{(t+1)}, \theta_{j+1}^{(t)}, \ldots, \theta_p^{(t)}\right)$$

No proposal distribution is needed, and every sample is accepted — provided the full conditionals can be sampled from.

### 6.2 Gibbs Is a Special Case of MH

Propose $\theta^*$ by sampling from the full conditional of $\theta_j$. The MH acceptance ratio is:

$$\alpha = \frac{\pi(\theta_{j}^{*})\,\pi(\theta_{-j})\,p(\theta_{j}^{(t)}\mid\theta_{-j})}{\pi(\theta_{j}^{(t)})\,\pi(\theta_{-j})\,p(\theta_{j}^{*}\mid\theta_{-j})} = \frac{\pi(\theta_{j}^{*}\mid\theta_{-j})\,p(\theta_{j}^{(t)}\mid\theta_{-j})}{\pi(\theta_{j}^{(t)}\mid\theta_{-j})\,p(\theta_{j}^{*}\mid\theta_{-j})} = 1$$

The ratio is always 1 because the proposal is exactly the target conditional. Gibbs sampling thus has 100% acceptance rate.

### 6.3 Limitation

Gibbs can mix slowly when variables are highly correlated — each coordinate move is small because others are held fixed. Reparameterisation or block updates (jointly sampling correlated variables) can help.

---

## 7. Practical Considerations

### 7.1 Burn-in

The chain starts from an arbitrary $\theta^{(0)}$, not from stationarity. Early samples are not from $\pi$. **Burn-in** refers to discarding the first $B$ samples:

$$\hat{\mu} = \frac{1}{S - B}\sum_{t=B+1}^S f(\theta^{(t)})$$

Choosing $B$ is heuristic; convergence diagnostics (Gelman-Rubin $\hat{R}$ statistic) provide guidance.

### 7.2 Mixing and Autocorrelation

Consecutive MCMC samples are correlated. The **effective sample size** (ESS) corrects for this:

$$\text{ESS} = \frac{S}{1 + 2\sum_{k=1}^\infty \rho_k}$$

where $\rho_k = \text{Corr}(f(\theta^{(t)}), f(\theta^{(t+k)}))$ is the lag-$k$ autocorrelation. High autocorrelation → small ESS → many more samples needed.

### 7.3 Convergence Diagnostics

- **Gelman-Rubin $\hat{R}$:** run multiple chains from different starting points; compare within-chain to between-chain variance. $\hat{R} \approx 1$ indicates convergence.
- **Trace plots:** visual inspection of $\theta^{(t)}$ over time; should look like white noise after burn-in.
- **Autocorrelation plots:** should decay quickly.

---

## 8. Hamiltonian Monte Carlo (Brief)

HMC introduces auxiliary **momentum** variables $r$ and simulates Hamiltonian dynamics:

$$H(\theta, r) = -\log\pi(\theta) + \frac{1}{2}r^\top M^{-1}r$$

Trajectories move along level sets of $H$, making large moves that are still accepted with high probability. This dramatically reduces autocorrelation compared to random-walk MH.

**Key cost:** requires gradients $\nabla_\theta \log \pi(\theta)$, making it suitable for differentiable posteriors (e.g., in Stan, PyMC, Pyro).

The **No-U-Turn Sampler (NUTS)** automatically adapts the trajectory length, eliminating a critical tuning parameter of HMC (Hoffman & Gelman, 2014).

---

## 9. Summary

```
Goal: sample from π(θ) ∝ π̃(θ)    [normalising constant unknown/intractable]

Metropolis-Hastings:
  Propose θ* ~ q(θ*|θ^(t))
  Accept with prob  α = min(1,  π̃(θ*)q(θ^(t)|θ*) / [π̃(θ^(t))q(θ*|θ^(t))])
  → satisfies detailed balance → π is stationary

Gibbs Sampling:
  Sample θ_j^(t+1) ~ p(θ_j | θ_{-j}^(t))    [full conditional]
  → special case of MH with α = 1 always

Key assumptions for correctness:
  (1) Ergodicity (irreducible + aperiodic)
  (2) Detailed balance (sufficient, not necessary)

Practical checks:
  - Discard burn-in samples
  - Monitor ESS, R̂, trace plots
```

---

## 10. References

1. **Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E.** (1953). Equation of state calculations by fast computing machines. *Journal of Chemical Physics*, 21(6), 1087–1092.
2. **Hastings, W. K.** (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97–109.
3. **Geman, S. & Geman, D.** (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 6(6), 721–741.
4. **Roberts, G. O., Gelman, A., & Gilks, W. R.** (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *The Annals of Applied Probability*, 7(1), 110–120.
5. **Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D.** (1987). Hybrid Monte Carlo. *Physics Letters B*, 195(2), 216–222.
6. **Hoffman, M. D. & Gelman, A.** (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(47), 1593–1623.
7. **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B.** (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall. Ch. 11–12 (MCMC).
8. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §11.2 (Metropolis-Hastings), §11.3 (Gibbs sampling).

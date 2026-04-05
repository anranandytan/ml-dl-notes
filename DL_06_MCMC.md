# Markov Chain Monte Carlo (MCMC)

> **Keywords:** Monte Carlo, rejection sampling, importance sampling, Markov chain, stationary distribution, detailed balance, Metropolis-Hastings, Gibbs sampling, mixing, burn-in

---

## 1. Motivation & Intuition

In Bayesian inference and probabilistic modelling, we frequently need to compute expectations of the form:

$$\mathbb{E}_{p(z|x)}[f(z)] = \int p(z|x)\,f(z)\,dz$$

For most models of interest, the posterior $p(z|x)$ is high-dimensional and analytically intractable — the normalisation constant $Z = \int p(z,x)\,dz$ cannot be computed in closed form.

**Monte Carlo** replaces the integral with a sample average:

$$\mathbb{E}_{p(z|x)}[f(z)] \approx \frac{1}{N}\sum_{i=1}^N f(z^{(i)}), \qquad z^{(i)} \sim p(z|x)$$

The challenge: **how do we draw samples from $p(z|x)$ when we cannot evaluate it normalised?**

MCMC answers this by constructing a Markov chain whose stationary distribution **is** the target $p(z|x)$. Running the chain long enough produces (correlated) samples from the target — no normalisation constant required.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1949 | **Ulam & von Neumann** coin the term "Monte Carlo method" for statistical sampling in physics simulations. |
| 1953 | **Metropolis, Rosenbluth, Rosenbluth, Teller & Teller** introduce the Metropolis algorithm for simulating particle systems. |
| 1970 | **Hastings** generalises the Metropolis algorithm to asymmetric proposal distributions — the Metropolis-Hastings algorithm. |
| 1984 | **Geman & Geman** introduce Gibbs sampling for image processing (MRF posteriors). |
| 1990 | **Gelfand & Smith** demonstrate that Gibbs sampling solves a broad class of Bayesian inference problems, sparking a revolution in applied Bayesian statistics. |
| 1994 | **Neal** introduces Hamiltonian Monte Carlo (HMC) to the statistics community, exploiting gradient information for efficient sampling. |
| 2011 | **Hoffman & Gelman** introduce the No-U-Turn Sampler (NUTS), making HMC parameter-free and practical — the default sampler in Stan and PyMC. |

---

## 3. Basic Sampling Methods

### 3.1 Inverse CDF (Probability Integral Transform)

For a one-dimensional distribution with CDF $F(z) = P(Z \leq z)$: if $u \sim \text{Uniform}(0,1)$, then $z = F^{-1}(u)$ follows the distribution with CDF $F$.

**Limitation:** Requires a closed-form invertible CDF, which is unavailable for most multivariate distributions.

### 3.2 Rejection Sampling

**Setup:** We want to sample from a target $p(z)$ (known up to a constant). Choose a proposal distribution $q(z)$ that we can sample from, and a constant $M > 0$ such that:

$$M\,q(z) \geq p(z) \quad \forall z$$

i.e., $Mq$ is an **envelope** that everywhere dominates $p$.

**Algorithm:**
1. Draw $z \sim q(z)$.
2. Draw $u \sim \text{Uniform}(0,1)$.
3. Compute acceptance probability $\alpha = \frac{p(z)}{M\,q(z)} \in [0,1]$.
4. If $u \leq \alpha$, **accept** $z$ as a sample; otherwise **reject** and return to step 1.

**Correctness:** The accepted samples are distributed as $p(z)$.

**Proof sketch:** The joint density of $(z, u)$ conditional on acceptance is proportional to $q(z) \cdot \frac{p(z)}{Mq(z)} \cdot \mathbf{1}[u \leq \alpha] \propto p(z)$. ✓

**Limitation:** In high dimensions, the acceptance rate $\mathbb{E}[p(z)/(Mq(z))]$ can be exponentially small — rejection sampling becomes impractical above a few dimensions.

### 3.3 Importance Sampling

Rather than sampling from $p(z)$ directly, sample from a tractable proposal $q(z)$ and correct by **importance weights**:

$$\mathbb{E}_{p(z)}[f(z)] = \int f(z)\,p(z)\,dz = \int f(z)\,\frac{p(z)}{q(z)}\,q(z)\,dz = \mathbb{E}_{q(z)}\!\left[f(z)\,\frac{p(z)}{q(z)}\right]$$

Approximating with $N$ samples $z^{(i)} \sim q(z)$:

$$\mathbb{E}_{p}[f(z)] \approx \frac{1}{N}\sum_{i=1}^{N} f(z^{(i)})\,\frac{p(z^{(i)})}{q(z^{(i)})} = \frac{1}{N}\sum_{i=1}^N f(z^{(i)})\,w_i$$

where $w_i = p(z^{(i)})/q(z^{(i)})$ are the **importance weights**.

**Self-normalised importance sampling:** When $p(z)$ is only known up to a constant $\hat{p}(z) = Z_p\,p(z)$:

$$\mathbb{E}_{p}[f(z)] \approx \frac{\sum_i f(z^{(i)})\,\hat{w}_i}{\sum_i \hat{w}_i}, \qquad \hat{w}_i = \frac{\hat{p}(z^{(i)})}{q(z^{(i)})}$$

**Key requirement:** $q(z) > 0$ wherever $p(z) > 0$ (the proposal must have heavier tails than the target in every relevant region). If $q$ misses a region of high $p$-probability, the estimate is badly biased.

### 3.4 Sampling-Importance-Resampling (SIR)

SIR combines importance sampling with resampling to produce (approximately) i.i.d. samples from $p$:

1. Draw $\{z^{(i)}\}_{i=1}^N$ from proposal $q(z)$; compute weights $w_i \propto p(z^{(i)})/q(z^{(i)})$.
2. Normalise: $\tilde{w}_i = w_i / \sum_j w_j$.
3. Resample $M$ points from $\{z^{(i)}\}$ with replacement using probabilities $\{\tilde{w}_i\}$.

The resampled points approximate i.i.d. draws from $p$. SIR is the basis of the **particle filter** for sequential Bayesian inference.

---

## 4. Markov Chains

### 4.1 Definition

A **Markov chain** is a sequence of random variables $\{X_t\}_{t \geq 0}$ satisfying the **Markov property**: the future state depends only on the current state, not the history:

$$P(X_{t+1} = x \mid X_1, X_2, \ldots, X_t) = P(X_{t+1} = x \mid X_t)$$

A **time-homogeneous** chain is characterised by a **transition matrix** $P$ with entries:

$$P_{ij} = p_{ij} = P(X_{t+1} = j \mid X_t = i), \qquad \sum_{j=1}^K p_{ij} = 1 \quad \forall i$$

$$P = \begin{bmatrix} p_{11} & p_{12} & \cdots & p_{1K} \\ p_{21} & p_{22} & \cdots & p_{2K} \\ \vdots & \vdots & \ddots & \vdots \\ p_{K1} & p_{K2} & \cdots & p_{KK} \end{bmatrix}$$

Representing the marginal distribution at time $t$ as a row vector $\pi^{(t)} = [\pi^{(t)}(1), \ldots, \pi^{(t)}(K)]$, the update is:

$$\pi^{(t+1)}(j) = \sum_{i=1}^K \pi^{(t)}(i)\,p_{ij} \qquad\Leftrightarrow\qquad \pi^{(t+1)} = \pi^{(t)} P$$

### 4.2 Stationary Distribution

A distribution $\pi$ is a **stationary distribution** of $P$ if:

$$\pi = \pi P \qquad\Leftrightarrow\qquad \pi_j = \sum_i \pi_i\,p_{ij} \quad \forall j$$

**Convergence theorem.** For an **ergodic** Markov chain (irreducible and aperiodic):
- There exists a unique stationary distribution $\pi$.
- All eigenvalues of $P$ have absolute value $\leq 1$; the eigenvalue 1 has multiplicity 1.
- $\lim_{t\to\infty} \pi^{(t)} = \pi$ regardless of the initial distribution $\pi^{(0)}$.

In MCMC, we **design** the transition kernel $P$ so that the stationary distribution equals our target $p$.

### 4.3 Detailed Balance (Reversibility)

A distribution $\pi$ satisfies **detailed balance** with transition kernel $P$ if:

$$\pi(x)\,p(x \to x^*) = \pi(x^*)\,p(x^* \to x) \qquad \forall\, x, x^*$$

**Claim:** If detailed balance holds, then $\pi$ is a stationary distribution of $P$.

**Proof:** Sum both sides over $x$:
$$\sum_x \pi(x)\,p(x \to x^*) = \sum_x \pi(x^*)\,p(x^*\to x) = \pi(x^*)\sum_x p(x^*\to x) = \pi(x^*)$$

So $(\pi P)_{x^*} = \pi(x^*)$ for all $x^*$, confirming stationarity. $\square$

Detailed balance is a **sufficient but not necessary** condition for stationarity. MCMC algorithms typically satisfy detailed balance — it provides a clean recipe for constructing valid kernels.

---

## 5. Metropolis-Hastings Algorithm

### 5.1 Construction

Suppose we have a target $p(z)$ known up to a constant: $p(z) \propto \hat{p}(z)$. Choose any **proposal distribution** $Q(z \to z^*)$ — a distribution over proposed moves from the current state $z$.

We want to accept or reject the proposal $z^*$ with probability $\alpha(z, z^*)$ such that the resulting chain satisfies detailed balance w.r.t. $p$:

$$p(z)\,Q(z\to z^*)\,\alpha(z, z^*) = p(z^*)\,Q(z^*\to z)\,\alpha(z^*, z)$$

Rearranging:

$$\frac{\alpha(z, z^*)}{\alpha(z^*, z)} = \frac{p(z^*)\,Q(z^*\to z)}{p(z)\,Q(z\to z^*)}$$

The Metropolis-Hastings choice sets one of the two acceptance probabilities to 1 (to maximise the acceptance rate) while satisfying the ratio constraint:

$$\boxed{\alpha(z, z^*) = \min\!\left(1,\;\frac{p(z^*)\,Q(z^*\to z)}{p(z)\,Q(z\to z^*)}\right)}$$

**Verification of detailed balance:** Without loss of generality assume $p(z^*)Q(z^*\to z) \leq p(z)Q(z\to z^*)$. Then $\alpha(z,z^*) = \frac{p(z^*)Q(z^*\to z)}{p(z)Q(z\to z^*)}$ and $\alpha(z^*,z) = 1$:

$$\text{LHS} = p(z)\,Q(z\to z^*)\cdot\frac{p(z^*)Q(z^*\to z)}{p(z)Q(z\to z^*)} = p(z^*)Q(z^*\to z)$$
$$\text{RHS} = p(z^*)\,Q(z^*\to z)\cdot 1 = p(z^*)Q(z^*\to z) \quad\checkmark$$

**Key property:** Since $\alpha$ depends on $p(z)$ and $p(z^*)$ only through their **ratio** $p(z^*)/p(z) = \hat{p}(z^*)/\hat{p}(z)$, the normalisation constant $Z_p$ cancels. We never need to evaluate it.

### 5.2 Algorithm

```
Initialise z^(0) arbitrarily.
For t = 1, 2, ...:
  1. Propose:   z* ~ Q(z | z^(t-1))
  2. Compute:   α  = min(1,  p(z*) Q(z* → z^(t-1))  /  p(z^(t-1)) Q(z^(t-1) → z*))
  3. Accept:    u ~ Uniform(0, 1)
                if u ≤ α:  z^(t) = z*        [accept]
                else:      z^(t) = z^(t-1)   [reject, stay]
```

### 5.3 Special Case: Metropolis Algorithm

When the proposal is **symmetric** — $Q(z\to z^*) = Q(z^*\to z)$ — the acceptance ratio simplifies to:

$$\alpha(z, z^*) = \min\!\left(1,\;\frac{p(z^*)}{p(z)}\right)$$

If the proposed state has higher probability, always accept. If lower, accept with probability equal to the probability ratio. This is the original **Metropolis algorithm** (1953).

---

## 6. Gibbs Sampling

### 6.1 Motivation

For high-dimensional $z = (z_1, \ldots, z_m)$, proposing jointly is hard. Gibbs sampling instead updates **one variable at a time**, cycling through each component by sampling from its **full conditional** distribution.

### 6.2 Algorithm

At each iteration, update each $z_i$ in turn (or in a random order), holding all other components fixed at their current values:

$$z_i^{(t+1)} \sim p(z_i \mid z_1^{(t+1)},\ldots,z_{i-1}^{(t+1)},\; z_{i+1}^{(t)},\ldots,z_m^{(t)}) = p(z_i \mid z_{-i}^{\text{current}})$$

where $z_{-i}$ denotes all components except $z_i$.

**Example with $m=3$:** At iteration $t \to t+1$:

1. $z_1^{(t+1)} \sim p(z_1 \mid z_2^{(t)}, z_3^{(t)})$
2. $z_2^{(t+1)} \sim p(z_2 \mid z_1^{(t+1)}, z_3^{(t)})$ — uses already-updated $z_1$
3. $z_3^{(t+1)} \sim p(z_3 \mid z_1^{(t+1)}, z_2^{(t+1)})$ — uses already-updated $z_1$ and $z_2$

Using the most recently updated values (rather than all values from step $t$) is called **systematic Gibbs sampling** and typically mixes faster.

### 6.3 Gibbs Sampling is a Special Case of MH with Acceptance Rate 1

When Gibbs updates $z_i \to z_i^*$ with $z_{-i}$ fixed ($z_{-i}^* = z_{-i}$), the implicit proposal is $Q(z \to z^*) = p(z_i^* \mid z_{-i})$. The MH acceptance ratio is:

$$\frac{p(z^*)\,Q(z^*\to z)}{p(z)\,Q(z\to z^*)} = \frac{p(z_i^*, z_{-i})\cdot p(z_i \mid z_{-i}^*)}{p(z_i, z_{-i})\cdot p(z_i^* \mid z_{-i})}$$

Since $z_{-i}^* = z_{-i}$, using $p(z_i, z_{-i}) = p(z_i|z_{-i})p(z_{-i})$:

$$= \frac{p(z_i^*\mid z_{-i})\,p(z_{-i})\cdot p(z_i \mid z_{-i})}{p(z_i\mid z_{-i})\,p(z_{-i})\cdot p(z_i^* \mid z_{-i})} = 1$$

The MH acceptance ratio is always 1, so **every Gibbs proposal is accepted**. This makes Gibbs especially efficient when full conditionals are easy to sample from (e.g., conjugate models).

**Requirement:** Gibbs sampling requires tractable full conditionals $p(z_i \mid z_{-i})$. For many graphical models (e.g., Bayesian networks with conjugate priors), these are available in closed form.

---

## 7. Practical Considerations

### 7.1 Burn-in

Early samples from a Markov chain are influenced by the (arbitrary) initial state and do not represent the stationary distribution. These initial samples — the **burn-in** period — are discarded. The length of burn-in depends on how quickly the chain mixes.

### 7.2 Mixing and Autocorrelation

Consecutive MCMC samples are **correlated** (unlike i.i.d. samples). The **autocorrelation time** $\tau$ measures how many steps are needed before two samples are approximately independent. The effective sample size is $N_{\text{eff}} = N / \tau$.

Poor mixing (high $\tau$) arises when:
- The proposal step size is too large or too small (MH).
- The target has strong correlations between components (Gibbs is slow).
- The target is multimodal and the chain gets trapped.

### 7.3 Diagnostics

Common diagnostics:
- **Trace plots:** visualise the chain's trajectory over time; should look like white noise after burn-in.
- **$\hat{R}$ (Gelman-Rubin statistic):** compare multiple chains; $\hat{R} \approx 1$ indicates convergence.
- **Effective sample size (ESS):** computed from autocorrelation; should be large relative to $N$.

### 7.4 Gradient-Based Methods

**Hamiltonian Monte Carlo (HMC)** uses the gradient $\nabla\log p(z)$ to propose distant states with high acceptance probability, dramatically reducing autocorrelation. The **No-U-Turn Sampler (NUTS)** automates HMC's step-size and trajectory-length selection, and is the default sampler in Stan and PyMC.

---

## 8. Summary

```
Goal: sample from p(z) ∝ p̂(z)  [normalisation constant unknown]

Basic methods:
  Rejection sampling:     O(1) cost per accepted sample; fails in high dimensions
  Importance sampling:    reweighted samples from q; fails if q misses high-p regions

MCMC: construct Markov chain with stationary distribution p

  Metropolis-Hastings:
    Propose z* ~ Q(z|z^(t-1))
    Accept with prob min(1, p(z*)Q(z*→z) / p(z)Q(z→z*))
    → Works for any proposal Q; normalisation cancels

  Gibbs sampling (special case of MH):
    Update each z_i ~ p(z_i | z_{-i})
    → Acceptance = 1; requires tractable full conditionals
```

---

## 9. References

1. **Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E.** (1953). Equation of state calculations by fast computing machines. *The Journal of Chemical Physics*, 21(6), 1087–1092. [Original Metropolis algorithm.]
2. **Hastings, W. K.** (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97–109. [Generalisation to asymmetric proposals; Metropolis-Hastings.]
3. **Geman, S. & Geman, D.** (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 6(6), 721–741. [Original Gibbs sampling paper.]
4. **Gelfand, A. E. & Smith, A. F. M.** (1990). Sampling-based approaches to calculating marginal densities. *Journal of the American Statistical Association*, 85(410), 398–409. [Established Gibbs sampling as a general-purpose Bayesian inference tool.]
5. **Neal, R. M.** (1994). An improved acceptance procedure for the hybrid Monte Carlo algorithm. *Journal of Computational Physics*, 111(1), 194–203. [HMC in statistics.]
6. **Hoffman, M. D. & Gelman, A.** (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15, 1593–1623. [NUTS — the default sampler in Stan and PyMC.]
7. **Robert, C. P. & Casella, G.** (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer. [Comprehensive reference for all Monte Carlo methods.]
8. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §11.1 (Basic sampling), §11.2 (Markov chain Monte Carlo), §11.3 (Gibbs sampling).
9. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Ch. 24 (Markov chain Monte Carlo inference).

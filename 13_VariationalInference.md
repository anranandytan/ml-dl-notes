# 13 — Variational Inference

> **Keywords:** variational inference, ELBO, mean-field, CAVI, score function estimator, REINFORCE, control variate, reparameterisation trick, amortised inference, VAE

---

## 1. The Problem

Bayesian inference requires computing the posterior $p(z \mid x)$. For all but the simplest models, this is intractable because the normalising constant $p(x) = \int p(x, z)\,dz$ cannot be evaluated in closed form.

**Variational inference (VI)** turns this integration problem into an **optimisation problem**: find the member $q(z)$ of a tractable family $\mathcal{Q}$ that is closest to $p(z \mid x)$ under KL divergence. The optimisation is typically faster than MCMC and scales to large datasets.

**SGVI** extends VI to the **amortised** setting: instead of fitting a separate $q(z)$ per observation (which requires re-running optimisation for each new $x$), a neural network $q_\phi(z \mid x)$ is trained once across all data. At test time, a single forward pass gives the approximate posterior. This amortised approach is the foundation of the VAE.

---

## 2. Historical Notes

| Year | Event |
|------|-------|
| 1950s | Mean-field theory used in statistical physics. |
| 1999 | **Jordan et al.** systematise VI for graphical models; popularise the ELBO and mean-field. |
| 2008 | **Wainwright & Jordan** give the definitive treatment of VI and exponential families. |
| 2012 | **Ranganath, Gerrish & Blei** introduce Black-Box VI (score function gradient; control variates). |
| 2013–14 | **Kingma & Welling** and **Rezende, Mohamed & Wierstra** independently derive the reparameterisation trick; introduce the VAE. |
| 2017 | **Blei, Kucukelbir & McAuliffe** publish a comprehensive review of VI for statisticians. |

---

## 3. Variational Inference Setup

**Notation:**

| Symbol | Meaning |
|--------|---------|
| $x$ | Observed data |
| $z$ | Latent variables |
| $p(x, z)$ | True joint (unnormalised) |
| $p(z \mid x)$ | True posterior (intractable) |
| $q(z)$ | Variational approximation |

**ELBO decomposition:**

$$\log p(x) = \underbrace{\mathbb{E}_{q(z)}\left[\log\frac{p(x, z)}{q(z)}\right]}_{\mathcal{L}(q)\;=\;\text{ELBO}} + D_{\mathrm{KL}}(q(z) \| p(z \mid x))$$

Since $D_{\mathrm{KL}} \geq 0$, the ELBO is a lower bound: $\mathcal{L}(q) \leq \log p(x)$. Maximising $\mathcal{L}(q)$ over $q \in \mathcal{Q}$ is equivalent to minimising $D_{\mathrm{KL}}(q \| p(z\mid x))$.

**Two equivalent forms:**

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(x, z)] + H[q] = \mathbb{E}_q[\log p(x \mid z)] - D_{\mathrm{KL}}(q(z) \| p(z))$$

The second form separates reconstruction (how well the model explains $x$) from regularisation (how far $q$ strays from the prior $p(z)$).

---

## 4. Mean-Field Variational Inference and CAVI

### 4.1 Mean-Field Approximation

Restrict $q$ to fully factorised distributions over $M$ blocks of variables:

$$q(z) = \prod_{j=1}^M q_j(z_j)$$

This assumes all latent variables are mutually independent under $q$. Under this restriction, for fixed $\{q_i\}_{i \neq j}$, the ELBO-maximising choice of $q_j$ is:

$$\boxed{\log q_j^*(z_j) = \mathbb{E}_{\prod_{i \neq j} q_i}[\log p(x, z)] + \text{const}}$$

i.e., $q_j^*(z_j) \propto \exp\{\mathbb{E}_{i \neq j}[\log p(x, z)]\}$.

**Derivation.** Treating all $q_i$ with $i \neq j$ as fixed, the ELBO as a function of $q_j$ is:

$$\mathcal{L}(q_j) = \int q_j(z_j)\underbrace{\mathbb{E}_{i \neq j}[\log p(x, z)]}_{=:\,\log \tilde{p}(z_j)}\,dz_j - \int q_j\log q_j\,dz_j + \text{const} = -D_{\mathrm{KL}}(q_j \| \tilde{p}) + \text{const}$$

This is maximised when $q_j = \tilde{p}$, i.e., $q_j^*(z_j) \propto \exp\{\mathbb{E}_{i\neq j}[\log p(x,z)]\}$.

### 4.2 Coordinate Ascent Variational Inference (CAVI)

Iterate the optimal updates cyclically:

```
Initialise q_1, ..., q_M.
Repeat until ELBO converges:
  For j = 1, ..., M:
    q_j(z_j) ∝ exp{ E_{i≠j}[log p(x, z)] }
```

Each step increases the ELBO, so the algorithm converges (to a local maximum). **Limitation:** requires re-running for every new data point $x$.

---

## 5. Gradient Estimation for SGVI

SGVI uses stochastic gradient ascent on the ELBO with a parametric family $q_\phi(z \mid x)$. The challenge: computing $\nabla_\phi \mathcal{L}(\phi)$ requires differentiating through an expectation over $q_\phi$.

### 5.1 Score Function Estimator (REINFORCE)

Using the log-derivative trick $\nabla_\phi q_\phi = q_\phi \nabla_\phi \log q_\phi$:

$$\nabla_\phi \mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z)}\left[\nabla_\phi \log q_\phi(z) \cdot \bigl(\log p(x, z) - \log q_\phi(z)\bigr)\right]$$

**Derivation.** Since $\int q_\phi\,dz = 1$, $\int \nabla_\phi q_\phi\,dz = 0$, which makes the entropy gradient term vanish. The remaining term uses $\nabla_\phi q_\phi = q_\phi \nabla_\phi \log q_\phi$.

This estimator is **unbiased** but has **high variance** in practice. The term $\log p(x,z) - \log q_\phi(z)$ acts as a scalar reward that multiplies the gradient of $\log q_\phi$, producing large fluctuations.

### 5.2 Variance Reduction: Baselines and Control Variates

A key technique to reduce variance: subtract a **baseline** $b$ (any constant w.r.t. $z$) from the reward. Since $\mathbb{E}_q[\nabla_\phi \log q_\phi] = 0$, this does not change the expectation:

$$\nabla_\phi \mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\left[\nabla_\phi \log q_\phi(z) \cdot \bigl(\log p(x,z) - \log q_\phi(z) - b\bigr)\right]$$

The optimal baseline minimising variance is $b^* = \mathbb{E}[\nabla_\phi\log q_\phi \cdot r] / \mathbb{E}[(\nabla_\phi\log q_\phi)^2]$ where $r$ is the reward. In practice, the **running mean of the reward** is a common approximation.

More generally, a **control variate** is any function $c(z)$ with known expectation $\mathbb{E}_q[c(z)]$. Subtracting $\lambda c(z)$ and adding back $\lambda\mathbb{E}_q[c(z)]$ does not bias the estimator but can reduce variance if $c$ correlates with the original integrand.

### 5.3 Reparameterisation Trick

When $q_\phi(z \mid x)$ is reparametrisable — i.e., $z$ can be written as a deterministic function of $\phi$ and a noise variable $\varepsilon$ with a fixed distribution:

$$\varepsilon \sim p(\varepsilon), \qquad z = g_\phi(\varepsilon, x)$$

then the gradient moves inside the expectation via the chain rule:

$$\nabla_\phi \mathcal{L}(\phi) = \mathbb{E}_{p(\varepsilon)}\left[\nabla_\phi \bigl(\log p(x, z) - \log q_\phi(z \mid x)\bigr)\right]_{z = g_\phi(\varepsilon, x)}$$

**Gaussian example.** For $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x)I)$:

$$\varepsilon \sim \mathcal{N}(0, I), \qquad z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon$$

The randomness is isolated in $\varepsilon$; $z$ is a deterministic function of $\phi$, so standard autodiff handles the gradient.

**Why lower variance?** The score function estimator uses $\log q_\phi$ as the only link to $\phi$, acting on a scalar reward. The reparameterisation gradient flows through $\log p(x,z)$ directly via $\nabla_z \log p(x,z) \cdot \nabla_\phi g_\phi$, which carries more information about the loss landscape.

### 5.4 Comparison

| Estimator | Requires | Variance | Applicable when |
|-----------|----------|----------|----------------|
| Score function (REINFORCE) | Only $\log q_\phi$ differentiable w.r.t. $\phi$ | High | Discrete or non-reparametrisable $z$ |
| Reparameterisation | $g_\phi$ differentiable; $z$ continuous | Low | Continuous $z$ with reparametrisable $q_\phi$ |

---

## 6. SGVI Algorithm (Amortised Inference)

```
For each minibatch {x^(1), ..., x^(N)}:

  [Encode]
    For each x^(i): compute μ_φ(x^(i)), σ_φ(x^(i)) via encoder network

  [Sample via reparameterisation]
    ε^(i) ~ N(0, I)
    z^(i) = μ_φ(x^(i)) + σ_φ(x^(i)) ⊙ ε^(i)

  [ELBO estimate]
    L(φ, θ) ≈ (1/N) Σ_i [
        log p_θ(x^(i) | z^(i))              [reconstruction]
      − KL(q_φ(z|x^(i)) ‖ p(z))            [regularisation, closed form if Gaussian]
    ]

  [Update]
    φ ← φ + η · ∇_φ L
    θ ← θ + η · ∇_θ L
```

For $q_\phi(z\mid x)=\mathcal{N}(\mu_\phi, \mathrm{diag}(\sigma_\phi^2))$ and $p(z)=\mathcal{N}(0,I)$:

$$D_{\mathrm{KL}}(q_\phi \| p) = \frac{1}{2}\sum_{d=1}^D\left(\mu_{\phi,d}^2 + \sigma_{\phi,d}^2 - \log\sigma_{\phi,d}^2 - 1\right)$$

This has a closed-form gradient, requiring no sampling.

---

## 7. Summary

```
Goal: max_{q ∈ Q}  L(q) = E_q[log p(x,z)/q(z)]
     ≡ min KL(q ‖ p(z|x))

Mean-field CAVI:
  log q_j*(z_j) = E_{i≠j}[log p(x,z)] + const   [coordinate ascent]

SGVI gradient estimators:

  Score function:
    ∇_φ L = E_q[∇_φ log q_φ · (log p − log q_φ)]
    Variance reduction: subtract baseline b from reward

  Reparameterisation (preferred when applicable):
    z = g_φ(ε, x),  ε ~ p(ε)
    ∇_φ L = E_{p(ε)}[∇_z(log p − log q) · ∇_φ g_φ]
    Lower variance; requires continuous, reparametrisable z
```

---

## 8. References

1. **Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K.** (1999). An introduction to variational methods for graphical models. *Machine Learning*, 37(2), 183–233.
2. **Wainwright, M. J. & Jordan, M. I.** (2008). Graphical models, exponential families, and variational inference. *Foundations and Trends in Machine Learning*, 1(1–2), 1–305.
3. **Williams, R. J.** (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3–4), 229–256. [REINFORCE / score function estimator.]
4. **Ranganath, R., Gerrish, S., & Blei, D. M.** (2014). Black box variational inference. *Proceedings of AISTATS 2014*, 814–822. [Score function gradient; control variates for variance reduction.]
5. **Kingma, D. P. & Welling, M.** (2014). Auto-encoding variational Bayes. *ICLR 2014*. [Reparameterisation trick; VAE.]
6. **Rezende, D. J., Mohamed, S., & Wierstra, D.** (2014). Stochastic backpropagation and approximate inference in deep generative models. *ICML 2014*. [Independent derivation of reparameterisation.]
7. **Blei, D. M., Kucukelbir, A., & McAuliffe, J. D.** (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877.
8. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §10.1–10.2.

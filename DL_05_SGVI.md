# Stochastic Gradient Variational Inference (SGVI)

> **Keywords:** variational inference, ELBO, mean-field approximation, CAVI, score function estimator, REINFORCE, reparameterisation trick, VAE, amortised inference

---

## 1. Motivation & Intuition

**The core problem of Bayesian inference** is computing the posterior $p(z|x)$ — the distribution over latent variables given observations. For models beyond simple conjugate families, this posterior is analytically intractable: the normalisation constant $p(x) = \int p(x,z)\,dz$ requires integrating over a high-dimensional latent space.

**Variational inference (VI)** reframes this integration problem as an **optimisation problem**: instead of computing $p(z|x)$ exactly, find the member $q(z)$ of a tractable family $\mathcal{Q}$ that is closest to $p(z|x)$ in KL divergence. The optimisation is typically much faster than MCMC and scales to large datasets.

**Stochastic Gradient Variational Inference (SGVI)** extends VI to the **amortised** setting: rather than fitting a separate $q(z)$ for each data point (which requires re-running optimisation for every new observation), a neural network $q_\phi(z|x)$ (the **encoder**) is trained once to predict the approximate posterior parameters as a function of $x$. At test time, a single forward pass through the encoder produces an approximate posterior.

The combination of amortised inference + SGVI + a neural network decoder is the **Variational Autoencoder (VAE)**.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1950s–60s | Mean-field theory used in statistical physics (Ising model, spin glasses). |
| 1993 | **Hinton & van Camp** introduce the "minimum description length" view connecting variational inference to neural networks. |
| 1999 | **Jordan, Ghahramani, Jaakkola & Saul** systematise variational methods for graphical models, popularising the ELBO and mean-field VI. |
| 2008 | **Wainwright & Jordan** provide a comprehensive treatment of variational inference and exponential families. |
| 2012 | **Ranganath, Gerrish & Blei** introduce Black Box Variational Inference (BBVI): the score-function gradient estimator for general VI. |
| 2013–14 | **Kingma & Welling** and **Rezende, Mohamed & Wierstra** independently introduce the reparameterisation trick, enabling low-variance gradient estimates and the VAE. |
| 2017 | **Blei, Kucukelbir & McAuliffe** publish a comprehensive review of variational inference for statisticians. |

---

## 3. Variational Inference

### 3.1 Setup

**Elements:**

| Symbol | Meaning |
|--------|---------|
| $x$ | Observed data |
| $z$ | Latent variables (and/or parameters) |
| $(x,z)$ | Complete data |
| $\theta$ | Model parameters |
| $q(z)$ | Variational approximation to $p(z|x)$ |

### 3.2 The ELBO Decomposition

Start from the trivially true identity:

$$\log p(x) = \log p(x,z) - \log p(z|x)$$

Add and subtract $\log q(z)$:

$$= \log\frac{p(x,z)}{q(z)} - \log\frac{p(z|x)}{q(z)}$$

Multiply both sides by $q(z)$ and integrate over $z$. The left-hand side gives $\log p(x)$ (since $\int q(z)\,dz = 1$):

$$\log p(x) = \underbrace{\int q(z)\log\frac{p(x,z)}{q(z)}\,dz}_{\mathcal{L}(q)\;=\;\text{ELBO}} + \underbrace{\int q(z)\log\frac{q(z)}{p(z|x)}\,dz}_{D_{\text{KL}}(q(z)\,\|\,p(z|x))\;\geq\;0}$$

$$\boxed{\log p(x) = \mathcal{L}(q) + D_{\text{KL}}(q(z)\,\|\,p(z|x))}$$

Since $D_{\text{KL}} \geq 0$, the ELBO $\mathcal{L}(q)$ is a **lower bound** on $\log p(x)$:

$$\mathcal{L}(q) \leq \log p(x)$$

The bound is tight iff $D_{\text{KL}} = 0$, i.e., $q(z) = p(z|x)$.

**Equivalent forms of the ELBO:**

$$\mathcal{L}(q) = \int q(z)\log\frac{p(x,z)}{q(z)}\,dz = \mathbb{E}_{q(z)}[\log p(x,z)] + H[q(z)]$$

$$= \mathbb{E}_{q(z)}[\log p(x|z)] - D_{\text{KL}}(q(z)\,\|\,p(z))$$

The last form separates the reconstruction term (how well the model explains the data) from the KL regularisation (how much the approximate posterior deviates from the prior).

**Goal:** Find:

$$\hat{q} = \arg\max_{q \in \mathcal{Q}}\;\mathcal{L}(q) = \arg\min_{q\in\mathcal{Q}}\;D_{\text{KL}}(q(z)\,\|\,p(z|x))$$

The optimal $\hat{q}$ is the projection of $p(z|x)$ onto the family $\mathcal{Q}$ in the KL sense.

---

## 4. Mean-Field Variational Inference

### 4.1 Mean-Field Family

The **mean-field approximation** restricts $q$ to fully factorised distributions:

$$q(z) = \prod_{i=1}^M q_i(z_i)$$

All latent variables are assumed to be mutually independent under $q$. This is a strong assumption but makes the optimisation tractable.

### 4.2 Deriving the Optimal Factor

With the mean-field family, the ELBO is:

$$\mathcal{L}(q) = \int q(z)\log p(x,z)\,dz - \int q(z)\log q(z)\,dz$$

**First term:** Expand with the mean-field factorisation:

$$\int q(z)\log p(x,z)\,dz = \int_{z_j} q_j(z_j)\underbrace{\left[\int_{\{z_i\}_{i\neq j}}\prod_{i\neq j}q_i(z_i)\log p(x,z)\,dz_{\neq j}\right]}_{\mathbb{E}_{\prod_{i\neq j}q_i}[\log p(x,z)]}\,dz_j$$

**Second term:** Since $\log q(z) = \sum_i\log q_i(z_i)$ and the factors are independent:

$$\int q(z)\log q(z)\,dz = \sum_{i=1}^M\int_{z_i}q_i(z_i)\log q_i(z_i)\,dz_i = \int_{z_j}q_j\log q_j\,dz_j + \text{const w.r.t. }q_j$$

**Combining:** Treating all $q_i$ ($i\neq j$) as fixed and optimising over $q_j$ alone:

$$\mathcal{L}(q_j) = \int_{z_j}q_j(z_j)\underbrace{\mathbb{E}_{i\neq j}[\log p(x,z)]}_{=:\,\log\hat{p}(x,z_j)}\,dz_j - \int_{z_j}q_j\log q_j\,dz_j + \text{const}$$

$$= -D_{\text{KL}}\!\left(q_j(z_j)\,\Big\|\,\frac{\exp\{\mathbb{E}_{i\neq j}[\log p(x,z)]\}}{\text{normaliser}}\right) + \text{const}$$

This is maximised (KL minimised to zero) when:

$$\boxed{\log q_j^*(z_j) = \mathbb{E}_{\prod_{i\neq j}q_i}\!\left[\log p(x,z)\right] + \text{const}}$$

i.e., $q_j^*(z_j) \propto \exp\!\left\{\mathbb{E}_{i\neq j}[\log p(x,z)]\right\}$

### 4.3 Coordinate Ascent Variational Inference (CAVI)

Since each optimal $q_j^*$ depends on all other $q_i$, we iterate — updating one factor at a time, holding all others fixed:

```
Initialise q_1, q_2, ..., q_M arbitrarily.
Repeat until convergence:
  For j = 1, 2, ..., M:
    q_j*(z_j) ∝ exp{ E_{i≠j}[log p(x, z)] }
    (expectation under current q_1,...,q_{j-1}, q_{j+1},...,q_M)
Check ELBO for convergence.
```

This is coordinate ascent on the ELBO — each step increases $\mathcal{L}$ (or leaves it unchanged), so the algorithm converges to a local maximum.

**Limitation of CAVI:** For each new data point, the entire CAVI must be re-run. This does not scale to large datasets. SGVI addresses this.

---

## 5. Stochastic Gradient Variational Inference (SGVI)

### 5.1 Amortised Inference

Instead of fitting a separate $q(z)$ per data point, parametrise the variational family with a global parameter $\phi$:

$$q_\phi(z|x^{(i)}) \qquad \text{(a neural network encoder)}$$

The ELBO for a single data point becomes:

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z|x^{(i)})}\!\left[\log p_\theta(x^{(i)},z) - \log q_\phi(z|x^{(i)})\right]$$

We jointly optimise over $\phi$ (encoder) and $\theta$ (decoder/model).

### 5.2 Score Function Gradient Estimator (REINFORCE)

**Goal:** Compute $\nabla_\phi\mathcal{L}(\phi)$ to apply stochastic gradient ascent.

Expanding:

$$\nabla_\phi\mathcal{L}(\phi) = \nabla_\phi\int q_\phi(z)\big[\log p_\theta(x,z)-\log q_\phi(z)\big]\,dz$$

Apply the product rule of differentiation:

$$= \underbrace{\int\nabla_\phi q_\phi(z)\big[\log p_\theta-\log q_\phi\big]\,dz}_{\text{Term 1}} + \underbrace{\int q_\phi(z)\nabla_\phi\big[\log p_\theta-\log q_\phi\big]\,dz}_{\text{Term 2}}$$

**Term 2 vanishes:** $\nabla_\phi[\log p_\theta - \log q_\phi] = -\nabla_\phi\log q_\phi = -\frac{\nabla_\phi q_\phi}{q_\phi}$, so:

$$\text{Term 2} = -\int q_\phi\cdot\frac{\nabla_\phi q_\phi}{q_\phi}\,dz = -\int\nabla_\phi q_\phi\,dz = -\nabla_\phi\!\int q_\phi\,dz = -\nabla_\phi(1) = 0$$

**Term 1 — log-derivative trick:** Use $\nabla_\phi q_\phi = q_\phi\nabla_\phi\log q_\phi$:

$$\text{Term 1} = \int q_\phi(z)\,\nabla_\phi\log q_\phi(z)\cdot\big[\log p_\theta(x,z)-\log q_\phi(z)\big]\,dz$$

$$\boxed{\nabla_\phi\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z)}\!\left[\nabla_\phi\log q_\phi(z)\cdot\big(\log p_\theta(x,z)-\log q_\phi(z)\big)\right]}$$

This is the **score function estimator** (also known as the REINFORCE estimator). Monte Carlo approximation with $L$ samples $z^{(l)}\sim q_\phi$:

$$\nabla_\phi\mathcal{L} \approx \frac{1}{L}\sum_{l=1}^L\nabla_\phi\log q_\phi(z^{(l)})\cdot\big(\log p_\theta(x,z^{(l)})-\log q_\phi(z^{(l)})\big)$$

**Problem: High variance.** The score function estimator is unbiased but has very high variance in practice, making training slow and unstable. The $\log p_\theta - \log q_\phi$ term can vary enormously across samples.

---

## 6. Reparameterisation Trick

### 6.1 Change of Variables

The reparameterisation trick reduces variance by making $z$ a deterministic function of $\phi$ and a noise variable $\varepsilon$ that does not depend on $\phi$:

$$z = g_\phi(\varepsilon,\, x), \quad \varepsilon \sim p(\varepsilon)$$

Under this transformation, the distributions are related by the change-of-variables formula:

$$q_\phi(z|x)\,dz = p(\varepsilon)\,d\varepsilon \quad\Leftrightarrow\quad q_\phi(z|x) = p(\varepsilon)\left|\frac{\partial\varepsilon}{\partial z}\right|$$

**Example (Gaussian):** For $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$:

$$\varepsilon \sim \mathcal{N}(0, I), \quad z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$$

The randomness is isolated in $\varepsilon$, which has a fixed distribution independent of $\phi$.

### 6.2 Reparameterised ELBO Gradient

Rewrite the ELBO expectation using the change of variables:

$$\mathcal{L}(\phi) = \int\big[\log p_\theta(x,z)-\log q_\phi(z|x)\big]\,q_\phi(z|x)\,dz = \int\big[\log p_\theta(x,z)-\log q_\phi(z|x)\big]\,p(\varepsilon)\,d\varepsilon$$

Since $p(\varepsilon)$ does not depend on $\phi$, we can move $\nabla_\phi$ inside the integral:

$$\nabla_\phi\mathcal{L}(\phi) = \nabla_\phi\int\big[\log p_\theta(x,z)-\log q_\phi(z|x)\big]p(\varepsilon)\,d\varepsilon = \mathbb{E}_{p(\varepsilon)}\!\left[\nabla_\phi\big(\log p_\theta(x,z)-\log q_\phi(z|x)\big)\right]$$

Applying the chain rule through $z = g_\phi(\varepsilon, x)$:

$$\boxed{\nabla_\phi\mathcal{L}(\phi) = \mathbb{E}_{p(\varepsilon)}\!\left[\nabla_z\big(\log p_\theta(x,z)-\log q_\phi(z|x)\big)\cdot\nabla_\phi g_\phi(\varepsilon,x)\right]}$$

where $z = g_\phi(\varepsilon, x)$.

**Monte Carlo estimate** with $L$ samples $\varepsilon^{(l)}\sim p(\varepsilon)$:

$$\nabla_\phi\mathcal{L}(\phi) \approx \frac{1}{L}\sum_{l=1}^L\nabla_z\big(\log p_\theta(x,z^{(l)})-\log q_\phi(z^{(l)}|x)\big)\cdot\nabla_\phi g_\phi(\varepsilon^{(l)},x)$$

where $z^{(l)} = g_\phi(\varepsilon^{(l)}, x)$.

In practice, $L=1$ is sufficient when mini-batching over data points. The computation graph flows through the deterministic function $g_\phi$, so standard autodiff handles the gradient automatically.

### 6.3 Why Reparameterisation Has Lower Variance

| Estimator | Gradient flows through | Variance |
|-----------|----------------------|---------|
| Score function (REINFORCE) | $\nabla_\phi\log q_\phi$ — depends only on the log-density gradient, not the actual function $\log p_\theta$ | **High** — $\log p_\theta$ enters as a scalar multiplier |
| Reparameterisation | $\nabla_z(\log p_\theta - \log q_\phi)$ — gradient of the objective w.r.t. $z$, which carries information about the landscape | **Low** — gradient signal is richer and more targeted |

The key difference: reparameterisation moves the gradient through $z$ explicitly (following the function landscape), while the score function uses only the log-density as a scalar reward signal.

---

## 7. SGVI Algorithm (VAE)

The full SGVI algorithm (as implemented in the VAE) is:

```
For each minibatch {x^(1), ..., x^(N)}:

  [Encode]
    Compute (μ_φ(x^(i)), σ_φ(x^(i))) via encoder network for each x^(i)

  [Sample]
    ε^(i) ~ N(0, I)
    z^(i) = μ_φ(x^(i)) + σ_φ(x^(i)) ⊙ ε^(i)    [reparameterisation]

  [ELBO]
    L(φ, θ) = E_q[log p_θ(x|z)] - KL(q_φ(z|x) ‖ p(z))

    ≈ (1/N) Σ_i [log p_θ(x^(i) | z^(i))              [reconstruction]
               - KL(N(μ_φ(x^(i)), diag(σ_φ^2(x^(i)))) ‖ N(0,I))]  [KL term]

  [Update]
    φ ← φ + η · ∇_φ L(φ, θ)    [encoder update]
    θ ← θ + η · ∇_θ L(φ, θ)    [decoder update]
```

For Gaussian $q_\phi(z|x) = \mathcal{N}(\mu_\phi, \text{diag}(\sigma_\phi^2))$ and $p(z) = \mathcal{N}(0,I)$, the KL has a closed form:

$$D_{\text{KL}}\!\left(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,I)\right) = \frac{1}{2}\sum_{d=1}^D\left(\mu_d^2 + \sigma_d^2 - \log\sigma_d^2 - 1\right)$$

---

## 8. Summary

```
Variational Inference:
  log p(x) = ELBO(q) + KL(q ‖ p(z|x))
  Maximise ELBO ⟺ Minimise KL to posterior

Mean-Field CAVI:
  log q_j*(z_j) = E_{i≠j}[log p(x,z)] + const    [coordinate ascent]

SGVI (amortised, neural network encoder):
  Two gradient estimators for ∇_φ ELBO:

  Score function (REINFORCE):
    ∇_φ L = E_{q_φ}[∇_φ log q_φ · (log p - log q_φ)]
    → Unbiased, high variance

  Reparameterisation:
    z = g_φ(ε, x),  ε ~ p(ε)
    ∇_φ L = E_{p(ε)}[∇_z(log p - log q) · ∇_φ g_φ]
    → Unbiased, low variance (preferred when applicable)
```

---

## 9. References

1. **Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K.** (1999). An introduction to variational methods for graphical models. *Machine Learning*, 37(2), 183–233. [Foundational paper systematising variational inference for graphical models.]
2. **Wainwright, M. J. & Jordan, M. I.** (2008). Graphical models, exponential families, and variational inference. *Foundations and Trends in Machine Learning*, 1(1–2), 1–305. [Comprehensive treatment of variational methods and exponential families.]
3. **Williams, R. J.** (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3–4), 229–256. [REINFORCE / score function gradient estimator.]
4. **Ranganath, R., Gerrish, S., & Blei, D. M.** (2014). Black box variational inference. *Proceedings of AISTATS 2014*, 814–822. [Score function gradient for general variational inference; variance reduction via control variates.]
5. **Kingma, D. P. & Welling, M.** (2014). Auto-encoding variational Bayes. *ICLR 2014*. [VAE; reparameterisation trick for amortised inference.]
6. **Rezende, D. J., Mohamed, S., & Wierstra, D.** (2014). Stochastic backpropagation and approximate inference in deep generative models. *ICML 2014*. [Independent derivation of reparameterisation and SGVI.]
7. **Blei, D. M., Kucukelbir, A., & McAuliffe, J. D.** (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877. [Comprehensive modern review.]
8. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §10.1 (Variational inference), §10.2 (Mean-field variational Bayes).

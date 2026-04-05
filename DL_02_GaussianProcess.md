# Gaussian Process

> **Keywords:** Gaussian process, kernel function, weight-space view, function-space view, GP regression, marginal likelihood, hyperparameter optimisation

---

## 1. Motivation & Intuition

Parametric models (linear regression, neural networks) represent a function $f$ through a **finite-dimensional parameter vector** $\theta$. The model class is fixed; adding more data refines $\theta$ but cannot change the functional form.

A **Gaussian Process (GP)** places a prior directly over **functions**: instead of saying "I believe $w \sim \mathcal{N}(0, I)$", we say "I believe the function $f$ is smooth with characteristic length-scale $\ell$." This is a prior over an infinite-dimensional space of functions.

The payoff is threefold:

1. **Exact Bayesian inference.** For Gaussian likelihoods, the posterior over functions is also a GP — computed analytically with no approximation.
2. **Calibrated uncertainty.** The predictive distribution is a Gaussian at every test point, with variance that is large where data is sparse and small where data is dense.
3. **Principled model selection.** The marginal likelihood $p(Y|X)$ can be maximised over kernel hyperparameters (length-scale, amplitude, noise level), automatically trading off fit and complexity.

**Intuition.** A GP is a consistent probability distribution over functions. Asking for $f(x^*)$ given training data $(X, Y)$ is just Bayesian conditioning — the posterior mean is the prediction and the posterior variance is the uncertainty.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1950s–60s | **Kriging** developed by Danie Krige and formalised by Georges Matheron for spatial interpolation in geostatistics — equivalent to GP regression. |
| 1978 | O'Hagan introduces GP priors explicitly in a Bayesian statistics context. |
| 1996 | **Neal** shows that a Bayesian neural network with one hidden layer and infinitely many units converges to a GP, connecting the two paradigms. |
| 1998 | **MacKay** writes an influential tutorial on GP regression for the machine learning community. |
| 2006 | **Rasmussen & Williams** publish *Gaussian Processes for Machine Learning* (MIT Press, free online), the standard reference for the field. |
| 2010s | Scalable GP approximations (sparse GPs, inducing points, Kronecker methods) extend GPs to large datasets. |
| 2018–present | **Neural Tangent Kernel (NTK)**: infinitely wide deep networks converge to GPs at initialisation, reviving interest in the GP–neural network connection. |

---

## 3. Formal Definition

**Definition.** A collection of random variables $\{f(x)\}_{x \in \mathcal{X}}$ is a **Gaussian Process** if every finite subcollection $(f(x_1), f(x_2), \ldots, f(x_n))^\top$ has a joint Gaussian distribution. We write:

$$f \sim \mathcal{GP}(m(x),\; K(x, x'))$$

where:
- $m(x) = \mathbb{E}[f(x)]$ is the **mean function**.
- $K(x, x') = \mathbb{E}[(f(x)-m(x))(f(x')-m(x'))]$ is the **covariance (kernel) function**.

The kernel $K$ must be **positive semi-definite**: for any finite set $\{x_1,\ldots,x_n\}$, the Gram matrix $[K(x_i, x_j)]_{i,j}$ must be PSD. This ensures valid (non-negative-definite) covariance matrices.

A GP is fully specified by $m$ and $K$. In practice we often take $m(x) = 0$ (the prior mean) and let the kernel encode all structural assumptions about $f$.

---

## 4. Weight-Space View

### 4.1 From Bayesian Linear Regression to GP

Consider a linear model with a feature map $\phi : \mathbb{R}^p \to \mathbb{R}^D$:

$$f(x) = \phi(x)^\top w, \qquad w \sim \mathcal{N}(0, \Sigma_p)$$

**Mean of $f(x)$:**

$$\mathbb{E}[f(x)] = \phi(x)^\top \mathbb{E}[w] = \phi(x)^\top \cdot 0 = 0$$

**Covariance between $f(x)$ and $f(x')$:** Since both means are zero, the covariance equals the second moment:

$$\text{Cov}(f(x), f(x')) = \mathbb{E}[f(x)\,f(x')] = \mathbb{E}[\phi(x)^\top w\, w^\top \phi(x')] = \phi(x)^\top \mathbb{E}[ww^\top] \phi(x') = \phi(x)^\top \Sigma_p\, \phi(x')$$

This defines a kernel:
$$K(x, x') = \phi(x)^\top \Sigma_p\, \phi(x')$$

**Key insight.** Any finite collection $(f(x_1), \ldots, f(x_n))^\top = \Phi w$ (where $\Phi$ is the feature matrix) is a linear transformation of a Gaussian $w$, hence jointly Gaussian. Therefore, $f$ is a **Gaussian Process**:

$$f \sim \mathcal{GP}(0,\; K(x,x') = \phi(x)^\top \Sigma_p\, \phi(x'))$$

### 4.2 The Kernel Trick

When $D$ (the feature dimension) is very large or infinite, we never need to form $\phi(x)$ explicitly. All computations depend on $x$ only through $K(x, x')$. By choosing $K$ to be a valid PSD kernel, we implicitly work in a (possibly infinite-dimensional) reproducing kernel Hilbert space (RKHS).

---

## 5. Function-Space View

In the function-space view, we work directly with the GP prior on $f$ without reference to an explicit weight vector.

### 5.1 Prior

$$f \sim \mathcal{GP}(m(x),\; K(x,x'))$$

For a finite set of training inputs $X = (x_1, \ldots, x_N)^\top$, define the **Gram matrix**:

$$K(X,X) \in \mathbb{R}^{N \times N}, \qquad [K(X,X)]_{ij} = K(x_i, x_j)$$

The joint prior over function values at $X$ is:

$$f(X) \sim \mathcal{N}(\mu(X),\; K(X,X))$$

where $\mu(X) = (m(x_1), \ldots, m(x_N))^\top$.

### 5.2 Likelihood

Assuming additive i.i.d. Gaussian noise:

$$Y = f(X) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

The marginal distribution of the observations is:

$$Y \sim \mathcal{N}(\mu(X),\; K(X,X) + \sigma^2 I)$$

---

## 6. Common Kernel Functions

The kernel encodes prior beliefs about the smoothness and structure of $f$.

| Kernel | Formula | Properties |
|--------|---------|------------|
| **Squared Exponential (RBF)** | $K(x,x') = \sigma_f^2\exp\!\left(-\frac{\|x-x'\|^2}{2\ell^2}\right)$ | Infinitely differentiable; universal approximator |
| **Matérn $\nu=3/2$** | $K(x,x') = \sigma_f^2\!\left(1+\frac{\sqrt{3}\|x-x'\|}{\ell}\right)\exp\!\left(-\frac{\sqrt{3}\|x-x'\|}{\ell}\right)$ | Once differentiable; more realistic for many physical processes |
| **Matérn $\nu=5/2$** | $K(x,x') = \sigma_f^2\!\left(1+\frac{\sqrt{5}r}{\ell}+\frac{5r^2}{3\ell^2}\right)e^{-\sqrt{5}r/\ell}$, $r=\|x-x'\|$ | Twice differentiable |
| **Linear** | $K(x,x') = x^\top x'$ | Recovers Bayesian linear regression |
| **Periodic** | $K(x,x') = \sigma_f^2\exp\!\left(-\frac{2\sin^2(\pi\|x-x'\|/p)}{\ell^2}\right)$ | Models periodic functions with period $p$ |

Hyperparameters $(\sigma_f, \ell, \sigma^2)$ control amplitude, length-scale, and noise, respectively. These are learned by maximising the marginal likelihood (Section 8).

---

## 7. GP Regression: Predictive Distribution

### 7.1 Joint Distribution

Given training data $(X, Y)$ and test inputs $X^* = (x^*_1, \ldots, x^*_M)^\top$, the prior over $(Y, f(X^*))$ is jointly Gaussian:

$$\begin{pmatrix} Y \\ f(X^*) \end{pmatrix} \sim \mathcal{N}\!\left( \begin{pmatrix} \mu(X) \\ \mu(X^*) \end{pmatrix},\; \begin{pmatrix} K(X,X) + \sigma^2 I & K(X,X^*) \\ K(X^*,X) & K(X^*,X^*) \end{pmatrix} \right)$$

where:
- $K(X,X) \in \mathbb{R}^{N\times N}$: training–training covariance.
- $K(X,X^*) \in \mathbb{R}^{N\times M}$: training–test cross-covariance.
- $K(X^*,X) = K(X,X^*)^\top \in \mathbb{R}^{M\times N}$: test–training cross-covariance.
- $K(X^*,X^*) \in \mathbb{R}^{M\times M}$: test–test covariance.

### 7.2 Posterior over $f(X^*)$

Conditioning on the observed $Y$ using the **Gaussian conditional formula** — if $(a, b)$ are jointly Gaussian with blocks $(\Sigma_{aa}, \Sigma_{ab}; \Sigma_{ba}, \Sigma_{bb})$, then $b \mid a \sim \mathcal{N}(\mu_b + \Sigma_{ba}\Sigma_{aa}^{-1}(a-\mu_a),\; \Sigma_{bb} - \Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab})$ — with $a = Y$, $b = f(X^*)$:

$$P(f(X^*) \mid Y, X, X^*) = \mathcal{N}(\mu^*,\; \Sigma^*)$$

$$\boxed{\mu^* = \mu(X^*) + K(X^*,X)\big(K(X,X)+\sigma^2 I\big)^{-1}(Y-\mu(X))}$$

$$\boxed{\Sigma^* = K(X^*,X^*) - K(X^*,X)\big(K(X,X)+\sigma^2 I\big)^{-1}K(X,X^*)}$$

**Interpreting $\mu^*$:**

The term $K(X^*,X)(K(X,X)+\sigma^2 I)^{-1}$ is an $M\times N$ matrix of **kernel weights** — it measures how similar each test point $x^*_j$ is to each training point $x_i$. The prediction is a weighted sum of the residuals $(Y - \mu(X))$, smoothed by the kernel.

**Interpreting $\Sigma^*$:**

The posterior variance $\Sigma^*$ starts with the prior covariance $K(X^*,X^*)$ and subtracts the **information gained from the data**: $K(X^*,X)(K(X,X)+\sigma^2I)^{-1}K(X,X^*)$. Uncertainty decreases monotonically as more data are observed.

### 7.3 Predictive Distribution for Noisy Observations

For a new noisy observation $y^* = f(x^*) + \varepsilon^*$ with $\varepsilon^* \sim \mathcal{N}(0, \sigma^2)$ independent of everything:

$$P(y^* \mid Y, X, X^*) = \mathcal{N}(\mu^*,\; \Sigma^* + \sigma^2 I)$$

The mean $\mu^*$ is unchanged; the predictive variance adds $\sigma^2$ (irreducible observation noise).

---

## 8. Marginal Likelihood and Hyperparameter Learning

### 8.1 The Marginal Likelihood

Let $\psi = (\sigma_f, \ell, \sigma^2)$ denote the kernel hyperparameters. The **marginal likelihood** (or evidence) is obtained by integrating out $f$:

$$p(Y \mid X, \psi) = \int p(Y \mid f(X), \sigma^2)\,p(f(X) \mid X, \psi)\,df(X) = \mathcal{N}(Y \mid \mu(X),\; K_\psi(X,X) + \sigma^2 I)$$

The log marginal likelihood is:

$$\log p(Y \mid X, \psi) = -\frac{1}{2}(Y-\mu(X))^\top (K+\sigma^2 I)^{-1}(Y-\mu(X)) - \frac{1}{2}\log\det(K+\sigma^2 I) - \frac{N}{2}\log 2\pi$$

where we write $K = K_\psi(X,X)$ for brevity. The three terms have clear interpretations:
1. **Data fit:** $(Y-\mu)^\top(K+\sigma^2I)^{-1}(Y-\mu)$ — penalises poor fit.
2. **Complexity penalty:** $\log\det(K+\sigma^2 I)$ — penalises models that can explain too many datasets (Bayesian Occam's Razor).
3. **Normalisation constant:** independent of $\psi$.

### 8.2 Optimising Hyperparameters

$$\hat{\psi} = \arg\max_\psi \log p(Y \mid X, \psi)$$

This is optimised by gradient ascent. The gradient w.r.t. each hyperparameter involves $\frac{\partial K}{\partial \psi_i}$, which is analytically available for standard kernels.

**Complexity:** Computing $\log p(Y|X,\psi)$ requires the Cholesky factorisation of $(K + \sigma^2 I)$, which costs $O(N^3)$ and $O(N^2)$ memory — the primary computational bottleneck of exact GPs.

---

## 9. Computational Complexity and Scalability

| Operation | Cost |
|-----------|------|
| Training (Cholesky of $K+\sigma^2I$) | $O(N^3)$ time, $O(N^2)$ space |
| Prediction (after training) | $O(N)$ per test point (mean), $O(N^2)$ (variance) |
| Hyperparameter gradient | $O(N^3)$ per step |

For large $N$, exact GP regression becomes infeasible. Standard approximations:

- **Sparse GPs / Inducing Points** (Snelson & Ghahramani, 2006): approximate $K(X,X)$ using $M \ll N$ inducing points, reducing cost to $O(NM^2)$.
- **State-Space / Kalman Filtering** (Solin & Särkkä, 2014): for time-series with Matérn kernels, equivalent to a Kalman filter, giving $O(N)$ inference.
- **Random Fourier Features** (Rahimi & Recht, 2007): approximate shift-invariant kernels with $D$ random features, giving $O(ND + D^3)$ cost.

---

## 10. Summary

```
Prior:         f ~ GP(m(x), K(x,x'))
Likelihood:    Y = f(X) + ε,  ε ~ N(0, σ²I)

Joint:
  [ Y      ]  ~  N( [μ(X) ], [K(X,X)+σ²I   K(X,X*) ] )
  [ f(X*)  ]       [μ(X*)]  [K(X*,X)       K(X*,X*)]

Posterior predictive:
  f(X*)|Y  ~  N(μ*, Σ*)
  μ* = μ(X*) + K(X*,X)(K(X,X)+σ²I)⁻¹(Y - μ(X))
  Σ* = K(X*,X*) - K(X*,X)(K(X,X)+σ²I)⁻¹K(X,X*)

  y*|Y  ~  N(μ*, Σ* + σ²I)      [adds observation noise]
```

---

## 11. References

1. **Rasmussen, C. E. & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press. [The definitive reference. Free PDF: [gaussianprocess.org/gpml](http://www.gaussianprocess.org/gpml)]
2. **MacKay, D. J. C.** (1998). Introduction to Gaussian processes. In Bishop, C. M. (ed.), *Neural Networks and Machine Learning*, NATO ASI Series, pp. 133–165. Springer. [Accessible tutorial introducing GP regression to the ML community.]
3. **Neal, R. M.** (1996). *Bayesian Learning for Neural Networks*. Springer. [Shows that infinite-width Bayesian neural networks converge to GPs; Ch. 2.]
4. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §6.4 (GP regression), §6.4.2 (Learning the hyperparameters).
5. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Ch. 15 (Gaussian processes).
6. **Snelson, E. & Ghahramani, Z.** (2006). Sparse Gaussian processes using pseudo-inputs. *NeurIPS 2005*. [Inducing-point sparse GP approximation.]
7. **Rahimi, A. & Recht, B.** (2007). Random features for large-scale kernel machines. *NeurIPS 2007*. [Random Fourier feature approximation for scalable GPs.]
8. **Jacot, A., Gabriel, F., & Hongler, C.** (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS 2018*. [Infinite-width deep networks as GPs; NTK.]

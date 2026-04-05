# DL 02 — Gaussian Process

> **Keywords:** Gaussian process, kernel function, positive semi-definite, weight-space view, function-space view, GP regression, marginal likelihood, hyperparameter optimisation, Cholesky factorisation

---

## 1. What Is a Gaussian Process?

In Bayesian linear regression (`DL_01`), we put a Gaussian prior over a finite-dimensional weight vector $w$. A **Gaussian Process (GP)** takes this one step further: we put a prior directly over **functions**.

Think of it this way. A function $f:\mathbb{R}^p\to\mathbb{R}$ is an infinite-dimensional object — it has a value at every point in $\mathbb{R}^p$. A GP is a probability distribution over all such functions. If you sample from a GP, you get a function. If you observe data, the posterior is another GP — the updated distribution over functions consistent with the data.

**Formal definition.** A collection of random variables $\{f(x)\}_{x\in\mathcal{X}}$ is a **Gaussian Process** if every finite subcollection $(f(x_1),\ldots,f(x_n))^\top$ follows a joint Gaussian distribution. We write:

$$f \sim \mathcal{GP}(m(x),\; k(x,x'))$$

where:
- $m(x) = \mathbb{E}[f(x)]$: the **mean function** (often set to 0).
- $k(x,x') = \mathbb{E}[(f(x)-m(x))(f(x')-m(x'))]$: the **covariance (kernel) function**.

The GP is completely specified by $m$ and $k$. The kernel $k$ encodes our prior beliefs about the smoothness and structure of $f$ — it is the most important design choice.

---

## 2. Historical Notes

| Year | Event |
|------|-------|
| 1951 | Krige and Matheron develop **kriging** for spatial interpolation in geostatistics — mathematically equivalent to GP regression, independently of the statistics literature. |
| 1978 | O'Hagan formally introduces GP priors in a Bayesian statistics context. |
| 1996 | **Neal** proves that a Bayesian neural network with one infinitely wide hidden layer converges to a GP, connecting the two paradigms. |
| 1998 | **MacKay** writes an influential tutorial on GP regression for the ML community. |
| 2006 | **Rasmussen & Williams** publish *Gaussian Processes for Machine Learning* (MIT Press, [free online](http://www.gaussianprocess.org/gpml/)), the standard reference. |
| 2018 | **Neural Tangent Kernel (NTK)** (Jacot et al.): infinitely wide deep networks at initialisation converge to a GP, reviving interest in the GP–neural network connection. |

---

## 3. From Bayesian Linear Regression to GPs

### 3.1 Weight-Space View

In `DL_01` we showed: place a Gaussian prior $w\sim\mathcal{N}(0,\Sigma_p)$ on the weights of a linear model $f(x)=\phi(x)^\top w$. Then for any finite set of inputs, the function values $(f(x_1),\ldots,f(x_n))$ are jointly Gaussian:

$$\mathbb{E}[f(x)] = \phi(x)^\top\mathbb{E}[w] = 0$$

$$\text{Cov}(f(x), f(x')) = \phi(x)^\top\Sigma_p\phi(x') =: k(x,x')$$

This is already a GP! The key insight: **any Bayesian linear model with a Gaussian weight prior is a GP**, with kernel $k(x,x') = \phi(x)^\top\Sigma_p\phi(x')$.

### 3.2 Function-Space View

Instead of thinking about weights $w$, we can think directly about the distribution over functions. This is cleaner when the feature space $\mathcal{H}$ is infinite-dimensional.

The **function-space view** says: define a GP prior by specifying $m(x)$ and $k(x,x')$ directly, without any reference to an underlying weight vector. Inference and prediction are then done entirely in function space using the kernel.

The two views give **identical predictions**. The function-space view is preferred computationally when the feature dimension is large or infinite, because all computation involves only the $N\times N$ kernel matrix $K$, not a potentially infinite-dimensional weight posterior.

---

## 4. Kernel Functions

The kernel is the heart of a GP. It encodes what kinds of functions are a priori plausible.

### 4.1 What Makes a Valid Kernel?

A function $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ is a valid (Mercer) kernel if and only if it is:
1. **Symmetric:** $k(x,x') = k(x',x)$ for all $x,x'$.
2. **Positive semi-definite (PSD):** for any finite set $\{x_1,\ldots,n\}$ and any $c\in\mathbb{R}^n$:

$$\sum_{i=1}^n\sum_{j=1}^n c_ic_j k(x_i,x_j) \geq 0$$

Equivalently, the Gram matrix $K$ with $K_{ij}=k(x_i,x_j)$ must be PSD.

**Why PSD?** A covariance matrix must be PSD (variances are non-negative, and any linear combination of jointly Gaussian variables has non-negative variance). A kernel that violates PSD would imply a covariance matrix with negative eigenvalues — an invalid covariance.

### 4.2 Common Kernels and Their Properties

| Kernel | Formula | Differentiability | Lengthscale |
|--------|---------|-------------------|-------------|
| **Squared Exponential (RBF)** | $\sigma_f^2\exp\!\left(-\frac{\|x-x'\|^2}{2\ell^2}\right)$ | Infinitely differentiable | $\ell$ |
| **Matérn $\nu=1/2$** | $\sigma_f^2\exp\!\left(-\frac{\|x-x'\|}{\ell}\right)$ | Continuous, not differentiable | $\ell$ |
| **Matérn $\nu=3/2$** | $\sigma_f^2\!\left(1+\frac{\sqrt{3}\|x-x'\|}{\ell}\right)\exp\!\left(-\frac{\sqrt{3}\|x-x'\|}{\ell}\right)$ | Once differentiable | $\ell$ |
| **Matérn $\nu=5/2$** | $\sigma_f^2\!\left(1+\frac{\sqrt{5}r}{\ell}+\frac{5r^2}{3\ell^2}\right)e^{-\sqrt{5}r/\ell}$ | Twice differentiable | $\ell$ |
| **Linear** | $\sigma_f^2\,x^\top x'$ | — | — |
| **Periodic** | $\sigma_f^2\exp\!\left(-\frac{2\sin^2(\pi\|x-x'\|/p)}{\ell^2}\right)$ | Infinitely differentiable | $\ell$, period $p$ |

**Choosing a kernel:**
- **RBF:** default choice; models very smooth functions. Oversmooths data that is rough.
- **Matérn:** more flexible; $\nu$ controls the smoothness. $\nu=3/2$ or $5/2$ is often better than RBF in practice (Rasmussen & Williams, 2006, §4.2).
- **Matérn $\nu=1/2$:** equivalent to an Ornstein-Uhlenbeck process; only continuous, not differentiable — appropriate for very rough data.
- **Linear:** recovers Bayesian linear regression.
- **Periodic:** models functions that repeat with period $p$.

**Kernel hyperparameters:**
- $\sigma_f^2$: **signal variance** — controls the amplitude (vertical scale) of functions.
- $\ell$: **lengthscale** — controls how quickly functions vary. Small $\ell$ = wiggly; large $\ell$ = smooth.
- $\sigma^2$: **noise variance** — the observation noise (not a kernel hyperparameter but appears in inference).

### 4.3 Building New Kernels

Valid kernels can be combined to produce new valid kernels:

| Operation | New kernel | Intuition |
|-----------|-----------|-----------|
| $k_1+k_2$ | Valid | Model = component 1 + component 2 |
| $k_1\cdot k_2$ | Valid | Both structures must hold simultaneously |
| $\alpha k_1$, $\alpha>0$ | Valid | Scale the amplitude |
| $k(h(x),h(x'))$ | Valid for any $h$ | Compose with a feature transformation |
| $\exp(k_1)$ | Valid | |

This allows building structured kernels: e.g., $k = k_{\text{linear}} + k_{\text{periodic}} + k_{\text{RBF}}$ models a trend plus periodic fluctuations plus smooth noise.

---

## 5. GP Regression: Step-by-Step

### 5.1 Prior over Function Values

For training inputs $X=(x_1,\ldots,x_N)^\top$, the GP prior over function values $f(X)=(f(x_1),\ldots,f(x_N))^\top$ is:

$$f(X) \sim \mathcal{N}(\mathbf{0},\; K(X,X))$$

where $K(X,X)\in\mathbb{R}^{N\times N}$ is the **Gram matrix** with $[K(X,X)]_{ij}=k(x_i,x_j)$. (We use zero mean function for simplicity; all results generalise trivially to non-zero $m$.)

### 5.2 Likelihood (Noisy Observations)

We observe noisy outputs $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i\overset{\text{iid}}{\sim}\mathcal{N}(0,\sigma^2)$. In matrix form:

$$Y \mid f(X) \sim \mathcal{N}(f(X),\; \sigma^2 I) \quad\Rightarrow\quad Y \sim \mathcal{N}(\mathbf{0},\; K(X,X)+\sigma^2 I)$$

The $\sigma^2 I$ term adds noise variance to the diagonal of the kernel matrix.

### 5.3 Joint Distribution of Training and Test Points

For test inputs $X^*=(x_1^*,\ldots,x_M^*)^\top$, the **joint prior** over $(Y, f(X^*))$ is Gaussian. By the GP prior:

$$\begin{pmatrix}Y \\ f(X^*)\end{pmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\; \begin{pmatrix}K(X,X)+\sigma^2 I & K(X,X^*) \\ K(X^*,X) & K(X^*,X^*)\end{pmatrix}\right)$$

where:
- $K(X,X^*)\in\mathbb{R}^{N\times M}$: training-test cross-covariance; $[K(X,X^*)]_{ij}=k(x_i,x_j^*)$.
- $K(X^*,X)=K(X,X^*)^\top\in\mathbb{R}^{M\times N}$: test-training cross-covariance.
- $K(X^*,X^*)\in\mathbb{R}^{M\times M}$: test-test prior covariance.

### 5.4 Posterior by Gaussian Conditioning

We use the standard formula for conditioning a joint Gaussian. If

$$\begin{pmatrix}a\\b\end{pmatrix}\sim\mathcal{N}\!\left(\begin{pmatrix}\mu_a\\\mu_b\end{pmatrix}, \begin{pmatrix}\Sigma_{aa}&\Sigma_{ab}\\\Sigma_{ba}&\Sigma_{bb}\end{pmatrix}\right)$$

then $b\mid a\sim\mathcal{N}(\mu_b+\Sigma_{ba}\Sigma_{aa}^{-1}(a-\mu_a),\; \Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab})$.

Applying with $a=Y$, $b=f(X^*)$, $\mu_a=\mu_b=0$:

$$\boxed{P(f(X^*)\mid Y, X, X^*) = \mathcal{N}(\mu^*,\; \Sigma^*)}$$

$$\mu^* = K(X^*,X)\big(K(X,X)+\sigma^2 I\big)^{-1}Y$$

$$\Sigma^* = K(X^*,X^*) - K(X^*,X)\big(K(X,X)+\sigma^2 I\big)^{-1}K(X,X^*)$$

**Reading the posterior mean:** $\mu^*_j = \sum_{i=1}^N \alpha_i k(x_i,x_j^*)$ where $\alpha = (K+\sigma^2I)^{-1}Y$. Each training point contributes to the prediction at $x_j^*$ in proportion to its kernel similarity $k(x_i,x_j^*)$ and its weight $\alpha_i$.

**Reading the posterior variance:** $\Sigma^*$ starts from the prior covariance $K(X^*,X^*)$ and subtracts $K(X^*,X)(K+\sigma^2I)^{-1}K(X,X^*)$ — the reduction due to observing $Y$. Uncertainty decreases monotonically as more data are added.

### 5.5 Predictive Distribution for Noisy Observations

For a noisy test observation $y^*=f(x^*)+\varepsilon^*$, add $\sigma^2$ to the predictive variance:

$$P(y^*\mid Y, X, x^*) = \mathcal{N}\!\Big(\mu^*,\;\; \Sigma^* + \sigma^2 I\Big)$$

---

## 6. Marginal Likelihood and Hyperparameter Learning

The kernel hyperparameters $\psi=(\ell, \sigma_f^2, \sigma^2)$ are not known in advance. The Bayesian approach is to select them by **maximising the marginal likelihood** (evidence):

$$p(Y\mid X,\psi) = \int p(Y\mid f(X),\sigma^2)\,p(f(X)\mid X,\psi)\,df(X) = \mathcal{N}(Y\mid \mathbf{0},\; K_\psi+\sigma^2 I)$$

Taking the log:

$$\log p(Y\mid X,\psi) = \underbrace{-\frac{1}{2}Y^\top(K+\sigma^2 I)^{-1}Y}_{\text{data fit}} \underbrace{-\frac{1}{2}\log\det(K+\sigma^2 I)}_{\text{complexity penalty}} - \frac{N}{2}\log 2\pi$$

**The two competing terms:**
- **Data fit** prefers hyperparameters that make $Y$ likely under the model. Larger $\sigma_f^2$ and smaller $\ell$ (wigglier functions) usually improve fit.
- **Complexity penalty** (log-determinant) penalises models that are flexible enough to explain many possible datasets. Smaller $\ell$ increases the determinant, adding a cost.

The marginal likelihood automatically balances these — this is the **Bayesian Occam's Razor**. Optimise by gradient ascent:

$$\hat{\psi} = \arg\max_\psi\;\log p(Y\mid X,\psi)$$

The gradients $\partial\log p/\partial\psi_i$ involve $\text{tr}[(K+\sigma^2I)^{-1}\partial K/\partial\psi_i]$ and can be computed analytically for standard kernels.

---

## 7. Computational Considerations

### 7.1 Cholesky Factorisation

Computing $(K+\sigma^2I)^{-1}Y$ by direct matrix inversion is numerically unstable. The standard approach is the **Cholesky factorisation**:

$$K+\sigma^2 I = LL^\top$$

where $L$ is lower triangular. Then $(K+\sigma^2I)^{-1}Y = L^{-\top}L^{-1}Y$, computed by two triangular solves — numerically stable and efficient.

**Complexity:** $O(N^3)$ for the Cholesky factorisation, $O(N^2)$ storage for $L$. This is the bottleneck for large datasets.

### 7.2 Scalable Approximations

| Method | Idea | Cost |
|--------|------|------|
| **Sparse GPs / Inducing points** (Snelson & Ghahramani, 2006) | Approximate $K(X,X)$ using $M\ll N$ inducing points | $O(NM^2)$ |
| **Random Fourier Features** (Rahimi & Recht, 2007) | Approximate shift-invariant kernels by $D$ random features | $O(ND+D^3)$ |
| **State-space / Kalman** (Solin & Särkkä, 2014) | For time series with Matérn kernels: equivalent to Kalman filter | $O(N)$ |
| **Deep kernel learning** (Wilson et al., 2016) | Learn a neural network feature map; kernel acts on learned features | Varies |

---

## 8. GP vs. Other Methods

| | Bayesian Lin. Reg. | GP | Neural Network |
|--|--|--|--|
| **Prior over** | Weight vector $w\in\mathbb{R}^p$ | Functions $f$ | Weight vector (but optimised, not integrated) |
| **Posterior** | Gaussian (exact) | GP (exact, $O(N^3)$) | Approximate (variational or MCMC) |
| **Uncertainty** | Calibrated | Calibrated | Often overconfident |
| **Nonlinearity** | Only via feature map | Via kernel choice | Automatic (deep architecture) |
| **Scalability** | $O(Np^2+p^3)$ | $O(N^3)$ exact | $O(\text{params}\times\text{data})$ |
| **Interpretability** | High (weight coefficients) | High (kernel = prior structure) | Low |

---

## 9. Summary

```
GP prior:
  f ~ GP(0, k(x,x'))
  Any finite set (f(x₁),...,f(xₙ)) is jointly Gaussian

Inference (GP regression):
  Observed: Y = f(X) + ε,   ε ~ N(0, σ²I)
  Prior:    f(X) ~ N(0, K(X,X))
  Joint:    [Y; f(X*)] ~ N(0, [[K+σ²I, K(X,X*)]; [K(X*,X), K(X*,X*)]])

Posterior by Gaussian conditioning:
  μ* = K(X*,X)(K+σ²I)⁻¹Y
  Σ* = K(X*,X*) - K(X*,X)(K+σ²I)⁻¹K(X,X*)

Hyperparameter learning:
  Maximise log p(Y|X,ψ) = N(Y|0, K+σ²I) via gradient ascent
  Balances data fit vs. complexity (Bayesian Occam's Razor)

Computation:
  Use Cholesky factorisation for numerical stability: K+σ²I = LLᵀ
  Cost: O(N³) exact; O(NM²) with M inducing points
```

---

## 10. References

1. **Rasmussen, C. E. & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press. [[Free PDF](http://www.gaussianprocess.org/gpml/)] [The definitive reference; §2 (GP regression), §4 (covariance functions), §5 (model selection).]
2. **MacKay, D. J. C.** (1998). Introduction to Gaussian processes. In Bishop, C. M. (ed.), *Neural Networks and Machine Learning*, NATO ASI Series, pp. 133–165. Springer.
3. **Neal, R. M.** (1996). *Bayesian Learning for Neural Networks*. Springer. [Shows infinite-width Bayesian NNs converge to GPs; Ch. 2.]
4. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §6.4 (GP regression), §6.4.2 (learning the hyperparameters).
5. **Snelson, E. & Ghahramani, Z.** (2006). Sparse Gaussian processes using pseudo-inputs. *NeurIPS 2005*. [Inducing-point sparse GP approximation.]
6. **Rahimi, A. & Recht, B.** (2007). Random features for large-scale kernel machines. *NeurIPS 2007*.
7. **Jacot, A., Gabriel, F., & Hongler, C.** (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS 2018*. [Infinite-width deep nets as GPs.]
8. **Stein, M. L.** (1999). *Interpolation of Spatial Data: Some Theory for Kriging*. Springer. [Matérn kernels and their properties; theoretical motivation for preferring Matérn over RBF.]

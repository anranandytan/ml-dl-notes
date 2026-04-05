# DL 01 — Bayesian Regression

> **Keywords:** Bayesian inference, prior, posterior, conjugate prior, completing the square, predictive distribution, epistemic uncertainty, aleatoric uncertainty, marginal likelihood, kernel trick

---

## 1. What Problem Are We Solving?

Imagine you measure the height of 3 students and want to predict the height of a 4th. Ordinary least squares gives you *one* line of best fit and then acts completely confident — it will predict the same number whether you trained on 3 points or 300,000 points.

This is a problem. **When you have very little data, you should be less sure of your predictions.** Bayesian regression is the principled way to carry uncertainty all the way through to the final answer.

**The core shift:** instead of finding *one* weight vector $w$ that "fits best", we maintain a **full probability distribution** over all possible $w$ values. Each $w$ gets a probability proportional to how well it explains the data. When predicting, we average over all of them.

This buys us four things:

| Benefit | What it means in practice |
|---------|--------------------------|
| **Uncertainty quantification** | Predictions come with honest error bars. Error bars widen where data is sparse. |
| **Automatic regularisation** | The prior encodes "weights should not be huge" — no need to tune $\lambda$ by cross-validation. |
| **No overfitting in the predictive sense** | Averaging over all plausible $w$'s prevents any single bad fit from dominating. |
| **Model comparison** | The marginal likelihood $p(\text{data})$ naturally penalises over-complex models (Bayesian Occam's Razor). |

---

## 2. Historical Notes

| Year | Event |
|------|-------|
| 1763 | Thomas Bayes' posthumous essay introduces Bayes' theorem. |
| 1812 | Laplace independently develops Bayesian probability and applies it to regression-like inverse problems. |
| 1970s | Conjugate prior analysis is systematised; analytic posterior computations become textbook material. |
| 1992 | MacKay's *evidence framework* provides a principled way to select hyperparameters from data alone, making Bayesian regression practical. |
| 1996–1998 | Rasmussen & Williams reinterpret kernelised Bayesian linear regression as **Gaussian Process Regression**. |

---

## 3. MAP Estimation: Halfway Between MLE and Full Bayes

Before going fully Bayesian, it helps to see a simpler step: **Maximum A Posteriori (MAP)** estimation. MAP still finds *one* weight vector, but it takes the prior into account.

By Bayes' theorem:

$$P(w \mid \text{Data}) \propto \underbrace{P(\text{Data} \mid w)}_{\text{likelihood}} \cdot \underbrace{P(w)}_{\text{prior}}$$

**MAP** finds the most probable weight under the posterior — the peak, not the full distribution:

$$w_{\text{MAP}} = \arg\max_w\; P(\text{Data} \mid w)\, P(w)$$

Taking the log (which is monotone, so it does not change which $w$ wins) turns the product into a sum:

$$w_{\text{MAP}} = \arg\max_w\;\Big[\underbrace{\log P(\text{Data} \mid w)}_{\text{fit the data}} + \underbrace{\log P(w)}_{\text{prefer simple }w}\Big]$$

### 3.1 Gaussian Prior = Ridge Regression

If the noise is Gaussian and the prior is $P(w) = \mathcal{N}(0, \tau^2 I)$ (a zero-mean Gaussian — we expect weights to be small):

$$\log P(\text{Data} \mid w) = -\frac{1}{2\sigma^2}\|Y - Xw\|^2 + \text{const}$$

$$\log P(w) = -\frac{1}{2\tau^2}\|w\|_2^2 + \text{const}$$

The MAP objective is:

$$w_{\text{MAP}} = \arg\min_w\;\left[\|Y - Xw\|^2 + \underbrace{\frac{\sigma^2}{\tau^2}}_{=\,\lambda}\|w\|_2^2\right]$$

This is exactly **Ridge regression** with $\lambda = \sigma^2/\tau^2$. The Gaussian prior is responsible for the $\ell_2$ penalty. **A tighter prior (small $\tau^2$) means stronger regularisation (large $\lambda$).**

### 3.2 Laplace Prior = Lasso

If instead $P(w_j) \propto \exp(-|w_j|/b)$ (a Laplace prior — sharply peaked at zero with heavy tails):

$$w_{\text{MAP}} = \arg\min_w\;\left[\|Y - Xw\|^2 + \frac{2\sigma^2}{b}\|w\|_1\right]$$

This is **Lasso**. The Laplace distribution has a sharp spike at zero, which encourages many weights to be pushed to *exactly* zero — the origin of sparsity.

**Geometric intuition:** The Laplace ball ($\ell_1$ ball) is diamond-shaped with corners on the axes; the quadratic loss ellipse tends to touch it at a corner, forcing some coordinates to zero. The Gaussian ball ($\ell_2$) is smooth; the ellipse touches it off the axes, so no coordinate hits exactly zero.

### 3.3 Why MAP Is Still Not Enough

MAP collapses the posterior to a single point. It tells you *where* the posterior peaks but not *how wide* it is. Two datasets can give the same MAP estimate but wildly different uncertainties. Full Bayesian inference carries the entire distribution forward.

---

## 4. Full Bayesian Linear Regression

### 4.1 Setup

Given $N$ training pairs $\{(x_i, y_i)\}$, with $x_i\in\mathbb{R}^p$ and $y_i\in\mathbb{R}$:

$$X = \begin{pmatrix}x_1^\top\\\vdots\\x_N^\top\end{pmatrix}\in\mathbb{R}^{N\times p}, \qquad Y = \begin{pmatrix}y_1\\\vdots\\y_N\end{pmatrix}\in\mathbb{R}^N$$

**Generative story (how we imagine the data was created):**

$$y_i = \underbrace{w^\top x_i}_{\text{signal}} + \underbrace{\varepsilon_i}_{\text{noise}}, \qquad \varepsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$$

**Prior belief about $w$ (before seeing data):**

$$P(w) = \mathcal{N}(0, \Sigma_p)$$

Think of $\Sigma_p$ as how "spread out" we expect the weights to be. Large $\Sigma_p$ = weak regularisation (weights can be anything); small $\Sigma_p$ = strong regularisation (weights should be near zero).

### 4.2 The Two Goals

**Goal 1 — Learning (posterior):** Update our belief about $w$ given the data.

$$P(w \mid \mathcal{D}) = \frac{P(Y \mid X, w)\, P(w)}{P(Y \mid X)} \propto P(Y \mid X, w)\, P(w)$$

**Goal 2 — Predicting (predictive distribution):** For a new input $x^*$, compute the full distribution over $y^*$:

$$P(y^* \mid \mathcal{D}, x^*) = \int P(y^* \mid w, x^*)\; P(w \mid \mathcal{D})\; dw$$

**Why integrate?** Think of it as taking a poll. Instead of asking one expert for a single prediction, you ask every plausible $w$ for its prediction and average, weighted by how likely each $w$ is. The result is more robust than any single estimate.

---

## 5. Deriving the Posterior Step by Step

### 5.1 Why Does a Gaussian Times a Gaussian Stay Gaussian?

This is the key algebraic miracle that makes Bayesian linear regression tractable.

**Conjugacy.** The Gaussian family is *self-conjugate* for Gaussian likelihoods: if the prior is Gaussian and the likelihood is Gaussian in $w$, then the posterior is also Gaussian. In general, a prior $P(w)$ is called *conjugate* to a likelihood $P(\text{data}|w)$ when the posterior is in the same family as the prior.

Here, the likelihood $P(Y|X,w)$ is Gaussian in $w$ (because $y_i - w^\top x_i$ appears inside an exponential squared), and the prior $P(w)$ is Gaussian. Both exponents are quadratic in $w$, so their product is also an exponential of a quadratic in $w$ — still a Gaussian. The whole derivation below is just *identifying which Gaussian* it is.

### 5.2 Writing Down the Likelihood

Since observations are independent given $w$:

$$P(Y \mid X, w) = \prod_{i=1}^N \mathcal{N}(y_i \mid w^\top x_i, \sigma^2) = \mathcal{N}(Y \mid Xw,\; \sigma^2 I)$$

In matrix form:

$$P(Y \mid X, w) = \frac{1}{(2\pi)^{N/2}\sigma^N}\exp\!\left(-\frac{1}{2\sigma^2}(Y-Xw)^\top(Y-Xw)\right)$$

**Important:** $\sigma^2 I$ is the *covariance* (how spread out the noise is). The *precision* (inverse covariance) is $\sigma^{-2}I$. These are different things — the original notes sometimes confused them.

### 5.3 Log-Posterior and Expanding the Quadratic

The log-posterior (dropping everything constant in $w$) is:

$$\log P(w \mid \mathcal{D}) \;\propto\; -\frac{1}{2\sigma^2}(Y-Xw)^\top(Y-Xw) - \frac{1}{2}w^\top\Sigma_p^{-1}w$$

**Expanding $(Y-Xw)^\top(Y-Xw)$ carefully** (this step trips people up):

$$(Y-Xw)^\top(Y-Xw) = Y^\top Y - Y^\top(Xw) - (Xw)^\top Y + (Xw)^\top(Xw)$$

- $Y^\top Y$: just a number, constant in $w$, drop it.
- $Y^\top(Xw) = w^\top X^\top Y$: a scalar. Note: every scalar equals its own transpose, so $(Y^\top Xw)^\top = w^\top X^\top Y$. These two middle terms are **the same scalar**, so they add to give $2\,w^\top X^\top Y$.
- $(Xw)^\top(Xw) = w^\top X^\top Xw$: a quadratic form.

Result:

$$(Y-Xw)^\top(Y-Xw) = \underbrace{Y^\top Y}_{\text{drop}} - 2\,w^\top X^\top Y + w^\top X^\top Xw$$

Substituting and dropping the constant:

$$\log P(w|\mathcal{D}) \;\propto\; \frac{1}{\sigma^2}\,w^\top X^\top Y - \frac{1}{2}\underbrace{\left(\frac{1}{\sigma^2}X^\top X + \Sigma_p^{-1}\right)}_{:=\,A}w^\top A w$$

Wait — let me write this cleanly: grouping the two quadratic terms in $w$:

$$\log P(w|\mathcal{D}) \;\propto\; \underbrace{\frac{1}{\sigma^2}\,w^\top X^\top Y}_{\text{linear in }w} - \frac{1}{2}\,w^\top\underbrace{\left(\frac{1}{\sigma^2}X^\top X + \Sigma_p^{-1}\right)}_{:=\,A}w$$

### 5.4 Completing the Square

A Gaussian $\mathcal{N}(\mu_w, A^{-1})$ has log-density (up to constants):

$$-\frac{1}{2}(w-\mu_w)^\top A(w-\mu_w) = -\frac{1}{2}w^\top Aw + w^\top A\mu_w - \frac{1}{2}\mu_w^\top A\mu_w$$

Ignoring the last constant term and matching with what we have:

- **Quadratic coefficient:** matrix $A = \dfrac{1}{\sigma^2}X^\top X + \Sigma_p^{-1}$ ✓
- **Linear coefficient:** $A\mu_w = \dfrac{1}{\sigma^2}X^\top Y \;\Rightarrow\; \mu_w = \dfrac{1}{\sigma^2}A^{-1}X^\top Y$ ✓

### 5.5 The Posterior

$$\boxed{P(w \mid \mathcal{D}) = \mathcal{N}(\mu_w,\; \Sigma_w)}$$

$$A = \frac{1}{\sigma^2}X^\top X + \Sigma_p^{-1}, \qquad \Sigma_w = A^{-1}, \qquad \mu_w = \frac{1}{\sigma^2}A^{-1}X^\top Y$$

**Reading these formulas intuitively:**

- $A$ is the **posterior precision** (inverse covariance). It equals *data precision* $\frac{1}{\sigma^2}X^\top X$ plus *prior precision* $\Sigma_p^{-1}$. Information adds in precision space — more data or a tighter prior both concentrate the posterior.
- $\Sigma_w = A^{-1}$: as $N$ grows, $X^\top X$ grows, $A$ grows, $\Sigma_w$ shrinks — we become more certain.
- $\mu_w$: as $N\to\infty$, the likelihood dominates the prior and $\mu_w$ converges to the ordinary least-squares estimate.
- $\mu_w$: as $N\to 0$, the prior dominates and $\mu_w\to 0$ (the prior mean).

### 5.6 Worked 1D Example

Let $p=1$, $\sigma^2=1$, $\Sigma_p=\tau^2=4$ (weak prior), training data $X=[1,2,3]^\top$, $Y=[1.2, 2.1, 2.9]^\top$.

$$A = \frac{1}{1}X^\top X + \frac{1}{4} = (1+4+9) + 0.25 = 14.25$$

$$\mu_w = \frac{1}{1}\cdot\frac{1}{14.25}X^\top Y = \frac{1\cdot1.2+2\cdot2.1+3\cdot2.9}{14.25} = \frac{13.9}{14.25}\approx 0.976$$

$$\Sigma_w = \frac{1}{14.25}\approx 0.070$$

So $P(w|\mathcal{D})\approx\mathcal{N}(0.976, 0.070)$. The OLS estimate (ignoring the prior) would be $w_{\text{OLS}} = (X^\top X)^{-1}X^\top Y = 13.9/14 \approx 0.993$. The posterior mean $0.976$ is slightly pulled toward zero by the prior — a mild regularisation effect.

---

## 6. Predictive Distribution

### 6.1 Distribution Over $f(x^*)$

A new test input $x^*$ gives prediction $f(x^*) = (x^*)^\top w$. Since $w\sim\mathcal{N}(\mu_w,\Sigma_w)$ and this is a linear transformation:

> **Fact:** If $w\sim\mathcal{N}(\mu,\Sigma)$ and $f = c^\top w$ for a fixed vector $c$, then $f\sim\mathcal{N}(c^\top\mu,\; c^\top\Sigma c)$.

So:

$$f(x^*)\sim\mathcal{N}\!\Big((x^*)^\top\mu_w,\; (x^*)^\top\Sigma_w x^*\Big)$$

### 6.2 Adding Observation Noise

The actual observation $y^* = f(x^*) + \varepsilon^*$ adds independent noise $\varepsilon^*\sim\mathcal{N}(0,\sigma^2)$. The variance of the sum of two independent random variables equals the sum of their variances:

$$\boxed{P(y^* \mid \mathcal{D}, x^*) = \mathcal{N}\!\Big((x^*)^\top\mu_w,\;\; \underbrace{(x^*)^\top\Sigma_w x^*}_{\text{model uncertainty}} + \underbrace{\sigma^2}_{\text{noise}}\Big)}$$

### 6.3 Two Kinds of Uncertainty

| Name | Symbol | What it is | Can more data help? |
|------|--------|-----------|-------------------|
| **Epistemic** (model) | $(x^*)^\top\Sigma_w x^*$ | We don't know the true $w$ | **Yes** — shrinks to 0 as $N\to\infty$ |
| **Aleatoric** (noise) | $\sigma^2$ | The world is noisy | **No** — irreducible |

**Key insight:** Epistemic uncertainty is large where training data is sparse (the model hasn't seen similar points) and small where data is dense. Aleatoric uncertainty is the same everywhere — it is a property of the problem, not the model.

---

## 7. Selecting Hyperparameters: The Marginal Likelihood

So far we treated $\sigma^2$ and $\Sigma_p$ as known. In practice, they are hyperparameters we need to choose. Bayesian regression provides a principled way: **maximise the marginal likelihood** (also called the *evidence*).

### 7.1 What Is the Marginal Likelihood?

Integrate out $w$ to get the probability of the data under the model alone:

$$p(Y \mid X, \sigma^2, \Sigma_p) = \int P(Y \mid X, w, \sigma^2)\, P(w \mid \Sigma_p)\, dw$$

Since both factors are Gaussian in $w$, this integral has a closed form. Using the Gaussian marginalisation formula:

$$p(Y \mid X, \sigma^2, \Sigma_p) = \mathcal{N}(Y \mid 0,\; \sigma^2 I + X\Sigma_p X^\top)$$

The log marginal likelihood is:

$$\log p(Y \mid X) = -\frac{1}{2}Y^\top(\sigma^2 I + X\Sigma_p X^\top)^{-1}Y - \frac{1}{2}\log\det(\sigma^2 I + X\Sigma_p X^\top) - \frac{N}{2}\log(2\pi)$$

The three terms have clear meanings:
1. **Data fit** — how well the model explains $Y$.
2. **Complexity penalty** — $\log\det$ grows when the model is flexible (large $\Sigma_p$), penalising unnecessary complexity.
3. **Constant** — independent of hyperparameters.

### 7.2 Optimising the Evidence

$$(\hat{\sigma}^2, \hat{\Sigma}_p) = \arg\max_{\sigma^2, \Sigma_p}\;\log p(Y \mid X, \sigma^2, \Sigma_p)$$

This is a smooth optimisation problem that can be solved with gradient ascent. It is the **Empirical Bayes** or **Type-II Maximum Likelihood** approach. MacKay (1992) showed that this automatically balances fit vs. complexity — a model that is too flexible will be penalised by the $\log\det$ term.

---

## 8. Kernel Bayesian Linear Regression and Connection to GPs

### 8.1 Feature Maps for Nonlinearity

The linear model $f(x) = w^\top x$ can only fit linear functions. To model curved relationships, apply a feature map $\phi:\mathbb{R}^p\to\mathcal{H}$ first:

$$f(x) = w^\top\phi(x)$$

All the derivations above go through unchanged, with $x_i$ replaced by $\phi(x_i)$ and $X$ replaced by the feature matrix $\Phi$.

### 8.2 The Kernel Trick

When $\phi$ maps to a very high (or infinite) dimensional space, we can never form $\Phi$ explicitly. But notice that the posterior mean and predictive formulas depend on $\phi$ only through **inner products** $\phi(x)^\top\Sigma_p\phi(x')$. Define:

$$k(x,x') := \phi(x)^\top\Sigma_p\phi(x')$$

This is the **kernel function**. It encodes similarity between inputs in the feature space. We can evaluate $k(x,x')$ directly — without ever computing $\phi$ — as long as $k$ is a valid positive semi-definite function.

**Common kernels:**

| Kernel | Formula | Feature space |
|--------|---------|--------------|
| Linear | $k(x,x')=x^\top x'$ | $\mathbb{R}^p$ (identity) |
| Polynomial | $k(x,x')=(x^\top x'+1)^d$ | All monomials up to degree $d$ |
| RBF (Gaussian) | $k(x,x')=\exp(-\|x-x'\|^2/2\ell^2)$ | Infinite-dimensional |

### 8.3 The GP Predictive Formula

Using the Gram matrix $K\in\mathbb{R}^{N\times N}$ with $K_{ij}=k(x_i,x_j)$ and the test vector $k^*\in\mathbb{R}^N$ with $k^*_i=k(x_i,x^*)$, the predictive distribution becomes:

$$P(y^* \mid \mathcal{D}, x^*) = \mathcal{N}\!\Big(\underbrace{k^{*\top}(K+\sigma^2 I)^{-1}Y}_{\text{posterior mean}},\;\; \underbrace{k(x^*,x^*) - k^{*\top}(K+\sigma^2 I)^{-1}k^*}_{\text{posterior variance}} + \sigma^2\Big)$$

This is the **Gaussian Process Regression** formula. See `DL_02_GaussianProcess.md` for a full treatment.

### 8.4 Two Equivalent Views

| | Weight-space | Function-space (GP) |
|--|--|--|
| **What we put a prior over** | Weight vector $w\in\mathcal{H}$ | Function $f:\mathbb{R}^p\to\mathbb{R}$ |
| **Prior** | $w\sim\mathcal{N}(0,\Sigma_p)$ | $f\sim\mathcal{GP}(0, k(\cdot,\cdot))$ |
| **Computation scales with** | $\dim(\mathcal{H})$ | $N$ (number of data points) |
| **Preferred when** | Feature space is finite-dimensional | Feature space is infinite-dimensional (RBF kernel) |

These two views are mathematically identical — they give the same predictive distribution. The GP view simply avoids working in the (possibly infinite-dimensional) weight space.

---

## 9. Summary

```
Step 1 — Set up the model:
  y_i = wᵀx_i + ε_i,   ε_i ~ N(0, σ²)
  Prior:  P(w) = N(0, Σ_p)

Step 2 — Posterior (Bayes' theorem + Gaussian conjugacy):
  P(w|D) = N(μ_w, Σ_w)
  A = (1/σ²)XᵀX + Σ_p⁻¹
  Σ_w = A⁻¹,   μ_w = (1/σ²)A⁻¹XᵀY

Step 3 — Predictive distribution:
  P(y*|D, x*) = N((x*)ᵀμ_w,  (x*)ᵀΣ_w x* + σ²)
                              ↑ epistemic     ↑ aleatoric

Step 4 — Hyperparameter selection (optional):
  Maximise log p(Y|X, σ², Σ_p) = N(Y|0, σ²I + XΣ_pXᵀ) over σ², Σ_p

Step 5 — Nonlinearity via kernels:
  Replace Xᵀx with Gram matrix K,  get GP regression formula
```

---

## 10. References

1. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §3.3 (Bayesian linear regression), §3.4 (Predictive distribution), §3.5 (Evidence approximation / hyperparameter selection).
2. **Rasmussen, C. E. & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press. §2.1 (Weight-space view), §2.2 (Function-space view). [[Free PDF](http://www.gaussianprocess.org/gpml/)]
3. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. §7.6 (Bayesian linear regression), §14.4 (Gaussian processes).
4. **MacKay, D. J. C.** (1992). Bayesian interpolation. *Neural Computation*, 4(3), 415–447. [Introduces the evidence framework for hyperparameter selection via marginal likelihood maximisation.]
5. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. §3.4 (Ridge and Lasso as MAP), §3.8 (Bayesian interpretation of regression).
6. **Schölkopf, B. & Smola, A. J.** (2002). *Learning with Kernels*. MIT Press. [Representer theorem; kernel methods.]

# 03 — Regression

> **Keywords:** ordinary least squares, hat matrix, ridge regression, regularisation path, Lasso, sparsity, Bayesian interpretation

---

## 1. Ordinary Least Squares (OLS)

### 1.1 Setup

Given data $(X,Y)$ with $X\in\mathbb{R}^{n\times p}$ and $Y\in\mathbb{R}^n$, fit a linear model $f_\lambda(x)=\lambda^\top x$:

$$\min_\lambda\;F(\lambda) = \|Y-X\lambda\|_2^2 = \sum_{i=1}^n(y_i-\lambda^\top x_i)^2$$

### 1.2 Gradient and Closed-Form Solution

**Gradient:**

$$\nabla F(\lambda) = -2X^\top(Y-X\lambda) = -2(X^\top Y - X^\top X\lambda)$$

Setting to zero: $X^\top X\lambda^* = X^\top Y$.

**OLS solution** (when $X^\top X$ is invertible):

$$\lambda^* = (X^\top X)^{-1}X^\top Y$$

**Fitted values (hat matrix):**

$$\hat{Y} = X\lambda^* = X(X^\top X)^{-1}X^\top Y =: HY$$

The matrix $H=X(X^\top X)^{-1}X^\top$ is the **hat matrix** — it projects $Y$ onto the column space of $X$.

**Bayesian interpretation:** OLS is equivalent to MAP estimation with a flat (improper) prior, or equivalently, MLE under Gaussian noise $Y\sim\mathcal{N}(X\lambda,\sigma^2 I)$.

---

## 2. Ridge Regression ($\ell_2$)

### 2.1 Objective

Ridge regression adds an $\ell_2$ penalty on the coefficients:

$$\min_\lambda\;F(\lambda) = \frac{1}{n}\sum_{i=1}^n(y_i-\lambda^\top x_i)^2 + C\|\lambda\|_2^2 = \|Y-X\lambda\|_2^2 + C\|\lambda\|_2^2$$

The penalty $C\|\lambda\|_2^2$ shrinks all coefficients toward zero, preventing overfitting when $p$ is large or $X^\top X$ is near-singular.

### 2.2 Closed-Form Solution

**Gradient:**

$$\nabla F(\lambda) = -2X^\top(Y-X\lambda) + 2C\lambda = 2(-X^\top Y + X^\top X\lambda + C\lambda) = 2(-(X^\top Y) + (X^\top X+CI)\lambda)$$

Setting to zero:

$$X^\top Y = (X^\top X+CI)\lambda^*$$

$$\boxed{\lambda^* = (X^\top X+CI)^{-1}X^\top Y}$$

The matrix $(X^\top X+CI)$ is always invertible for $C>0$ (the eigenvalues of $X^\top X$ are $\geq 0$, so adding $C>0$ to each makes them all $>0$). This is one reason ridge is preferred to OLS when $X^\top X$ is singular.

### 2.3 Bayesian Interpretation

Ridge regression is the MAP estimator under:
- Gaussian likelihood: $Y|X,\beta\sim\mathcal{N}(X\beta,\sigma^2 I)$
- Gaussian prior: $\beta\sim\mathcal{N}(0,\tau^2 I)$

The posterior is:

$$P(\beta|Y,X) \propto P(Y|X,\beta)\,P(\beta) \propto \exp\!\left(-\frac{1}{2\sigma^2}\|Y-X\beta\|_2^2 - \frac{1}{2\tau^2}\|\beta\|_2^2\right)$$

Taking $-\log$:

$$-\log P(\beta|Y,X) = \frac{1}{\sigma^2}\|Y-X\beta\|_2^2 + \frac{1}{\tau^2}\|\beta\|_2^2 + \text{const}$$

This matches the ridge objective with $C=\sigma^2/\tau^2$. Strong prior belief that $\beta$ is small (small $\tau^2$) corresponds to strong regularisation (large $C$).

### 2.4 Regularisation Path

As $C$ varies from 0 to $\infty$:
- **$C=0$:** OLS solution (no regularisation).
- **$C\to\infty$:** all coefficients shrink to 0.

**Visualisation — regularisation path:**
- Horizontal axis: regularisation constant $C$.
- Vertical axis: coefficient values $\lambda^*_j$.
- Each curve shows how one coefficient changes as $C$ increases.

---

## 3. Lasso Regression ($\ell_1$)

### 3.1 Objective

$$\min_\lambda\;F_1(\lambda) = \frac{1}{n}\sum_{i=1}^n(y_i-f_\lambda(x_i))^2 + C\|\lambda\|_1$$

Unlike ridge, the $\ell_1$ penalty **does not** have a closed-form solution (because $\|\lambda\|_1$ is not differentiable at zero). Algorithms such as coordinate descent, ADMM, or proximal gradient methods are used.

### 3.2 Sparsity from the $\ell_1$ Penalty

**Geometric intuition.** The level sets of $\|\lambda\|_1$ are diamond-shaped (in 2D). The level sets of the squared loss are ellipses. The optimum of the constrained version:

$$\min_\lambda \|Y-X\lambda\|_2^2 \quad\text{s.t.}\quad \|\lambda\|_1\leq t$$

occurs where the ellipse first touches the diamond. The corners of the diamond lie on the coordinate axes — points where some $\lambda_j=0$ exactly. This is why $\ell_1$ regularisation produces **sparse** solutions.

In contrast, the $\ell_2$ ball is smooth (sphere), so the ellipse touches it at a point off the axes, giving non-zero (but small) coefficients for all features.

### 3.3 When to Use Ridge vs. Lasso

| | Ridge ($\ell_2$) | Lasso ($\ell_1$) |
|--|--|--|
| **Solution** | Closed form | Requires iterative algorithms |
| **Sparsity** | All coefficients shrunk, none exactly zero | Many coefficients driven to exactly zero |
| **Interpretation** | All features contribute | Automatic feature selection |
| **When to prefer** | All features are relevant | Only a few features are truly predictive |
| **Stability** | More stable (smooth penalty) | Less stable when features are correlated |

---

## 4. Summary

```
OLS:    λ* = (XᵀX)⁻¹XᵀY          [may be singular, high variance when p ≈ n]
Ridge:  λ* = (XᵀX + CI)⁻¹XᵀY     [always invertible; shrinks all coefficients]
Lasso:  no closed form; sparse solution via iterative methods

Bayesian view:
  Ridge ↔ Gaussian prior on λ (MAP)
  Lasso ↔ Laplace prior on λ  (MAP)
  (See DL_01_BayesianRegression.md for the full derivation)

Kernel ridge regression: replace XᵀX with Gram matrix K
  (See ML_06_Kernels.md)
```

---

## 5. References

1. **Hoerl, A. E. & Kennard, R. W.** (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. [Original ridge regression paper.]
2. **Tibshirani, R.** (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288. [Original Lasso paper.]
3. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Ch. 3 (Linear regression, ridge, Lasso).
4. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §3.1 (Linear basis function models), §3.3 (Bayesian linear regression).

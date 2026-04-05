# 05 — Logistic Regression, Convex Optimisation, and SVM

> **Keywords:** logistic regression, odds ratio, MLE, convex function, Lagrangian, KKT conditions, strong duality, SVM, support vectors, soft margin, SMO

---

## 1. Logistic Regression

### 1.1 Historical Background

The logistic function has roots in population dynamics: Pierre-François Verhulst (1804–1849) derived it from the ODE modelling bounded population growth. Raymond Pearl and Lowell Reed reinvented it for population studies in 1920. The name **logistic** was coined by Udny Yule in 1925.

The logistic function competed with the normal CDF (probit) for decades. Joseph Berkson (1899–1982) popularised the logit form at the Mayo Clinic; its closed-form analytical expression made it far easier to work with before calculators. By 1970, logit had overtaken probit in published research.

### 1.2 The Logistic Function from an ODE

The logistic function solves the **bounded growth ODE**:

$$\frac{d}{dt}P(t) = bP(t)(1-P(t)) \implies P(t) = \frac{\exp\{a+bt\}}{1+\exp\{a+bt\}}$$

This saturates at 0 and 1, making it a natural model for probabilities.

### 1.3 Logistic Regression Model

**Encoding:** $y_i\in\{-1,+1\}$ (use $y=2y^{\text{orig}}-1$ to convert from $\{0,1\}$).

**Log-odds (logit):**

$$\ln\!\left(\frac{P(Y=1|x,\theta)}{P(Y=-1|x,\theta)}\right) = \theta^\top x$$

**Probability:**

$$P(Y=y_i|x_i,\theta) = \frac{1}{1+\exp\{-y_i\theta^\top x_i\}} = \sigma(y_i\theta^\top x_i)$$

**Maximum Likelihood Estimation:**

$$\hat{\theta} = \arg\max_\theta\prod_{i=1}^n P(Y_i=y_i|\theta,x_i) = \arg\min_\theta\sum_{i=1}^n\log(1+e^{-y_i\theta^\top x_i})$$

The logistic loss $\log(1+e^{-y_if(x_i)})$ is the negative log-likelihood. This is a convex function of $\theta$, so gradient descent finds the global minimum.

**Regularised logistic regression:**
- $\ell_1$ (Lasso-like): $\sum_i\log(1+e^{-y_if(x_i)}) + C\|\lambda\|_1$
- $\ell_2$ (Ridge-like): $\sum_i\log(1+e^{-y_if(x_i)}) + C\|\lambda\|_2^2$

---

## 2. Convex Optimisation

### 2.1 Standard Form

$$\min_{x\in\mathbb{R}^n}\;f(x) \quad\text{s.t.}\quad g_i(x)\leq 0,\;i=1,\ldots,m;\quad h_i(x)=0,\;i=1,\ldots,p$$

where $f$ and $g_i$ are differentiable convex functions, and $h_i$ are affine.

**Convex function:** $g:G\to\mathbb{R}$ is convex if for all $x,z\in G$ and $\theta\in(0,1)$:

$$g(\theta x + (1-\theta)z) \leq \theta g(x) + (1-\theta)g(z)$$

**Affine function:** $h(x) = a^\top x + b$.

### 2.2 Lagrangian

Incorporate constraints using multipliers:

$$\mathcal{L}(x,\alpha,\beta) = f(x) + \sum_{i=1}^m\alpha_i g_i(x) + \sum_{i=1}^p\beta_i h_i(x)$$

- $x$: **primal** variables.
- $\alpha\geq 0$, $\beta$: **dual** variables.

### 2.3 Primal and Dual Problems

**Primal problem:**

$$\min_x\;\Theta_\mathcal{P}(x) = \min_x\left[\max_{\alpha\geq 0,\beta}\;\mathcal{L}(x,\alpha,\beta)\right]$$

If $x$ is primal feasible (constraints satisfied), the inner max equals $f(x)$. If infeasible, the inner max is $+\infty$. So minimising $\Theta_\mathcal{P}$ is equivalent to the original problem. Optimal value: $p^*$.

**Dual problem:**

$$\max_{\alpha\geq 0,\beta}\;\Theta_\mathcal{D}(\alpha,\beta) = \max_{\alpha\geq 0,\beta}\left[\min_x\;\mathcal{L}(x,\alpha,\beta)\right]$$

$\Theta_\mathcal{D}$ is always concave (even if the primal is non-convex). Optimal value: $d^*$.

### 2.4 Weak and Strong Duality

**Weak duality:** Always $d^* \leq p^*$ (the dual is a lower bound on the primal).

**Strong duality:** Under certain conditions (e.g., **Slater's condition**), $d^* = p^*$.

**Slater's condition:** there exists a strictly feasible primal point $x$ such that all inequality constraints are strictly satisfied: $g_i(x) < 0$ for all $i$.

For convex problems satisfying Slater's condition, strong duality holds.

### 2.5 KKT Conditions

If $x^*$, $\alpha^*$, $\beta^*$ satisfy all four KKT conditions simultaneously, then $x^*$ is primal-optimal and $(\alpha^*,\beta^*)$ are dual-optimal (under strong duality):

1. **Primal feasibility:** $g_i(x^*)\leq 0$ and $h_i(x^*)=0$.
2. **Dual feasibility:** $\alpha_{i}^{*}\geq 0$.
3. **Complementary slackness:** $\alpha_{i}^{*}g_i(x^*)=0$ for all $i$.  
   (Either the constraint is active $g_i=0$, or the multiplier is zero $\alpha_i=0$.)
4. **Lagrangian stationarity:**

$$\nabla_x\mathcal{L}(x^*,\alpha^*,\beta^*) = \nabla f(x^*) + \sum_i\alpha_{i}^{*}\nabla g_i(x^*) + \sum_i\beta_{i}^{*}\nabla h_i(x^*) = 0$$

**Theorem (KKT):** Under strong duality, any primal-dual optimal pair must satisfy all four KKT conditions, and any point satisfying all four is primal/dual optimal.

---

## 3. Support Vector Machines (SVM)

### 3.1 Motivation

SVMs find the decision boundary that **maximises the margin** — the distance from the boundary to the nearest training point.

**Why large margins matter:**
- The decision boundary is more stable to small perturbations in the data.
- A large margin provides a geometric guarantee that similar test points will be classified correctly.

Logistic regression and decision trees do not generally produce large margins. AdaBoost and SVMs do.

### 3.2 Geometric Margin

For a linear classifier $f(x) = \lambda^\top x + \lambda_0$, the **geometric margin** of point $x_i$ is the signed distance from $x_i$ to the decision boundary:

$$\tilde{f}(x_i) = \gamma_i = \frac{\lambda^\top x_i + \lambda_0}{\|\lambda\|_2}$$

The margin is positive if $x_i$ is on the correct side.

### 3.3 Hard-Margin SVM (Separable Case)

**Goal:** find $(\lambda, \lambda_0)$ that maximises the minimum margin.

Starting from $\max \gamma\;\text{s.t.}\;y_i(\lambda^\top x_i+\lambda_0)/\|\lambda\|\geq\gamma$, normalise so that the margin equals $1/\|\lambda\|$:

$$\min_{\lambda,\lambda_0}\;\frac{1}{2}\|\lambda\|_2^2 \quad\text{s.t.}\quad y_i(\lambda^\top x_i+\lambda_0)\geq 1,\;i=1,\ldots,n$$

Equivalently: $-y_i(\lambda^\top x_i+\lambda_0)+1\leq 0$ for all $i$.

**Lagrangian:**

$$\mathcal{L}(\lambda,\lambda_0,\alpha) = \frac{1}{2}\|\lambda\|_2^2 + \sum_{i=1}^n\alpha_i\big[-y_i(\lambda^\top x_i+\lambda_0)+1\big]$$

**KKT stationarity conditions:**

$$\frac{\partial\mathcal{L}}{\partial\lambda}=0 \implies \lambda = \sum_{i=1}^n\alpha_i y_i x_i$$

$$\frac{\partial\mathcal{L}}{\partial\lambda_0}=0 \implies \sum_{i=1}^n\alpha_i y_i = 0$$

Substituting $\lambda=\sum_i\alpha_iy_ix_i$ into the Lagrangian:

$$\mathcal{L}(\alpha) = \sum_{i=1}^n\alpha_i - \frac{1}{2}\sum_{i,k}\alpha_i\alpha_ky_iy_k x_i^\top x_k$$

**Dual problem:**

$$\max_\alpha\;\sum_{i=1}^n\alpha_i - \frac{1}{2}\sum_{i,k}\alpha_i\alpha_ky_iy_k x_i^\top x_k \quad\text{s.t.}\quad \alpha_i\geq 0,\;\sum_{i=1}^n\alpha_iy_i=0$$

This is a quadratic program (QP) in $\alpha$.

**Recovering the primal:**

$$\lambda^* = \sum_{i:\alpha_i>0}\alpha_{i}^{*}y_ix_i, \qquad \lambda_0^* = 1 - \min_{i:y_i=1}(\lambda^*)^\top x_i$$

**Support vectors:** points with $\alpha_{i}^{*}>0$ — they lie exactly on the margin boundary and are the only points that determine $\lambda^*$.

### 3.4 Soft-Margin SVM (Non-Separable Case)

When data is not linearly separable, allow violations via **slack variables** $\xi_i\geq 0$:

$$\min_{\lambda,\lambda_0,\xi}\;\frac{1}{2}\|\lambda\|_2^2 + C\sum_{i=1}^n\xi_i \quad\text{s.t.}\quad y_i(\lambda^\top x_i+\lambda_0)\geq 1-\xi_i,\;\xi_i\geq 0$$

This is equivalent to minimising hinge loss with $\ell_2$ regularisation:

$$\min_{\lambda,\lambda_0}\;\frac{1}{2}\|\lambda\|_2^2 + C\sum_{i=1}^n\max(0,1-y_if(x_i))$$

**Dual problem** (same form as separable case but with box constraint):

$$\max_\alpha\;\sum_{i=1}^n\alpha_i - \frac{1}{2}\sum_{i,k}\alpha_i\alpha_ky_iy_k x_i^\top x_k \quad\text{s.t.}\quad 0\leq\alpha_i\leq C,\;\sum_i\alpha_iy_i=0$$

**Three types of points:**

| $\alpha_i$ value | $\xi_i$ | Interpretation |
|---|---|---|
| $\alpha_i\in(0,C)$ | $\xi_i=0$ | On the margin boundary; support vector |
| $\alpha_i=C$ | $\xi_i\in(0,1)$ | Inside the margin; correctly classified |
| $\alpha_i=C$ | $\xi_i\geq 1$ | Misclassified |

### 3.5 Dual Derivation for Soft-Margin

**Lagrangian:**

$$\mathcal{L}(\lambda,\lambda_0,\xi,\alpha,r) = \frac{1}{2}\|\lambda\|_2^2 + C\sum_i\xi_i - \sum_i\alpha_i[y_i(\lambda^\top x_i+\lambda_0)-1+\xi_i] - \sum_i r_i\xi_i$$

**Stationarity:**

$$\nabla_\lambda\mathcal{L}=0 \implies \lambda=\sum_i\alpha_iy_ix_i$$

$$\frac{\partial\mathcal{L}}{\partial\lambda_0}=0 \implies \sum_i\alpha_iy_i=0$$

$$\nabla_\xi\mathcal{L}=0 \implies r_i=C-\alpha_i\geq 0 \implies \alpha_i\leq C$$

Substituting these into $\mathcal{L}$, the $\xi$ terms cancel (since $C-\alpha_i-r_i=0$), and the dual takes the same form as the hard-margin case with the added constraint $\alpha_i\leq C$.

### 3.6 Sequential Minimal Optimisation (SMO)

SVMs require solving a QP. SMO is a specialised **coordinate ascent** algorithm:

1. Choose two multipliers $\alpha_1$, $\alpha_2$ to update jointly (they must be updated together to preserve the constraint $\sum_i\alpha_iy_i=0$).
2. Optimise the dual objective over $\alpha_2\in[L,H]$ with $\alpha_1 = y_1(\zeta - \alpha_2 y_2)$:

$$\max_{\alpha_2\in[L,H]}\;\alpha_1+\alpha_2+\text{const} - \frac{1}{2}\sum_{i,k}\alpha_i\alpha_ky_iy_k x_i^\top x_k$$

The objective is **quadratic in $\alpha_2$**, so the optimal value is found by setting its derivative to zero analytically. If the optimum lies outside $[L, H]$, clip to the boundary.

3. Repeat until convergence.

---

## 4. Summary

```
Logistic Regression:
  P(y|x,θ) = σ(y θᵀx)
  Loss: log(1 + e^{-yθᵀx})   [convex, no closed-form solution]

Convex Optimisation (KKT):
  Lagrangian:  L = f(x) + Σ αᵢgᵢ(x) + Σ βᵢhᵢ(x)
  KKT:         stationarity + primal/dual feasibility + complementary slackness
  Strong duality via Slater's condition

SVM:
  Hard margin:  min ½‖λ‖² s.t. yᵢ(λᵀxᵢ+λ₀) ≥ 1
  Soft margin:  min ½‖λ‖² + CΣξᵢ s.t. yᵢ(λᵀxᵢ+λ₀) ≥ 1−ξᵢ
  Dual:         max Σαᵢ − ½ΣΣαᵢαₖyᵢyₖxᵢᵀxₖ,  0≤αᵢ≤C, Σαᵢyᵢ=0
  λ* = Σ αᵢ*yᵢxᵢ  (sum over support vectors only)
```

---

## 5. References

1. **Cortes, C. & Vapnik, V.** (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. [Original SVM paper.]
2. **Platt, J.** (1998). Sequential minimal optimization: A fast algorithm for training support vector machines. *Microsoft Research Technical Report MSR-TR-98-14*. [SMO algorithm.]
3. **Boyd, S. & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press. [Free PDF at stanford.edu/~boyd/cvxbook; comprehensive treatment of Lagrangians, duality, KKT.]
4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Ch. 12 (SVMs).
5. **Schölkopf, B. & Smola, A. J.** (2002). *Learning with Kernels*. MIT Press. [SVMs with kernels; theoretical foundations.]
6. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §4.3 (Logistic regression), §7.1 (Maximum margin classifiers).

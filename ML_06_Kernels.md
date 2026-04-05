# ML 06 — Kernel Methods, RKHS, and Kernel Regression

> **Keywords:** polynomial kernel, RBF kernel, inner product, Hilbert space, RKHS, reproducing property, Mercer's theorem, representer theorem, kernel ridge regression, kernel regression

---

## 1. The Kernel Trick

Many learning algorithms depend on the data only through **inner products** $x_i^\top x_l$. By replacing this inner product with a **kernel function**:

$$k(x_i, x_l) = \langle\Phi(x_i), \Phi(x_l)\rangle_\mathcal{H}$$

we implicitly map the data into a high- or infinite-dimensional feature space $\mathcal{H}$ without ever computing $\Phi(x)$ explicitly. All we need is the ability to evaluate $k(\cdot,\cdot)$.

---

## 2. Polynomial Kernels

For $x, z\in\mathbb{R}^p$, the polynomial kernel of degree 2:

$$k(x,z) = (x^\top z + c)^2$$

Expanding:

$$= \sum_{j,l=1}^p (x^{(j)}x^{(l)})(z^{(j)}z^{(l)}) + \sum_{j=1}^p(\sqrt{2c}x^{(j)})(\sqrt{2c}z^{(j)}) + c^2$$

This equals $\langle\Phi(x),\Phi(z)\rangle$ where $\Phi(x)$ contains all degree-2 monomials plus scaled linear terms and a constant. The constant $c$ controls the relative weight of linear vs. quadratic terms.

For degree $d$: $k(x,z)=(x^\top z+c)^d$ implicitly maps to a feature space containing all monomials up to degree $d$. The decision boundary in input space is a degree-$d$ polynomial; in feature space it is a hyperplane.

---

## 3. Gaussian (RBF) Kernel

$$k(x,z) = \exp\!\left(-\frac{\|x-z\|^2}{\sigma^2}\right)$$

This corresponds to an **infinite-dimensional** feature space. It is translation-invariant ($k$ depends only on $x-z$).

**SVM with Gaussian kernel:** The decision function is:

$$f(x) = \sum_i\alpha_i^*y_i k(x_i,x) + \lambda_0^*$$

where $\lambda_0^* = 1 - \sum_i\alpha_i^*y_ik(x_i, x_{sv})$ for any support vector $x_{sv}$.

---

## 4. Inner Products and Hilbert Spaces

An **inner product** $\langle\cdot,\cdot\rangle$ on a vector space $\mathcal{X}$ satisfies:

1. **Symmetry:** $\langle u,v\rangle = \langle v,u\rangle$.
2. **Bilinearity:** $\langle\alpha u+\beta v,w\rangle = \alpha\langle u,w\rangle+\beta\langle v,w\rangle$.
3. **Strict positive definiteness:** $\langle u,u\rangle\geq 0$, with equality iff $u=0$.

An **inner product space** (pre-Hilbert space) is a vector space with an inner product. A **Hilbert space** is a complete inner product space (Cauchy sequences converge to elements of the space).

**Examples:**
- $\mathbb{R}^p$ with $\langle u,v\rangle = u^\top v$.
- $\ell^2$: square-summable sequences, $\langle u,v\rangle_{\ell^2}=\sum_{i=1}^\infty u_iv_i$.
- $L_2(\mathcal{X},\mu)$: square-integrable functions, $\langle f,g\rangle=\int f(x)g(x)\,d\mu(x)$.

---

## 5. Reproducing Kernel Hilbert Space (RKHS)

### 5.1 Definition

A **Reproducing Kernel Hilbert Space (RKHS)** $H_k$ is a Hilbert space of functions $f:\mathcal{X}\to\mathbb{R}$ equipped with a kernel $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ such that the **reproducing property** holds:

$$f(x) = \langle k(x,\cdot),\, f(\cdot)\rangle_{H_k} \quad\forall f\in H_k,\;\forall x\in\mathcal{X}$$

Setting $f=k(\cdot,z)$: $k(x,z) = \langle k(x,\cdot),k(\cdot,z)\rangle_{H_k}$.

So $\Phi(x) := k(x,\cdot)\in H_k$ and $k(x,z)=\langle\Phi(x),\Phi(z)\rangle_{H_k}$.

### 5.2 Construction of the RKHS

Define the feature map $\Phi(x) = k(\cdot,x)$. The RKHS is:

$$H_k = \overline{\text{span}\{k(\cdot,x):x\in\mathcal{X}\}} = \left\{f(\cdot) = \sum_{i=1}^m\alpha_ik(\cdot,x_i):\alpha_i\in\mathbb{R},\;m<\infty\right\}$$

(with completion for infinite sums). The inner product for $f=\sum_i\alpha_ik(\cdot,x_i)$ and $g=\sum_j\beta_jk(\cdot,x_j')$:

$$\langle f,g\rangle_{H_k} = \sum_{i,j}\alpha_i\beta_jk(x_i,x_j')$$

This is symmetric, bilinear, and positive semidefinite — a valid inner product.

### 5.3 Kernel Definition

A function $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ is a **valid kernel** if:
1. **Symmetric:** $k(x,z) = k(z,x)$.
2. **Positive semidefinite (PSD):** for any finite set $\{x_1,\ldots,x_m\}\subset\mathcal{X}$, the Gram matrix $K=[k(x_i,x_j)]$ satisfies $c^\top Kc\geq 0$ for all $c\in\mathbb{R}^m$.

**Properties of kernels:**
- $k(u,u)\geq 0$.
- $k(u,v)\leq\sqrt{k(u,u)k(v,v)}$ (Cauchy-Schwarz).

---

## 6. Mercer's Theorem

**Theorem (Mercer).** Let $\mathcal{X}\subset\mathbb{R}^n$ and $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ be a continuous symmetric function such that the integral operator $T_k:L_2(\mathcal{X})\to L_2(\mathcal{X})$ defined by $(T_kf)(\cdot)=\int k(\cdot,x)f(x)\,dx$ is positive:

$$\int_{\mathcal{X}\times\mathcal{X}}k(x,z)f(x)f(z)\,dx\,dz\geq 0 \quad\forall f\in L_2(\mathcal{X})$$

Then $k$ can be expanded as a uniformly convergent series of eigenfunctions $\psi_j\in L_2(\mathcal{X})$ with non-negative eigenvalues $\lambda_j\geq 0$:

$$k(x,z) = \sum_{j=1}^\infty\lambda_j\psi_j(x)\psi_j(z)$$

The corresponding feature map is:

$$\Phi(x) = [\sqrt{\lambda_1}\psi_1(x),\;\sqrt{\lambda_2}\psi_2(x),\;\ldots]$$

**Consequence:** any PSD kernel has a valid feature map — possibly infinite-dimensional. The Gaussian kernel corresponds to an infinite-dimensional $\Phi$.

**Finite input space.** For $\{x_1,\ldots,x_m\}$, the Gram matrix $K=[k(x_i,x_l)]$ must be symmetric PSD ($K=V\Lambda V^\top$). The feature map can then be taken as $\Phi(x_i)=[\sqrt{\lambda_1}v_1^{(i)},\ldots,\sqrt{\lambda_m}v_m^{(i)}]$.

---

## 7. Representer Theorem

**Theorem.** Let $H_k$ be an RKHS with kernel $k$. For any loss $\ell:\mathbb{R}^2\to\mathbb{R}$ and any nondecreasing $\Omega:\mathbb{R}\to\mathbb{R}$, the solutions of:

$$f^*\in\arg\min_{f\in H_k}\sum_{i=1}^n\ell(f(x_i),y_i)+\Omega(\|f\|_{H_k}^2)$$

can all be expressed as:

$$f^*(\cdot) = \sum_{i=1}^n\alpha_i k(x_i,\cdot)$$

**Implication:** even though $H_k$ may be infinite-dimensional, the optimal solution lies in the **finite $n$-dimensional subspace** spanned by $\{k(x_i,\cdot)\}_{i=1}^n$. This makes the optimisation over an infinite-dimensional function class tractable.

---

## 8. Kernel Construction Rules

Given valid kernels $k_1$ and $k_2$, the following are also valid:

| Rule | New kernel |
|---|---|
| Non-negative linear combination | $k = \alpha k_1 + \beta k_2,\;\alpha,\beta\geq 0$ |
| Product | $k = k_1 k_2$ |
| Composition | $k(x,z) = k_1(h(x),h(z))$ for any $h$ |
| Outer product | $k(x,z) = g(x)g(z)$ for any $g:\mathcal{X}\to\mathbb{R}$ |
| Polynomial of kernel | $k(x,z)=h(k_1(x,z))$ for polynomial $h$ with positive coefficients |
| Exponential | $k(x,z) = \exp(k_1(x,z))$ |

---

## 9. Kernel Ridge Regression

### 9.1 Motivation

Standard ridge regression: $\min_\lambda\|Y-X\lambda\|_2^2+C\|\lambda\|_2^2$. Solution: $\lambda^*=(X^\top X+CI)^{-1}X^\top Y$.

By the representer theorem, the optimal $\lambda^*$ lies in the column space of $X^\top$: $\lambda^*=X^\top r$ for some $r\in\mathbb{R}^n$. Substituting this ansatz:

### 9.2 Derivation

Substitute $\lambda=X^\top r$:

$$F(r) = \|Y-XX^\top r\|_2^2+C\|X^\top r\|_2^2 = \|Y-Kr\|_2^2+C\langle r,Kr\rangle$$

where $K=XX^\top\in\mathbb{R}^{n\times n}$ is the (linear) Gram matrix.

Setting the gradient to zero:

$$\nabla_r F = -2K(Y-Kr)+2CKr = 0$$

Simplifying (multiply through by $K^{-1}$ or note $K(Y-Kr)-CKr=0$):

$$(K+CI)r^* = Y \implies \boxed{r^* = (K+CI)^{-1}Y}$$

**Prediction at a new point $\tilde{x}$:**

$$f(\tilde{x}) = \tilde{x}^\top\lambda^* = \tilde{x}^\top X^\top r^* = \sum_i\tilde{x}^\top x_i\,r_i^* = K_{\tilde{x}}^\top r^* = K_{\tilde{x}}^\top(K+CI)^{-1}Y$$

where $K_{\tilde{x},i} = k(\tilde{x},x_i)$.

**With arbitrary kernel:** replace all inner products $x_i^\top x_l$ with $k(x_i,x_l)$:

$$r^*=(K+CI)^{-1}Y, \qquad f(\tilde{x}) = K_{\tilde{x}}^\top(K+CI)^{-1}Y$$

### 9.3 Equivalence with Standard Ridge

Using the Woodbury / push-through identity with $P=I$, $B=X$, $R=CI$:

$$X(XX^\top+CI)^{-1} = (X^\top X+CI)^{-1}X^\top$$

So $Xr^* = X(XX^\top+CI)^{-1}Y = (X^\top X+CI)^{-1}X^\top Y = \lambda^*$. The kernel ridge and standard ridge solutions are equivalent for the linear kernel.

### 9.4 Complexity Comparison

| | Standard Ridge | Kernel Ridge |
|--|--|--|
| Training cost | $O(np^2+p^3)$ | $O(n^2p+n^3)$ |
| Prediction cost | $O(p)$ | $O(n)$ per test point |
| Scales with | feature dim $p$ | number of training points $n$ |
| Advantage when | $n\gg p$ | $p\gg n$ or infinite-dim features |

---

## 10. Kernel Regression (Nadaraya-Watson)

**Kernel regression** (non-parametric) estimates the conditional expectation directly:

$$\hat{f}(\tilde{x}) = \frac{\sum_{i=1}^n K_h(\tilde{x}-x_i)\,y_i}{\sum_{i=1}^n K_h(\tilde{x}-x_i)}$$

where $K_h(\cdot)$ is a kernel **density** function with bandwidth $h$ (not to be confused with a Mercer kernel).

**Differences from kernel ridge regression:**
- No optimisation step — just a weighted average of training labels.
- Weights are determined by proximity only, not globally optimised.
- Kernel ridge regression generally outperforms kernel regression because the "heights of the bumps" (the $r^*$ coefficients) are globally optimised to fit the data.

---

## 11. Summary

```
Feature map:   Φ: X → H    (possibly infinite-dimensional)
Kernel:        k(x,z) = ⟨Φ(x), Φ(z)⟩_H    (never compute Φ explicitly)

Valid kernel ⟺ symmetric + PSD Gram matrix

RKHS reproducing property:
  f(x) = ⟨k(x,·), f(·)⟩_{H_k}

Representer theorem:
  f*(·) = Σᵢ αᵢ k(xᵢ, ·)   [optimal f is an n-dim expansion]

Kernel ridge regression:
  r* = (K + CI)⁻¹ Y
  f(x̃) = K_{x̃}ᵀ r*

Kernel regression (Nadaraya-Watson):
  f̂(x̃) = Σᵢ Kₕ(x̃−xᵢ)yᵢ / Σᵢ Kₕ(x̃−xᵢ)
```

---

## 12. References

1. **Schölkopf, B. & Smola, A. J.** (2002). *Learning with Kernels*. MIT Press. [Comprehensive treatment of kernels, RKHS, SVMs, and the representer theorem.]
2. **Mercer, J.** (1909). Functions of positive and negative type and their connection with the theory of integral equations. *Philosophical Transactions of the Royal Society of London A*, 209, 415–446. [Original Mercer's theorem.]
3. **Aronszajn, N.** (1950). Theory of reproducing kernels. *Transactions of the American Mathematical Society*, 68(3), 337–404. [Foundational RKHS theory.]
4. **Kimeldorf, G. S. & Wahba, G.** (1970). A correspondence between Bayesian estimation on stochastic processes and smoothing by splines. *Annals of Mathematical Statistics*, 41(2), 495–502. [Original representer theorem.]
5. **Nadaraya, E. A.** (1964). On estimating regression. *Theory of Probability and its Applications*, 9(1), 141–142. [Kernel regression estimator.]
6. **Watson, G. S.** (1964). Smooth regression analysis. *Sankhyā: The Indian Journal of Statistics*, 26(4), 359–372. [Kernel regression estimator (independently).]
7. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Ch. 6 (Kernel smoothing), Ch. 12 (SVMs and kernels).

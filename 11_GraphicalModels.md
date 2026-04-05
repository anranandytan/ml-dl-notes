# 11 — Gaussian Graphical Models

> **Keywords:** precision matrix, conditional independence, partial correlation, Gaussian Bayesian network, Gaussian Markov network, structural equation model, graphical lasso, moralisation

---

## 1. What Problem Are We Solving?

A multivariate Gaussian with $p$ variables has a covariance matrix $\Sigma\in\mathbb{R}^{p\times p}$ with $O(p^2)$ free parameters. When $p$ is large (e.g., genes, brain regions, financial assets), this is:
- **Too many parameters** to estimate reliably from limited data.
- **Hard to interpret** — every variable appears correlated with every other.

Real systems are **sparse**: most variables are conditionally independent given the rest. A **Gaussian Graphical Model (GGM)** makes this sparsity explicit by representing the dependency structure as a graph.

**Two complementary types:**

| Type | Graph | Independence encoded via |
|------|-------|------------------------|
| Gaussian Bayesian Network (GBN) | Directed Acyclic Graph (DAG) | Factorisation of joint distribution |
| Gaussian Markov Network (GMN) | Undirected graph | Zero entries of the precision matrix |

---

## 2. Historical Notes

| Year | Event |
|------|-------|
| 1921 | Sewall Wright introduces **path analysis** for genetics — the precursor to linear Gaussian BNs. |
| 1972 | **Dempster** introduces covariance selection: fitting a sparse inverse covariance matrix. |
| 1990 | **Whittaker** systematises Gaussian graphical models in *Graphical Models in Applied Multivariate Statistics*. |
| 1996 | **Lauritzen** provides the definitive mathematical treatment in *Graphical Models* (Oxford). |
| 2008 | **Friedman, Hastie & Tibshirani** introduce the **graphical lasso (glasso)**: $\ell_1$-penalised MLE for sparse precision matrices — the dominant method for structure learning. |

---

## 3. The Multivariate Gaussian and Its Precision Matrix

Let $x=(x_1,\ldots,x_p)^\top\sim\mathcal{N}(\mu,\Sigma)$ with:

$$\Sigma = \begin{pmatrix}
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p}\\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p}\\
\vdots      & \vdots      & \ddots & \vdots     \\
\sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp}
\end{pmatrix}, \qquad \sigma_{ij} = \text{Cov}(x_i,x_j)$$

The **precision matrix** (also: concentration matrix, information matrix) is:

$$\Lambda = \Sigma^{-1}, \qquad \Lambda_{ij} = \lambda_{ij}$$

### 3.1 Why the Precision Matrix Is More Natural for Graphs

The covariance $\sigma_{ij}$ captures *marginal* dependence between $x_i$ and $x_j$ — even if their association is entirely explained by a common cause. The precision entry $\lambda_{ij}$ captures *conditional* dependence — what remains after accounting for all other variables. This is precisely what a graph edge should represent.

### 3.2 Three Key Independence Properties

**Property 1 — Marginal independence:**

$$x_i\perp x_j \iff \sigma_{ij}=0$$

If the covariance is zero, $x_i$ and $x_j$ are uncorrelated, and (for Gaussians) uncorrelated implies independent.

**Property 2 — Conditional independence:**

$$x_i\perp x_j\mid x_{-ij} \iff \lambda_{ij}=0$$

where $x_{-ij}=\{x_k:k\neq i, k\neq j\}$ denotes all other variables. **This is the central result**: the zero pattern of $\Lambda$ is the conditional independence graph. Edge $(i,j)$ is absent iff $\lambda_{ij}=0$.

**Why?** The density of a multivariate Gaussian involves $x^\top\Lambda x$. Terms $\lambda_{ij}x_ix_j$ (for $i\neq j$) couple $x_i$ and $x_j$ in the exponent. If $\lambda_{ij}=0$, the density factorises over $x_i$ and $x_j$ conditional on all others.

**Property 3 — Partial correlation:**

$$\rho_{ij\cdot\text{rest}} = -\frac{\lambda_{ij}}{\sqrt{\lambda_{ii}\lambda_{jj}}}$$

The **partial correlation** between $x_i$ and $x_j$ given all others is proportional to $-\lambda_{ij}$. It is zero if and only if $\lambda_{ij}=0$.

**Warning:** $\sigma_{ij}=0 \not\Rightarrow \lambda_{ij}=0$ (marginal independence does not imply conditional independence) and vice versa. These are genuinely different concepts.

---

## 4. Gaussian Bayesian Network (Directed)

### 4.1 Definition

A **Gaussian Bayesian Network** is a DAG where each variable $x_i$ has a Gaussian conditional distribution given its parents $x_{\text{pa}(i)}$:

$$p(x) = \prod_{i=1}^p p(x_i\mid x_{\text{pa}(i)})$$

$$p(x_i\mid x_{\text{pa}(i)}) = \mathcal{N}\left(\mu_i + \sum_{j\in\text{pa}(i)}w_{ij}(x_j-\mu_j), \sigma_i^2\right)$$

The conditional mean is an affine function of the parents (each centred at its own marginal mean $\mu_j$).

**Parameters:** For each node $i$, we need $|\text{pa}(i)|$ regression coefficients $w_{ij}$, a conditional variance $\sigma_i^2$, and an intercept $\mu_i$.

### 4.2 Structural Equation Model (SEM)

Subtracting the marginal mean from both sides of the conditional gives a **structural equation**:

$$x_i - \mu_i = \sum_{j\in\text{pa}(i)}w_{ij}(x_j-\mu_j) + \sigma_i\varepsilon_i, \qquad \varepsilon_i\overset{\text{iid}}{\sim}\mathcal{N}(0,1)$$

The deviation of $x_i$ from its mean is explained by deviations of its parents from their means, plus independent noise.

**Matrix form.** Stack all $p$ equations: let $W\in\mathbb{R}^{p\times p}$ be the weight matrix ($W_{ij}=w_{ij}$ if $j\in\text{pa}(i)$, else $0$) and $S=\text{diag}(\sigma_1,\ldots,\sigma_p)$:

$$(x-\mu) = W(x-\mu) + S\varepsilon$$

$$(I-W)(x-\mu) = S\varepsilon$$

$$\boxed{x-\mu = (I-W)^{-1}S\varepsilon}$$

**Why is $(I-W)$ invertible?** Because the DAG can always be topologically ordered. In that order, $W$ is *strictly lower triangular* (parents always have smaller indices than children). A strictly lower-triangular matrix has all eigenvalues equal to 0, so $(I-W)$ has all eigenvalues equal to 1 and is always invertible.

### 4.3 Covariance from the SEM

Since $\varepsilon\sim\mathcal{N}(0,I)$ and $x-\mu=(I-W)^{-1}S\varepsilon$:

$$\Sigma = \text{Cov}(x) = (I-W)^{-1}S\,\mathbb{E}[\varepsilon\varepsilon^\top]\,S^\top(I-W)^{-\top} = (I-W)^{-1}S^2(I-W)^{-\top}$$

where $S^2=\text{diag}(\sigma_1^2,\ldots,\sigma_p^2)$.

**Equivalently:** $\Lambda=\Sigma^{-1}=(I-W)^\top S^{-2}(I-W)$. The precision matrix has a simple factored form in terms of the DAG weights.

### 4.4 d-Separation

In a DAG, conditional independence is read off via **d-separation** (directional separation). Three nodes $A$, $B$, $C$ satisfy $A\perp B\mid C$ if all paths from $A$ to $B$ are *blocked* given $C$.

A path is blocked at a node $Z$ if:
- **Chain** ($A\to Z\to B$) or **fork** ($A\leftarrow Z\to B$): $Z$ is in the conditioning set $C$.
- **Collider** ($A\to Z\leftarrow B$): $Z$ and all its descendants are *not* in $C$.

**Example:** In $X\to Z\to Y$, conditioning on $Z$ blocks the path, so $X\perp Y\mid Z$. In $X\to Z\leftarrow Y$ (a collider), not conditioning on $Z$ means $X\perp Y$ marginally, but conditioning on $Z$ opens the path — $X\not\perp Y\mid Z$.

---

## 5. Gaussian Markov Network (Undirected)

### 5.1 MRF Factor Decomposition

A **Gaussian Markov Network (GMN)** is an undirected graphical model. The joint factorises over cliques:

$$p(x) = \frac{1}{Z}\prod_{c\in\mathcal{C}}\psi_c(x_c)$$

For a Gaussian, write each potential in log-linear form $\psi_c = \exp(-E_c)$:

$$p(x) = \frac{1}{Z}\exp\left(-\sum_c E_c(x_c)\right) = \frac{1}{Z}\exp(-E(x))$$

### 5.2 Reading the Potentials from the Gaussian

The Gaussian density:

$$p(x)\propto\exp\left(-\frac{1}{2}(x-\mu)^\top\Lambda(x-\mu)\right)$$

Expanding (and defining $h=\Lambda\mu$):

$$= \exp\left(-\frac{1}{2}x^\top\Lambda x + h^\top x\right)$$

Decomposing by pairs of variables:

$$= \exp\left(\sum_i \left(-\frac{1}{2}\lambda_{ii}x_i^2+h_ix_i\right) + \sum_{i<j}\left(-\lambda_{ij}x_ix_j\right)\right)$$

This gives us:
- **Node potential** for $x_i$: $\psi_i(x_i)\propto\exp\left(-\frac{1}{2}\lambda_{ii}x_i^2+h_ix_i\right)$
- **Edge potential** for $(x_i,x_j)$: $\psi_{ij}(x_i,x_j)\propto\exp\left(-\lambda_{ij}x_ix_j\right)$

**Graph structure from $\Lambda$:** Edge $(i,j)$ exists in the GMN if and only if $\lambda_{ij}\neq 0$.

### 5.3 Full Conditional Distribution (Markov Blanket)

The full conditional of $x_i$ given all others $x_{-i}$ is found by collecting all terms in $\log p(x)$ that involve $x_i$:

$$\log p(x_i\mid x_{-i}) = -\frac{1}{2}\lambda_{ii}x_i^2 + x_i\left(h_i-\sum_{j\neq i}\lambda_{ij}x_j\right) + \text{const}$$

Completing the square in $x_i$:

$$\boxed{x_i\mid x_{-i}\sim\mathcal{N}\left(\frac{h_i-\sum_{j\neq i}\lambda_{ij}x_j}{\lambda_{ii}},  \frac{1}{\lambda_{ii}}\right)}$$

where $h_i=(\Lambda\mu)_i$.

**For zero-mean $\mu=0$** (so $h=0$):

$$x_i\mid x_{-i}\sim\mathcal{N}\left(-\sum_{j\neq i}\frac{\lambda_{ij}}{\lambda_{ii}}x_j,  \frac{1}{\lambda_{ii}}\right)$$

**Key observations:**
1. The conditional mean of $x_i$ depends only on its **neighbours** in the graph (nodes $j$ with $\lambda_{ij}\neq 0$). All non-neighbours drop out — this is the Markov property.
2. The conditional variance is $1/\lambda_{ii}$, determined solely by the diagonal precision entry.
3. The sign of $\lambda_{ij}$ determines whether neighbour $j$ has a positive or negative influence on the conditional mean of $x_i$.

---

## 6. Comparing GBN and GMN

| | GBN (Directed) | GMN (Undirected) |
|--|--|--|
| **Graph** | DAG | Undirected graph |
| **Independence** | d-separation in DAG | $\lambda_{ij}=0$ iff no edge |
| **Parameters** | Weights $w_{ij}$, variances $\sigma_i^2$ | Precision matrix $\Lambda$ |
| **Covariance** | $\Sigma=(I-W)^{-1}S^2(I-W)^{-\top}$ | $\Sigma=\Lambda^{-1}$ |
| **Learning** | OLS regression per node | MLE / graphical lasso for $\Lambda$ |
| **Directed cycles** | Not allowed | N/A |

### 6.1 Converting a GBN to a GMN: Moralisation

Every GBN can be converted to a GMN representing the same independence structure via **moralisation**:

1. For each child node that has multiple parents, add an undirected edge between every pair of its parents (this is the "marrying" step — parents of the same child become connected).
2. Replace all directed edges with undirected edges.

The resulting undirected graph may have more edges than the precision matrix sparsity pattern of $\Sigma^{-1}$ — the GBN encodes more independence than the corresponding GMN after moralisation.

---

## 7. Structure Learning: Graphical Lasso

### 7.1 The Problem

Given $N$ i.i.d. observations from $\mathcal{N}(\mu,\Sigma)$, we want to estimate $\Lambda=\Sigma^{-1}$ under the assumption that it is **sparse** (few non-zero off-diagonal entries = sparse graph).

The MLE for $\Lambda$ is $\hat{\Sigma}^{-1}$ (inverse sample covariance), which is always dense and undefined when $p>N$. We need regularisation.

### 7.2 The Graphical Lasso Objective

**Friedman, Hastie & Tibshirani (2008)** propose the $\ell_1$-penalised log-likelihood:

$$\hat{\Lambda} = \arg\max_{\Lambda\succ 0} \left[\log\det\Lambda - \text{tr}(\hat{\Sigma}\Lambda) - \rho\|\Lambda\|_1\right]$$

where:
- $\hat{\Sigma}=\frac{1}{N}\sum_i(x_i-\bar{x})(x_i-\bar{x})^\top$ is the sample covariance.
- $\|\Lambda\|_1=\sum_{i\neq j}|\lambda_{ij}|$ is the off-diagonal $\ell_1$ norm (diagonal not penalised).
- $\rho>0$ controls sparsity: larger $\rho$ forces more entries to zero.

**Why $\log\det\Lambda - \text{tr}(\hat{\Sigma}\Lambda)$?** The Gaussian log-likelihood for $\Lambda$ is $\frac{N}{2}[\log\det\Lambda - \text{tr}(\hat{\Sigma}\Lambda)] + \text{const}$. The penalised objective replaces this with an $\ell_1$ penalty.

### 7.3 Properties

- The objective is **concave** in $\Lambda$ (for fixed penalty): $\log\det\Lambda$ is concave, $-\text{tr}(\hat{\Sigma}\Lambda)$ is linear.
- Solved by **coordinate descent** (the glasso algorithm): iterate over each row/column of $\Lambda$, solving a Lasso-type subproblem.
- The penalty $\rho$ can be chosen by cross-validation or by BIC.
- Entries set to zero correspond to absent edges in the estimated graph.

---

## 8. Summary

```
Gaussian Graphical Model = Gaussian + Graph structure

Two types:
  GBN (directed):     p(x) = ∏ p(xᵢ|x_{pa(i)})
                      Independence via d-separation
                      x − μ = (I−W)⁻¹Sε   [SEM form]

  GMN (undirected):   p(x) ∝ exp(−½xᵀΛx + hᵀx)
                      λᵢⱼ = 0  ⟺  xᵢ ⊥ xⱼ | rest  [KEY RESULT]
                      Node potential: N(−Σⱼ≠ᵢ(λᵢⱼ/λᵢᵢ)xⱼ, 1/λᵢᵢ)

Precision matrix Λ = Σ⁻¹:
  σᵢⱼ = 0  ⟺  marginal independence
  λᵢⱼ = 0  ⟺  conditional independence

Structure learning:
  Graphical Lasso: max_{Λ≻0} [log det Λ − tr(Σ̂Λ) − ρ‖Λ‖₁]
```

---

## 9. References

1. **Lauritzen, S. L.** (1996). *Graphical Models*. Oxford University Press. — Definitive mathematical treatment of Gaussian and discrete graphical models.
2. **Whittaker, J.** (1990). *Graphical Models in Applied Multivariate Statistics*. Wiley. — Classical reference for Gaussian graphical models and covariance selection.
3. **Koller, D. & Friedman, N.** (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press. Ch. 7 (Gaussian networks). — Comprehensive; covers both directed and undirected Gaussian models.
4. **Friedman, J., Hastie, T., & Tibshirani, R.** (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432–441. — Graphical lasso algorithm.
5. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §8.1 (Bayesian networks), §8.3 (Markov random fields).
6. **Wright, S.** (1921). Correlation and causation. *Journal of Agricultural Research*, 20(7), 557–585. — Original path analysis paper.
7. **Dempster, A. P.** (1972). Covariance selection. *Biometrics*, 28(1), 157–175. — First systematic treatment of sparse inverse covariance estimation.

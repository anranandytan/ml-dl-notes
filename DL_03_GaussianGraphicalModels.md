# Gaussian Graphical Models

> **Keywords:** Gaussian Bayesian network, Gaussian Markov network, precision matrix, conditional independence, structural equation model, Markov blanket, graphical lasso

---

## 1. Motivation & Intuition

A **Gaussian graphical model** (GGM) encodes conditional independence relationships among a collection of jointly Gaussian random variables using a graph. The nodes are variables; the edges (or their absence) represent statistical dependencies.

**Why graphical models?** In high dimensions $p$, a full covariance matrix $\Sigma \in \mathbb{R}^{p\times p}$ has $O(p^2)$ free parameters. Real-world systems are sparse: most variables are conditionally independent given others. A graphical model makes this sparsity explicit, enabling:

- **Interpretability:** the graph reveals the dependency structure (e.g., gene regulatory networks, brain connectivity).
- **Efficient inference:** sparse graphs admit fast message-passing algorithms.
- **Regularised estimation:** enforcing a sparse graph (few edges) prevents overfitting in high dimensions.

**Two complementary views:**

| View | Model | Graph type | Independence encoded in |
|------|-------|-----------|------------------------|
| **Directed** | Gaussian Bayesian Network (GBN) | DAG | Conditional distribution factorisation |
| **Undirected** | Gaussian Markov Network (GMN) | Undirected graph | Precision matrix sparsity |

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1921 | Sewall Wright introduces **path analysis** — the precursor to linear Gaussian Bayesian networks — for genetics. |
| 1972 | **Dempster** introduces covariance selection: estimating a sparse inverse covariance matrix as a graphical model. |
| 1990 | **Whittaker** systematises Gaussian graphical models in *Graphical Models in Applied Multivariate Statistics*. |
| 1996 | **Lauritzen** provides the definitive mathematical treatment of graphical models including the Gaussian case. |
| 2008 | **Friedman, Hastie & Tibshirani** introduce the **graphical lasso (glasso)**: $\ell_1$-penalised MLE for sparse precision matrices — the dominant method for GGM structure learning. |
| 2010s | GGMs applied at scale to genomics, neuroscience, and finance for network discovery. |

---

## 3. Multivariate Gaussian and the Precision Matrix

Let $x = (x_1, \ldots, x_p)^\top \sim \mathcal{N}(\mu, \Sigma)$:

$$x = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_p \end{pmatrix}, \quad \mu = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_p \end{pmatrix}, \quad \Sigma = \begin{pmatrix} \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p} \\ \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp} \end{pmatrix}$$

where $\Sigma$ is symmetric positive definite and $\sigma_{ij} = \text{Cov}(x_i, x_j)$.

The **precision matrix** (also called the information matrix or concentration matrix) is:

$$\Lambda = \Sigma^{-1}, \qquad \Lambda_{ij} = \lambda_{ij}$$

### 3.1 Key Independence Properties

**Marginal independence:**
$$x_i \perp x_j \iff \sigma_{ij} = 0$$

The $(i,j)$ entry of the covariance matrix is zero.

**Conditional independence (given all other variables):**
$$x_i \perp x_j \mid x_{-ij} \iff \lambda_{ij} = 0$$

where $x_{-ij} = \{x_k : k \neq i, k \neq j\}$ denotes all variables except $x_i$ and $x_j$.

This is the central result of Gaussian graphical models: **the sparsity pattern of $\Lambda$ directly encodes the conditional independence graph.** Edge $(i,j)$ is absent in the MRF iff $\lambda_{ij} = 0$.

**Partial correlation.** The standardised precision entry:
$$\rho_{ij \cdot \text{rest}} = -\frac{\lambda_{ij}}{\sqrt{\lambda_{ii}\lambda_{jj}}}$$
is the **partial correlation** between $x_i$ and $x_j$ given all others. It is zero iff $\lambda_{ij} = 0$.

---

## 4. Gaussian Bayesian Network (Directed)

### 4.1 Factorisation

A **Gaussian Bayesian Network (GBN)** is a directed acyclic graph (DAG) where each variable $x_i$ has a Gaussian conditional distribution given its parents $x_{\text{pa}(i)}$:

$$p(x) = \prod_{i=1}^{p} p(x_i \mid x_{\text{pa}(i)})$$

$$p(x_i \mid x_{\text{pa}(i)}) = \mathcal{N}\!\left(x_i \;\Big|\; \mu_i + \sum_{j \in \text{pa}(i)} w_{ij}(x_j - \mu_j),\; \sigma_i^2\right)$$

The conditional mean is an **affine function of the parents** (shifted by their means). This is also called a **linear Gaussian model**.

### 4.2 Structural Equation Form

The conditional distribution implies a structural equation for each variable. Subtracting the marginal mean from both sides:

$$x_i - \mu_i = \sum_{j \in \text{pa}(i)} w_{ij}(x_j - \mu_j) + \sigma_i\,\varepsilon_i, \qquad \varepsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)$$

This expresses each centred variable as a weighted sum of its parents' centred values plus independent noise. Stacking all $p$ equations in matrix form:

$$(x - \mu) = W(x - \mu) + S\varepsilon$$

where:
- $W \in \mathbb{R}^{p\times p}$: weight matrix with $W_{ij} = w_{ij}$ if $j \in \text{pa}(i)$, else $0$. For a DAG with a topological ordering, $W$ is **strictly lower triangular**.
- $S = \text{diag}(\sigma_1, \ldots, \sigma_p)$: diagonal noise scale matrix.
- $\varepsilon \sim \mathcal{N}(0, I)$: independent noise vector.

### 4.3 Solving for $x - \mu$

Rearranging:

$$(I - W)(x - \mu) = S\varepsilon$$

Since $W$ is strictly lower triangular (all eigenvalues are $0$), $I - W$ is lower triangular with all diagonal entries equal to $1$, hence **always invertible**:

$$x - \mu = (I - W)^{-1}S\varepsilon$$

### 4.4 Covariance Matrix

Since $x - \mu = (I-W)^{-1}S\varepsilon$ and $\mathbb{E}[\varepsilon\varepsilon^\top] = I$:

$$\Sigma = \text{Cov}(x) = (I-W)^{-1}S\,\mathbb{E}[\varepsilon\varepsilon^\top]\,S^\top\big((I-W)^{-1}\big)^\top = (I-W)^{-1}S^2\big((I-W)^\top\big)^{-1}$$

where $S^2 = \text{diag}(\sigma_1^2, \ldots, \sigma_p^2)$. This links the DAG structure (encoded in $W$) and the noise variances to the full covariance matrix.

---

## 5. Gaussian Markov Network (Undirected)

### 5.1 MRF Factor Decomposition

A **Gaussian Markov Network (GMN)** is an undirected graphical model with node potentials $\psi_i(x_i)$ and edge potentials $\psi_{ij}(x_i, x_j)$:

$$p(x) = \frac{1}{Z}\prod_{i=1}^{p}\psi_i(x_i)\prod_{(i,j)\in\mathcal{E}}\psi_{ij}(x_i, x_j)$$

### 5.2 Reading off Potentials from the Gaussian Form

The multivariate Gaussian density is:

$$p(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}\exp\!\left\{-\frac{1}{2}(x-\mu)^\top\Lambda(x-\mu)\right\}$$

Expanding the exponent using $\Lambda = \Sigma^{-1}$:

$$-\frac{1}{2}(x-\mu)^\top\Lambda(x-\mu) = -\frac{1}{2}x^\top\Lambda x + (\Lambda\mu)^\top x - \frac{1}{2}\mu^\top\Lambda\mu$$

The last term is a constant. Define the **potential vector** $h = \Lambda\mu \in \mathbb{R}^p$. Then:

$$p(x) \propto \exp\!\left\{-\frac{1}{2}x^\top\Lambda x + h^\top x\right\}$$

**Expanding into node and edge terms.** Write $x^\top\Lambda x = \sum_i \lambda_{ii}x_i^2 + 2\sum_{i < j}\lambda_{ij}x_ix_j$ and $h^\top x = \sum_i h_i x_i$:

$$\log p(x) + \text{const} = \sum_i\left(-\frac{1}{2}\lambda_{ii}x_i^2 + h_i x_i\right) + \sum_{i < j}\left(-\lambda_{ij}x_ix_j\right)$$

Reading off the potentials:

| Potential | Variables | Log-form |
|-----------|-----------|----------|
| Node potential $\psi_i(x_i)$ | $x_i$ alone | $-\frac{1}{2}\lambda_{ii}x_i^2 + h_i x_i$ |
| Edge potential $\psi_{ij}(x_i,x_j)$ | pair $(x_i, x_j)$ | $-\lambda_{ij}x_ix_j$ |

**Graph structure from $\Lambda$:** An edge $(i,j)$ exists in the graph iff $\lambda_{ij} \neq 0$, because only then does the edge potential $\psi_{ij}$ contribute a non-trivial dependency between $x_i$ and $x_j$.

### 5.3 Summary of Independence Properties

| Property | Criterion | Interpretation |
|----------|-----------|----------------|
| Marginal independence: $x_i \perp x_j$ | $\sigma_{ij} = 0$ | No covariance; unconditionally unrelated |
| Conditional independence: $x_i \perp x_j \mid x_{-ij}$ | $\lambda_{ij} = 0$ | No edge in the MRF; conditionally unrelated |

These two are **different**:
- $\sigma_{ij} = 0$ does **not** imply $\lambda_{ij} = 0$ (two variables may be marginally uncorrelated but conditionally dependent through confounders).
- $\lambda_{ij} = 0$ does **not** imply $\sigma_{ij} = 0$ (conditional independence does not imply marginal independence).

### 5.4 Full Conditional Distribution (Markov Blanket)

For any node $x_i$, its conditional distribution given all other nodes $x_{-i}$ is obtained by collecting all terms in $\log p(x)$ that involve $x_i$:

$$\log p(x_i \mid x_{-i}) = -\frac{1}{2}\lambda_{ii}x_i^2 + x_i\!\left(h_i - \sum_{j \neq i}\lambda_{ij}x_j\right) + \text{const}$$

Completing the square:

$$\boxed{x_i \mid x_{-i} \sim \mathcal{N}\!\left(\frac{h_i - \displaystyle\sum_{j\neq i}\lambda_{ij}x_j}{\lambda_{ii}},\;\; \frac{1}{\lambda_{ii}}\right)}$$

where $h_i = (\Lambda\mu)_i$.

For the zero-mean case ($\mu = 0$, so $h = 0$):

$$x_i \mid x_{-i} \sim \mathcal{N}\!\left(-\frac{\displaystyle\sum_{j\neq i}\lambda_{ij}x_j}{\lambda_{ii}},\;\; \frac{1}{\lambda_{ii}}\right) = \mathcal{N}\!\left(-\sum_{j\neq i}\frac{\lambda_{ij}}{\lambda_{ii}}x_j,\;\; \frac{1}{\lambda_{ii}}\right)$$

**Key observations:**

1. The conditional mean of $x_i$ depends only on its **Markov blanket** — its neighbours in the graph, i.e., nodes $j$ with $\lambda_{ij} \neq 0$. All other nodes drop out.
2. The conditional variance is $1/\lambda_{ii}$, controlled solely by the diagonal precision.
3. The sign of $\lambda_{ij}$ determines whether neighbour $x_j$ has a positive or negative influence on the conditional mean of $x_i$.

---

## 6. Comparing GBN and GMN

| Aspect | Gaussian BN (directed) | Gaussian MN (undirected) |
|--------|----------------------|--------------------------|
| **Graph** | DAG | Undirected graph |
| **Parameters** | Weights $w_{ij}$, noise variances $\sigma_i^2$ | Precision matrix $\Lambda$ |
| **Independence** | d-separation in DAG | $\lambda_{ij} = 0$ |
| **Conversion** | BN $\to$ MN via **moralisation** (marry parents, drop orientation) | MN $\to$ BN possible but may require triangulation |
| **Covariance** | $\Sigma = (I-W)^{-1}S^2(I-W)^{-\top}$ | $\Sigma = \Lambda^{-1}$ |
| **Learning** | MLE of $w_{ij}$ by OLS regression per node | MLE of $\Lambda$ (graphical lasso for sparse case) |

---

## 7. Structure Learning: Graphical Lasso

Given $N$ i.i.d. samples, the MLE of $\Lambda$ is $\hat{\Sigma}^{-1}$ (sample covariance inverse), which is dense. To encourage **sparsity** (and hence a sparse graph), the **graphical lasso** (Friedman et al., 2008) solves:

$$\hat{\Lambda} = \arg\max_{\Lambda \succ 0}\;\left[\log\det\Lambda - \text{tr}(\hat{\Sigma}\Lambda) - \rho\|\Lambda\|_1\right]$$

where $\|\Lambda\|_1 = \sum_{i,j}|\lambda_{ij}|$ is the element-wise $\ell_1$ penalty (excluding diagonal) and $\rho > 0$ controls sparsity. Entries driven to exactly zero correspond to absent edges.

---

## 8. References

1. **Lauritzen, S. L.** (1996). *Graphical Models*. Oxford University Press. [Definitive mathematical treatment of Gaussian and discrete graphical models.]
2. **Whittaker, J.** (1990). *Graphical Models in Applied Multivariate Statistics*. Wiley. [Classical reference specifically for Gaussian graphical models and covariance selection.]
3. **Koller, D. & Friedman, N.** (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press. Ch. 7 (Gaussian networks). [Comprehensive reference covering both directed and undirected Gaussian models.]
4. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §8.1 (Bayesian networks), §8.3 (Markov random fields), §8.4 (Inference in graphical models).
5. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. §19.1–19.4 (Undirected graphical models and Gaussian MRFs).
6. **Friedman, J., Hastie, T., & Tibshirani, R.** (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432–441. [Graphical lasso algorithm for sparse precision matrix estimation.]
7. **Wright, S.** (1921). Correlation and causation. *Journal of Agricultural Research*, 20(7), 557–585. [Original paper on path analysis — the precursor to linear Gaussian BNs.]
8. **Dempster, A. P.** (1972). Covariance selection. *Biometrics*, 28(1), 157–175. [First systematic treatment of sparse inverse covariance (precision matrix) estimation.]

# Spectral Clustering

> **Keywords:** graph Laplacian, normalised cut, affinity matrix, Rayleigh quotient, eigenvectors, k-means, connectivity-based clustering

---

## 1. Motivation & Intuition

Classical clustering algorithms like **k-means** and **GMM** partition data based on **compactness** — they group points that are close in Euclidean space. This fails for clusters with non-convex shapes, varying densities, or complex topological structure (e.g., two interleaved spirals, concentric rings).

**Spectral clustering** instead captures **connectivity**: two points belong to the same cluster if they are connected through a dense region of the data, regardless of absolute distance. It represents the data as a **weighted graph** where edge weights encode pairwise similarity, then partitions the graph by cutting the fewest (and weakest) edges between clusters.

The key insight is that the **graph Laplacian** — a matrix derived from the similarity graph — encodes the cluster structure in its eigenvectors. The bottom $K$ eigenvectors of the Laplacian embed the data into a $K$-dimensional space where standard clustering (k-means) easily separates the clusters.

**Analogy.** Imagine the data as a landscape. Spectral clustering finds valleys (dense connected regions) separated by ridges (sparse connections). The graph Laplacian is sensitive to these ridges; its eigenvectors flatten the landscape within each valley.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1973 | **Fiedler** introduces the graph Laplacian and the "Fiedler vector" (second-smallest eigenvector) for graph bisection. |
| 1990 | **Pothen, Simon & Liou** use Fiedler vectors for sparse matrix reordering — an early spectral partitioning method. |
| 2000 | **Shi & Malik** introduce the **Normalised Cut (Ncut)** criterion and show it leads to a generalised eigenvector problem — the foundation of modern spectral clustering. |
| 2001 | **Ng, Jordan & Weiss** introduce a clean spectral clustering algorithm (NJW): build affinity matrix, compute normalised Laplacian eigenvectors, run k-means. |
| 2003 | **Belkin & Niyogi** introduce Laplacian Eigenmaps, connecting spectral clustering to manifold learning. |
| 2007 | **von Luxburg** publishes the definitive tutorial on spectral clustering, unifying and clarifying the field. |

---

## 3. Two Philosophies of Clustering

| Philosophy | Criterion | Algorithm | Fails when |
|---|---|---|---|
| **Compactness** | Points close to their centroid | k-means, GMM | Non-convex clusters |
| **Connectivity** | Points connected through dense regions | Spectral clustering | Very large graphs (expensive eigendecomposition) |

---

## 4. Graph Construction

### 4.1 Weighted Graph

Represent $N$ data points $\{x_i\}_{i=1}^N$ as a weighted undirected graph $G = (V, E)$:

- $V = \{1, 2, \ldots, N\}$: nodes (data points).
- $W = [w_{ij}] \in \mathbb{R}^{N\times N}$: edge weight matrix (affinity/similarity matrix), symmetric with $w_{ij} \geq 0$.

A common choice is the **Gaussian (RBF) kernel**:

$$w_{ij} = \begin{cases} \exp\!\left(-\dfrac{\|x_i - x_j\|^2}{2\sigma^2}\right) & \text{if } (i,j) \in E \\ 0 & \text{otherwise} \end{cases}$$

The bandwidth $\sigma$ controls the neighbourhood scale. The graph structure (which edges exist) is typically chosen by $\epsilon$-neighbourhood or $k$-nearest-neighbours.

### 4.2 Degree Matrix and Graph Laplacian

The **degree** of node $i$ is the total weight of all its edges:

$$d_i = \sum_{j=1}^N w_{ij}$$

Collected into a diagonal matrix:

$$D = \text{diag}(d_1, d_2, \ldots, d_N) = \text{diag}(W\mathbf{1}_N)$$

The **unnormalised graph Laplacian** is:

$$L = D - W$$

**Key properties of $L$:**
- $L$ is symmetric positive semi-definite.
- $L\mathbf{1} = 0$: the all-ones vector is always an eigenvector with eigenvalue 0.
- The number of connected components of $G$ equals the multiplicity of eigenvalue 0.
- For a connected graph, the second-smallest eigenvalue (Fiedler value) $\lambda_2 > 0$ measures connectivity — larger $\lambda_2$ means harder to cut.

---

## 5. Graph Cutting Objectives

### 5.1 Minimum Cut

Partition $V$ into $K$ disjoint sets $A_1, \ldots, A_K$ (with $\bigcup_k A_k = V$). Define the **cut value** between a set $A$ and its complement $\bar{A} = V\setminus A$:

$$w(A, \bar{A}) = \sum_{i\in A,\; j\in \bar{A}} w_{ij}$$

The total cut:

$$\text{cut}(A_1,\ldots,A_K) = \sum_{k=1}^K w(A_k, \bar{A}_k) = \sum_{k=1}^K\big[w(A_k, V) - w(A_k, A_k)\big]$$

Minimising the cut tends to isolate **outlier nodes** (singleton clusters with few connections) rather than balanced partitions. This is undesirable.

### 5.2 Normalised Cut (Ncut)

**Shi & Malik (2000)** propose normalising each cut by the **volume** of the cluster, where $\text{vol}(A_k) = \sum_{i\in A_k}d_i$ is the sum of degrees of nodes in $A_k$:

$$\text{Ncut}(A_1,\ldots,A_K) = \sum_{k=1}^K \frac{w(A_k, \bar{A}_k)}{\text{vol}(A_k)} = \sum_{k=1}^K\frac{w(A_k,\bar{A}_k)}{\sum_{i\in A_k}d_i}$$

Dividing by volume penalises cutting off small, isolated clusters — it favours **balanced** partitions. The objective is:

$$\{\hat{A}_k\}_{k=1}^K = \arg\min_{\{A_k\}} \text{Ncut}(A_1,\ldots,A_K)$$

---

## 6. From Ncut to the Eigenvalue Problem

### 6.1 Matrix Formulation

Encode the partition using an indicator matrix $Y \in \{0,1\}^{N\times K}$, where $y_i \in \{0,1\}^K$ is the one-hot cluster assignment of node $i$:

$$Y_{ik} = 1 \iff i \in A_k, \qquad \sum_{k=1}^K Y_{ik} = 1$$

**Computing $Y^\top Y$:** Since $y_i$ is one-hot:

$$Y^\top Y = \sum_{i=1}^N y_i y_i^\top = \begin{pmatrix} N_1 & & \\ & \ddots & \\ & & N_K \end{pmatrix} = \text{diag}(N_1,\ldots,N_K)$$

where $N_k = |A_k|$ is the number of nodes in cluster $k$.

**Computing $P = Y^\top D Y$:**

$$(Y^\top D Y)_{kl} = \sum_{i=1}^N Y_{ik}\,d_i\,Y_{il}$$

Since $y_i$ is one-hot, $Y_{ik}Y_{il} = 0$ for $k \neq l$. For $k = l$:

$$(Y^\top DY)_{kk} = \sum_{i\in A_k} d_i = \text{vol}(A_k)$$

So $P = Y^\top DY = \text{diag}(\text{vol}(A_1),\ldots,\text{vol}(A_K))$.

**Computing $Y^\top WY$:**

$$(Y^\top WY)_{kl} = \sum_{i=1}^N\sum_{j=1}^N Y_{ik}\,w_{ij}\,Y_{jl}$$

For $k=l$: $\sum_{i\in A_k, j\in A_k}w_{ij} = w(A_k, A_k)$. So $Y^\top WY = \text{diag}(w(A_1,A_1),\ldots,w(A_K,A_K))$.

**Therefore:**

$$Y^\top(D-W)Y = Y^\top DY - Y^\top WY = \text{diag}\!\big(\text{vol}(A_k) - w(A_k,A_k)\big) = \text{diag}\!\big(w(A_k,\bar{A}_k)\big)$$

### 6.2 Ncut as a Trace

$$\text{Ncut}(V) = \sum_{k=1}^K\frac{w(A_k,\bar{A}_k)}{\text{vol}(A_k)} = \text{tr}\!\left(\text{diag}\!\big(w(A_k,\bar{A}_k)\big)\cdot\text{diag}\!\big(\text{vol}(A_k)\big)^{-1}\right)$$

$$= \text{tr}\!\Big(Y^\top(D-W)Y\cdot(Y^\top DY)^{-1}\Big) = \text{tr}\!\Big(Y^\top LY\,(Y^\top DY)^{-1}\Big)$$

The optimisation problem becomes:

$$\hat{Y} = \arg\min_{Y\in\{0,1\}^{N\times K}}\;\text{tr}\!\Big(Y^\top LY\,(Y^\top DY)^{-1}\Big)$$

### 6.3 Relaxation to a Continuous Problem

The integer constraint $Y\in\{0,1\}^{N\times K}$ makes this NP-hard. Relax to real-valued $\tilde{Y}\in\mathbb{R}^{N\times K}$ with $\tilde{Y}^\top\tilde{Y}=I$ (orthonormality replaces the one-hot constraint).

**Change of variables:** Let $F = D^{1/2}Y$. Then $Y^\top DY = F^\top F$ and $Y^\top LY = F^\top D^{-1/2}LD^{-1/2}F$. The problem becomes:

$$\hat{F} = \arg\min_{F^\top F=I}\;\text{tr}\!\big(F^\top L_{\text{norm}}F\big)$$

where $L_{\text{norm}} = D^{-1/2}LD^{-1/2}$ is the **symmetric normalised Laplacian**.

By the **Courant-Fischer (Rayleigh-Ritz) theorem**, this is minimised by taking $F$ to be the matrix of the $K$ eigenvectors corresponding to the $K$ smallest eigenvalues of $L_{\text{norm}}$.

Equivalently, solve the **generalised eigenvalue problem**:

$$L\,u = \lambda D\,u$$

The $K$ generalised eigenvectors with smallest eigenvalues give $Y$.

---

## 7. The Spectral Clustering Algorithm

```
Input: data {x_1, ..., x_N}, number of clusters K, bandwidth sigma

1. Construct affinity matrix W:
     w_{ij} = exp(-||x_i - x_j||^2 / 2sigma^2)   [or k-NN graph]

2. Compute degree matrix D = diag(W 1) and Laplacian L = D - W.

3. Compute normalised Laplacian:
     L_norm = D^{-1/2} L D^{-1/2}    [symmetric normalised]
     (Alternatively: L_rw = D^{-1} L   [random-walk normalised])

4. Compute the K smallest eigenvectors u_1, ..., u_K of L_norm.
   Form the matrix U = [u_1 | ... | u_K] in R^{N x K}.

5. Normalise rows of U:  U_hat_{ij} = U_{ij} / (sum_k U_{ik}^2)^{1/2}

6. Treat each row of U_hat as a point in R^K.
   Run k-means on these N points to get K clusters.

7. Assign data point x_i to cluster k if row i was assigned to cluster k.
```

**Why does this work?** For a graph with $K$ perfectly separated connected components, the normalised Laplacian has $K$ zero eigenvalues. The corresponding eigenvectors are indicator vectors of the components — clustering them with k-means perfectly recovers the partition. For real data, the clusters are approximately separated, and the eigenvectors are smooth approximations of these indicator vectors.

---

## 8. Comparison with K-means

| Aspect | K-means | Spectral Clustering |
|---|---|---|
| **Cluster shape** | Convex (Voronoi cells) | Arbitrary (connectivity-based) |
| **Distance used** | Euclidean | Graph-based (can use any kernel) |
| **Scalability** | $O(NKd)$ per iteration | $O(N^3)$ for dense graph (eigendecomp) |
| **Number of clusters** | Must choose $K$ | Must choose $K$ |
| **Global optimum** | Not guaranteed | Not guaranteed (NP-hard relaxed) |
| **Initialisation** | Sensitive | Less sensitive (eigenvectors are unique) |

---

## 9. References

1. **Shi, J. & Malik, J.** (2000). Normalized cuts and image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(8), 888–905. [Original Ncut paper; shows the graph-partitioning connection to the generalised eigenvector problem.]
2. **Ng, A. Y., Jordan, M. I., & Weiss, Y.** (2001). On spectral clustering: Analysis and an algorithm. *Advances in Neural Information Processing Systems 14 (NeurIPS 2001)*. [NJW algorithm; clean three-step procedure.]
3. **von Luxburg, U.** (2007). A tutorial on spectral clustering. *Statistics and Computing*, 17(4), 395–416. [Definitive tutorial covering all variants, theory, and practical guidelines.]
4. **Belkin, M. & Niyogi, P.** (2003). Laplacian eigenmaps for dimensionality reduction and data representation. *Neural Computation*, 15(6), 1373–1396. [Connection to manifold learning.]
5. **Fiedler, M.** (1973). Algebraic connectivity of graphs. *Czechoslovak Mathematical Journal*, 23(2), 298–305. [Original work on graph Laplacian and the Fiedler vector.]
6. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §12.3 (Kernel PCA, which shares the spectral structure).
7. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. §25.5 (Spectral clustering).

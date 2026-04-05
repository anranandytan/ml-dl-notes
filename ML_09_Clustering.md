# ML 09 — Clustering

> **Keywords:** K-means, coordinate descent, convergence, hierarchical agglomerative clustering, linkage, dendrogram, distance metric

---

## 1. Two Philosophies of Clustering

| Philosophy | Criterion | Algorithm |
|---|---|---|
| **Compactness** | Points should be close to their cluster centre | K-means, K-medians, GMM |
| **Connectivity** | Points connected through dense regions | Hierarchical agglomerative, spectral clustering |

The right choice depends on the cluster shape: K-means excels for spherical clusters; hierarchical methods handle elongated or manifold-structured clusters.

---

## 2. K-Means

### 2.1 Problem Formulation

**Input:** data points $\{x_1,\ldots,x_n\}\subset\mathbb{R}^d$, number of clusters $K$.

**Output:** cluster centres $\{c_1,\ldots,c_K\}$ and a partition $\{\text{cluster}_1,\ldots,\text{cluster}_K\}$ of the data.

**Objective:**

$$\min_{c_1,\ldots,c_K,\,\text{cluster}_1,\ldots,\text{cluster}_K}\;\text{cost} = \sum_{k=1}^K\sum_{x_i\in\text{cluster}_k}\|x_i - c_k\|_2^2$$

This is minimised jointly over the cluster assignment and cluster centres.

### 2.2 Algorithm (Coordinate Descent)

K-means alternates between two minimisation steps:

**Step 1 — Minimise over assignment (fix centres):**

$$\text{cluster}_k = \{x_i : k = \arg\min_j \|x_i - c_j\|_2^2\}$$

Assign each point to its nearest cluster centre. This is the closest point to each $c_k$.

**Step 2 — Minimise over centres (fix assignment):**

$$c_k = \frac{1}{|\text{cluster}_k|}\sum_{x_i\in\text{cluster}_k}x_i$$

Move each cluster centre to the mean of its assigned points.

**K-Means algorithm:**

```
Randomly initialise cluster centres c_1, ..., c_K.
Repeat until convergence:
  1. Assign each point to its nearest cluster centre.
  2. Recompute each centre as the mean of its assigned points.
```

### 2.3 Convergence Analysis

K-means is a **coordinate descent** algorithm: each step decreases (or leaves unchanged) the objective. Since the number of possible assignments is finite ($K^n$), the algorithm must converge in finite iterations.

**Convergence guarantee:** the cost decreases at every iteration and the algorithm terminates. However, it may converge to a **local minimum**, not the global minimum. The solution depends on the initialisation.

**K-means++ initialisation** (not in original notes but standard practice): choose centres sequentially, with each new centre selected with probability proportional to $\|x_i-c_{\text{nearest}}\|_2^2$. This gives a $O(\log K)$ approximation guarantee in expectation.

### 2.4 Choosing $K$

K-means requires specifying $K$ in advance. Common strategies:
- **Elbow method:** plot cost vs. $K$; choose the "elbow" where additional clusters give diminishing returns.
- **Silhouette score:** measures how similar each point is to its own cluster vs. other clusters.
- **Gap statistic:** compare cost to a null reference distribution.

### 2.5 Properties

**Pros:**
- Simple and fast: $O(nKd)$ per iteration.
- Works well for spherical, well-separated clusters.
- Has a clear objective function.

**Cons:**
- Sensitive to initialisation.
- Assumes spherical clusters of similar size.
- Does not work well for elongated or non-convex clusters.
- Assumes Euclidean distance — changing the metric substantially changes results.
- Must specify $K$ in advance.

---

## 3. Hierarchical Agglomerative Clustering

### 3.1 Algorithm

**Idea:** start with each point in its own cluster; iteratively merge the two closest clusters.

```
Initialise: each point {x_i} is its own cluster.
Repeat until one cluster remains:
  Find the two clusters with smallest distance d(A, B).
  Merge them into a single cluster.
Record the merge order and distances → dendrogram.
```

The result is a **dendrogram** — a tree showing the complete merge history. Cutting the dendrogram at any height gives a partition of the data into clusters.

### 3.2 Linkage Criteria

The choice of $d(A,B)$ — how to measure the distance between two clusters — is crucial:

| Linkage | $d(A,B)$ | Tends to produce |
|---|---|---|
| **Single** | $\min_{a\in A,b\in B}\|a-b\|$ | Long chain-like clusters |
| **Complete** | $\max_{a\in A,b\in B}\|a-b\|$ | Compact, roughly equal-size clusters |
| **Average** | $\frac{1}{|A||B|}\sum_{a,b}\|a-b\|$ | Balanced clusters |
| **Ward** | Increase in total within-cluster variance when merged | Compact clusters; similar to K-means |

### 3.3 When to Use Hierarchical vs. K-Means

| | K-Means | Hierarchical Agglomerative |
|--|--|--|
| **Cluster shape** | Spherical | Arbitrary (manifold-structured) |
| **$K$ specification** | Required upfront | Can be chosen after seeing the dendrogram |
| **Complexity** | $O(nKd\cdot\text{iters})$ | $O(n^2)$ to $O(n^2\log n)$ |
| **Sensitivity to metric** | Yes | **Very** sensitive — both metric and linkage matter |
| **Robustness** | Sensitive to initialisation | Deterministic |

> K-Means is excellent for spherical clusters, whereas Hierarchical Agglomerative Clustering is useful when the data has manifold structures.

### 3.4 Sensitivity to Distance Metric

Both algorithms depend heavily on the distance metric:
- **Standard Euclidean** treats all dimensions equally.
- **Weighted Euclidean** can emphasise important features.
- **Cosine similarity** ignores magnitude (useful for text/embeddings).
- **Manhattan ($\ell_1$)** is more robust to outliers.

Changing the distance metric can **substantially** alter the clustering. Always validate that the metric reflects the relevant notion of similarity for the problem.

---

## 4. Relationship to GMM and Soft Clustering

K-means is the **hard-assignment limit** of the Gaussian Mixture Model (GMM):
- GMM assigns each point a **soft** (fractional) membership in each cluster.
- As the component variances shrink to zero, GMM collapses to K-means.

For a full treatment of GMM and the EM algorithm, see **DL_04_EM.md**.

For connectivity-based clustering via the graph Laplacian, see **DL_16_SpectralClustering.md**.

---

## 5. References

1. **Lloyd, S.** (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137. [The K-means algorithm (originally a 1957 Bell Labs report).]
2. **Arthur, D. & Vassilvitskii, S.** (2007). K-means++: The advantages of careful seeding. *Proceedings of SODA 2007*, 1027–1035. [K-means++ initialisation with $O(\log K)$ approximation guarantee.]
3. **Ward, J. H.** (1963). Hierarchical grouping to optimise an objective function. *Journal of the American Statistical Association*, 58(301), 236–244. [Ward linkage criterion.]
4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Ch. 14 (Unsupervised learning: K-means, hierarchical clustering).
5. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Ch. 25 (Clustering).

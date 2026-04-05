# ML 02 — Decision Trees

> **Keywords:** information gain, Gini index, C4.5, CART, pruning, GOSDT, branch-and-bound, optimal sparse trees

---

## 1. Motivation & Intuition

A decision tree partitions the input space into rectangular regions by asking a sequence of yes/no questions about the features. At each internal node, one feature is tested; each branch corresponds to a range of that feature's values; each leaf assigns a prediction.

**Key appeal:**
- Naturally **interpretable** — a human can follow the path from root to leaf.
- Handles **nonlinearities** and **interactions** without feature engineering.
- Works for both classification and regression.
- Can handle imbalanced data by reweighting.

**Key weakness:** a single tree grown without care is a high-variance estimator — small changes in training data can produce a very different tree. Ensemble methods (random forests, boosting) address this.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1963 | Morgan & Sonquist introduce AID (Automatic Interaction Detection) — an early recursive partitioning method. |
| 1984 | **Breiman, Friedman, Olshen & Stone** introduce **CART** (Classification And Regression Trees). |
| 1993 | **Quinlan** introduces **C4.5**, improving on his earlier ID3 algorithm with better pruning and continuous features. |
| 2022 | **Lin, Zhong, Seltzer & Rudin** introduce **GOSDT** — the first practical algorithm for *optimal* sparse decision trees with provable guarantees. |

---

## 3. Greedy Tree Induction (General Procedure)

All standard algorithms grow a tree greedily:

1. **Start** at the root with all training data.
2. **Choose** a feature and split point to partition the data, according to a splitting criterion.
3. **Recurse** on each child node.
4. **Stop** when a stopping condition is met:
   - All observations in a node have the same class (pure leaf).
   - No features remain to split on.
   - The node has fewer than $n_{\min}$ observations.
5. **Prune** the grown tree to reduce overfitting.
6. **Assign** each leaf the majority class (classification) or mean response (regression).

**Training probability at leaf $j$:**

$$\left[\frac{\#\text{pos}_j}{\#\text{pos}_j + \#\text{neg}_j},\;\frac{\#\text{neg}_j}{\#\text{pos}_j + \#\text{neg}_j}\right]$$

---

## 4. Splitting Criteria

### 4.1 Entropy and Information Gain (C4.5)

**Shannon entropy** of a distribution $[p_1, \ldots, p_M]$:

$$H(p_1,\ldots,p_M) = -\sum_{j=1}^M p_j \log p_j$$

Entropy is 0 when the distribution is pure (one class dominates) and maximal when uniform.

**Information Gain** of splitting on feature $A$, which partitions the data into $J$ branches:

$$\text{Gain}(S, A) = H\!\left(\frac{\#\text{pos}}{\#\text{pos}+\#\text{neg}}, \frac{\#\text{neg}}{\#\text{pos}+\#\text{neg}}\right) - \sum_{j=1}^J \frac{\#\text{pos}_j + \#\text{neg}_j}{\#\text{pos}+\#\text{neg}}\,H\!\left(\frac{\#\text{pos}_j}{\#\text{pos}_j+\#\text{neg}_j}, \frac{\#\text{neg}_j}{\#\text{pos}_j+\#\text{neg}_j}\right)$$

This is the reduction in entropy achieved by the split. A gain of 0 means the feature carries no information about the label.

**Information Gain Ratio (IGR)** — corrects for features with many values (which always achieve high gain):

$$\text{IGR}(S, A) = \frac{\text{Gain}(S,A)}{\text{SplitInfo}(S,A)}, \qquad \text{SplitInfo}(S,A) = -\sum_{j=1}^J \frac{|S_j|}{|S|}\log\frac{|S_j|}{|S|}$$

### 4.2 Gini Index (CART)

$$\text{Gini}(p) = 2p(1-p) = 2 \times \text{Var}(\text{Bernoulli}(p))$$

This equals twice the variance of a Bernoulli random variable with probability $p$. Like entropy, it is 0 for pure nodes and maximal at $p = 0.5$.

**Misclassification error:** $1 - \max(p, 1-p)$ is another option, but it is less sensitive to changes in class proportions than entropy or Gini and is rarely used for splitting.

---

## 5. C4.5

**Splitting criterion:** Information Gain Ratio.

**Pruning strategy (three options for each subtree):**

1. **Leave as-is** — keep the subtree unchanged.
2. **Collapse to a leaf** — replace the subtree with a leaf labelled by the majority class in the data $S$ at that node.
3. **Replace with best subtree** — replace the subtree with the most-used child branch.

C4.5 computes an upper confidence bound on the error probability for each option using a binomial bound with $\alpha = 0.25$, and chooses the option with the lowest bound.

**Advantages vs. CART:**
- Handles multi-way splits naturally.
- Uses information gain ratio, which penalises high-arity features.

**Disadvantages:**
- Default settings produce trees that are completely uninterpretable (too deep, too many branches).

---

## 6. CART — Classification and Regression Trees

**Splitting criterion:** Gini index (classification) or squared error (regression).

**Structural restriction:** CART makes only **binary splits** ($\leq s$ vs. $> s$).

### 6.1 Classification

At each node, find feature $j$ and split point $s$ to minimise the weighted Gini impurity across the two children.

**Pruning (Minimum Cost Complexity):** Each candidate subtree is assigned a cost:

$$\text{cost}(\text{subtree}) = \sum_{\text{leaves } j}\sum_{x_i\in\text{leaf }j} \mathbf{1}[y_i \neq \text{leaf's class}] + C \cdot \#\text{leaves}$$

Choose the subtree with the lowest cost. Increasing $C$ prunes more aggressively.

### 6.2 Regression

For real-valued targets, find feature $j$ and split point $s$ that minimise the total within-leaf variance:

$$\min_{j,s}\left[\min_{C_1}\sum_{x_i:x_i^{(j)}\leq s}(y_i-C_1)^2 + \min_{C_2}\sum_{x_i:x_i^{(j)}>s}(y_i-C_2)^2\right]$$

The optimal leaf values are simply the means: $C_1 = \bar{y}_{\text{left}}$, $C_2 = \bar{y}_{\text{right}}$.

**Regression tree cost:**

$$\text{cost} = \sum_{\text{leaves }j}\sum_{x_i\in S_j}(y_i-\bar{y}_{S_j})^2 + C\cdot\#\text{leaves}$$

### 6.3 Analytical Bound on Tree Size

**Basic Size Bound.** Given any reference objective value $R^c$, any optimal tree $\text{tree}^*$ satisfies:

$$\#\text{leaves in tree}^* \leq \left\lfloor \frac{R^c}{C}\right\rfloor$$

This is because each additional leaf must reduce the unregularised objective by at least $C$.

### 6.4 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Interpretable | Greedy — not globally optimal |
| Handles nonlinearities | No proof of optimality |
| No feature scaling needed | Tends to struggle with imbalanced data |
| Can be easily reweighted | More interpretable but less accurate than C4.5 |

---

## 7. GOSDT — Generalised and Scalable Optimal Sparse Decision Trees

GOSDT finds **provably optimal** sparse decision trees — no splitting or pruning heuristics.

### 7.1 Objective

$$R(\text{tree},\{(x_i,y_i)\}) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[y_i \neq \hat{y}_i^{\text{tree}}] + \lambda \cdot \#\text{leaves}(\text{tree}), \quad \text{depth}(\text{tree}) \leq D$$

- $\lambda$ (soft): penalises the number of leaves (sparsity).
- $D$ (hard): caps the depth of the tree.

### 7.2 Preprocessing: Binarisation

Convert all features to binary variables:
- For a continuous feature, find all midpoints between consecutive distinct values and create indicator variables $\mathbf{1}[x^{(j)} \geq \theta]$.
- To limit explosion in binary features: run a fast greedy algorithm (C4.5, random forest) first and only keep splits that were used by those algorithms.

### 7.3 Key Data Structures

Each subproblem (a node in the search space) is identified by a **bit-vector** $s$ indicating which training points fall into that node.

| Structure | Purpose |
|---|---|
| **Dependency graph $G$** | Stores all subproblems, their parent-child relationships, and bounds on the objective |
| **Priority queue $Q$** | Orders subproblems for exploration (higher priority = more promising) |

### 7.4 Bounds

For each subproblem $p$ (identified by data subset $s$):

$$p.\text{lb} = 0 + 2\lambda \qquad \text{(at least 2 leaves needed)}$$
$$p.\text{ub} = \frac{\#\text{minority labels in }s}{n} + \lambda \qquad \text{(trivial leaf solution)}$$

**Theorem 1.** If $p.\text{ub} - p.\text{lb} \leq 0$, then the trivial (leaf) solution is already optimal for this subproblem — no further splitting can improve it.

### 7.5 FindOrCreateNode Algorithm

```
FindOrCreateNode(G, s):
  if G.find(s) = NULL:
    p.id  ← s
    p.lb  ← 0 + 2λ
    p.ub  ← (# minority labels in s)/n + λ
    if p.ub - p.lb ≤ 0:
      p.lb ← p.ub          [node is already a leaf: bound is tight]
    G.insert(p)
  return G.find(s)
```

### 7.6 Main Branch-and-Bound Loop

```
while p_root.lb ≠ p_root.ub:
  s ← Q.pop()
  p ← G.find(s)
  if p.ub = p.lb: continue    [already solved]

  (lb', ub') ← (∞, ∞)
  for j = 1, ..., #features:
    (s_L, s_R) ← split(s, j)
    p_L ← FindOrCreateNode(G, s_L)
    p_R ← FindOrCreateNode(G, s_R)
    lb' ← min(lb', p_L.lb + p_R.lb)
    ub' ← min(ub', p_L.ub + p_R.ub)

  p.ub ← min(p.ub, ub')
  p.lb ← min(p.ub, max(p.lb, lb'))    [bound update]

  propagate bound updates to parents; push parents to Q

  for j = 1, ..., #features:           [push promising children]
    if p_L.lb + p_R.lb < p_L.ub + p_R.ub  and  ≤ p.ub:
      Q.push(s_L, s_R)

return G
```

**Pruning logic:** If the sum of children's lower bounds already exceeds the parent's upper bound, that split can never be part of any optimal solution — prune immediately.

### 7.7 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Provably optimal (or near-optimal) | Cannot prove optimality for very large datasets |
| No splitting/pruning heuristics | More complex to implement |
| Handles custom objectives | Scalability still an active research area |
| Interpretable, small models | |

---

## 8. Summary

```
Splitting criteria:
  C4.5  → Information Gain Ratio    (multi-way splits, entropy-based)
  CART  → Gini index / MSE          (binary splits, simple formula)

Pruning:
  C4.5  → Upper confidence bound on error (binomial, α=0.25)
  CART  → Minimum Cost Complexity: cost = misclassification + C·#leaves

Optimal trees:
  GOSDT → Branch-and-bound with bit-vector subproblems
          lb = 0 + 2λ,  ub = minority_fraction + λ
          Optimise: (1/n)·misclassification + λ·#leaves
```

---

## 9. References

1. **Quinlan, J. R.** (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann. [The definitive C4.5 reference.]
2. **Breiman, L., Friedman, J., Olshen, R., & Stone, C.** (1984). *Classification and Regression Trees*. Wadsworth. [Original CART book.]
3. **Lin, J., Zhong, C., Hu, D., Rudin, C., & Seltzer, M.** (2020). Generalized and scalable optimal sparse decision trees. *Proceedings of ICML 2020*. [GOSDT algorithm.]
4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Ch. 9 (Additive models, trees, and related methods).
5. **Mitchell, T.** (1997). *Machine Learning*. McGraw-Hill. Ch. 3 (Decision tree learning). [Accessible introduction including ID3.]

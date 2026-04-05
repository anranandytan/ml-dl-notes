# ML 03 — Ensembles, Variable Importance, Boosting, and GAMs

> **Keywords:** bagging, random forest, OOB error, variable importance, model reliance, SHAP, AdaBoost, coordinate descent, exponential loss, GAM, FastSparse

---

## 1. Random Forest

### 1.1 Motivation

A single decision tree has high variance: a small change in the training data can produce a very different tree. **Random forests** reduce variance by averaging many trees that are deliberately made diverse.

> **Core idea:** if you average many different yet accurate models, the variance decreases proportionally, while bias stays low.

### 1.2 Bagging (Bootstrap Aggregating)

Draw $n$ points from the training set **with replacement** (a bootstrap sample). Grow a full tree on this sample. Repeat $T$ times and average (regression) or majority-vote (classification) the $T$ trees.

Bagging reduces variance without increasing bias, because the ensemble expectation equals the single-tree expectation.

### 1.3 Decision Forests (Random Forest)

Random forests improve on bagging by **decorrelating** the trees: at each split, only $m$ randomly chosen features (out of $p$ total) are considered.

**Algorithm:**

```
For t = 1, ..., T:
  1. Draw a bootstrap sample of size n from training data.
  2. Grow tree_t using:
       - At each split, pick m features at random (out of p).
       - Evaluate splitting criterion on all m candidates.
       - Split on the best feature.
       - Stop splitting when node has < n_min observations.
Output: majority vote (classification) or mean (regression) of all T trees.
```

**Effect of $m$:**
- Small $m$: trees are less correlated (lower variance), but each tree is weaker (higher bias).
- Large $m$ (→ $p$): trees resemble bagged trees (more correlated, less diverse).
- Typical defaults: $m = \lfloor\sqrt{p}\rfloor$ for classification, $m = \lfloor p/3 \rfloor$ for regression.

**Comparison with a single decision tree:**

| Modification | Effect |
|---|---|
| Bootstrap resamples | Diverse trees → reduces variance |
| Random feature subset at each split | Less correlation among trees → reduces variance further |
| No pruning | Trees fit tightly → reduces bias |
| Majority vote / average | Smooths individual errors → reduces variance |

### 1.4 Out-of-Bag (OOB) Error

Each bootstrap sample omits roughly $1/e \approx 37\%$ of the training points. These **out-of-bag** points provide a free validation set for each tree.

**OOB error:** For each point $x_i$, predict using only the trees for which $x_i$ was OOB. Average these predictions to get an unbiased error estimate — no separate validation set needed.

### 1.5 Variable Importance (Permutation Importance)

For tree $\text{tree}_t$ and feature $j$:

1. Compute the OOB error $\text{err}_t$ on $\text{OOB}_t$.
2. Randomly **permute** only feature $j$ in $\text{OOB}_t$ → $\text{OOB}_{t,\text{perm}}$.
3. Compute the new OOB error $\text{err}_{t,\text{perm}}$ on the permuted data.
4. The raw importance of feature $j$ is the average increase in error:

$$\text{VarImportance}(j) = \frac{1}{T}\sum_t\left(\text{err}_{t,\text{perm}} - \text{err}_t\right)$$

A large positive value means permuting feature $j$ hurts performance — the feature is important.

**Pros:** Complex and powerful; has a natural notion of variable importance.  
**Cons:** Black box; prone to overfitting without careful tuning; slow to train.

---

## 2. Variable Reliance

Beyond "is feature $j$ important to model $f$?", several richer questions arise. This section covers the key measures.

### 2.1 Model Reliance (MR)

**Idea:** scramble feature $j$ (breaking its relationship with the label and other features) and measure how much performance degrades.

**Swapping-based MR** (more principled than random permutation):

$$\text{MR}_{\text{diff}}(f, j) = \frac{1}{n(n-1)}\sum_{i\neq k}\text{loss}(y_i, f([x_{i,j\text{swapped with }k}]))-\text{Loss}(f(\mathbf{X}),\mathbf{Y})$$

$$\text{MR}_{\text{ratio}}(f, j) = \frac{\text{ScrambledLoss}}{\text{Loss}(f(\mathbf{X}),\mathbf{Y})}$$

- $\text{MR}_{\text{diff}} = 0$ or $\text{MR}_{\text{ratio}} = 1$: feature $j$ is not used by $f$.

**Caveat:** scrambled data points may fall outside the training distribution, so the measure reflects model reliance, not necessarily causal importance.

### 2.2 Conditional Model Reliance (CMR)

CMR isolates the **unique information** in feature $j$ that cannot be predicted from the other features.

**Steps:**

1. For each point $i$, predict $x_{i,1}$ from all other features: $\hat{x}_{i,1} = \hat{\mathbb{E}}(X_{i,1}|X_{i,-1}=x_{i,-1})$.
2. The unique information for point $i$ is the residual: $x_{i,1} - \hat{x}_{i,1}$.
3. Create a scrambled version: $x_{ik,1}^{\text{scr}} = (x_{i,1}-\hat{x}_{i,1}) + \hat{x}_{k,1}$ (transfer unique info of $i$ to context of $k$).
4. Compute scrambled loss and form the ratio:

$$\text{CMR}_{\text{ratio}}(f, 1) = \frac{\frac{1}{n(n-1)}\sum_{i\neq k}\text{loss}(y_k, f(x_{ik}^{\text{scr}}))}{\text{Loss}(f(\mathbf{X}),\mathbf{Y})}$$

$\text{CMR}_{\text{ratio}} = 1$ means feature 1 provides no unique information to model $f$.

### 2.3 SHAP (SHapley Additive exPlanations)

SHAP explains the contribution of feature $j$ to a **specific prediction** for a specific model.

Based on the Shapley value from cooperative game theory: add features one at a time in every possible order, measure the marginal contribution of feature $j$, and average over all orderings:

$$\phi_j = \sum_{S\subseteq F\setminus\{j\}}\frac{|S|!(|F|-|S|-1)!}{|F|!}\left[f_{S\cup\{j\}}(x_{S\cup\{j\}})-f_S(x_S)\right]$$

where $f_S(x_S) = \mathbb{E}_{\mathbf{X}_{\bar{S}}}[f(x_S, \mathbf{X}_{\bar{S}})|\mathbf{X}_S=x_S]$ is the model's prediction marginalising out the features not in $S$.

**Approximations:**
- Feature independence: $f_S(x_S)\approx \mathbb{E}_{\mathbf{X}_{\bar{S}}}[f(x_S,\mathbf{X}_{\bar{S}})]$.
- Linear model: $f_S(x_S)\approx f([x_S, \mathbb{E}[\mathbf{X}_{\bar{S}}]])$.

### 2.4 Model Class Reliance (MCR) — The Rashomon Set

**Rashomon set:** the set of all models that are nearly as good as a reference model $f_{\text{ref}}$:

$$\text{Rset}(\mathcal{F},\epsilon,f_{\text{ref}}) = \{f\in\mathcal{F}: \text{Loss}(f)\leq\text{Loss}(f_{\text{ref}})+\epsilon\}$$

**MCR** gives the *range* of model reliance values across all models in the Rashomon set:

$$[\text{MCR}_-, \text{MCR}_+] = \left[\min_{f\in\text{Rset}}\text{MR}(f,j),\;\max_{f\in\text{Rset}}\text{MR}(f,j)\right]$$

**Interpretation:**
- $[1, 1.001]$: no good model needs feature $j$ — safe to remove.
- $[1.5, 2.3]$: all good models rely on feature $j$ — it is essential.
- $[1, 4]$: some good models use it, some don't — ambiguous.

**CMCR** extends this to conditional reliance (unique information × Rashomon set).

### 2.5 Summary: Which Measure for Which Question?

| Question | Measure |
|---|---|
| Important to this specific model? | MR |
| Unique information important to this model? | CMR |
| Important for this specific prediction by this model? | SHAP |
| Important to the learning algorithm? | Algorithm Reliance (AR) |
| Important to any good model in a class? | MCR |
| Unique information important to any good model? | CMCR |
| Stable importance across near-optimal models? | RID (Rashomon Important Distribution) |

---

## 3. Boosting and AdaBoost

### 3.1 Boosting Philosophy

**Boosting** iteratively trains weak classifiers with respect to a changing distribution over the training data:
- Increase weights of misclassified points (focus on hard cases).
- Decrease weights of correctly classified points.
- Combine all weak classifiers with learned weights into a strong classifier.

**Why coordinate descent?** The gradient of the boosted objective is expensive or impossible to compute directly (e.g., for decision trees). Coordinate descent — moving one weak classifier coefficient at a time — sidesteps this.

### 3.2 AdaBoost

**Setup:** Training data $\{(x_i, y_i)\}_{i=1}^n$ with $y_i\in\{-1,+1\}$; weak learning algorithm $A$; $T$ rounds.

**Notation:**
- $\mathbf{d}_t$: distribution (weights) over examples at round $t$.
- $h_t$: weak classifier chosen at round $t$.
- $\alpha_t$: coefficient for $h_t$ in the ensemble.
- $\mathbf{M}\in\mathbb{R}^{n\times p}$: margin matrix, $M_{ij} = y_i h_j(x_i)\in\{-1,+1\}$.
- Final classifier: $H(x) = \text{sign}\!\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$.

**Exponential loss** (upper bounds misclassification error):

$$R^{\text{train}}(\boldsymbol{\lambda}) = \frac{1}{n}\sum_{i=1}^n e^{-(\mathbf{M}\boldsymbol{\lambda})_i} \geq \frac{1}{n}\sum_{i=1}^n \mathbf{1}[y_i f(x_i)\leq 0]$$

**AdaBoost update at round $t$:**

At each round, find the weak classifier $h_t$ minimising the weighted error:

$$\text{Error}_t = \sum_{i: h_t(x_i)\neq y_i} d_{t,i}$$

Compute the coefficient:

$$\alpha_t = \frac{1}{2}\ln\!\left(\frac{1-\text{Error}_t}{\text{Error}_t}\right)$$

Update weights (unnormalised):

$$d_{t+1,i} \propto d_{t,i}\cdot e^{-\alpha_t y_i h_t(x_i)}$$

Normalise so $\sum_i d_{t+1,i} = 1$.

**Training error bound.** Define $\gamma_t = \frac{1}{2} - \text{Error}_t$ (the edge of the weak classifier). Then:

$$\frac{1}{n}\sum_i \mathbf{1}[y_i\neq H(x_i)] \leq R^{\text{train}}(\boldsymbol{\lambda}_T) = \prod_{t=1}^T 2\sqrt{\text{Error}_t(1-\text{Error}_t)} = \prod_t\sqrt{1-4\gamma_t^2}$$

Using $1+x\leq e^x$:

$$R^{\text{train}}(\boldsymbol{\lambda}_T)\leq e^{-2\sum_t\gamma_t^2}\leq e^{-2T\gamma_{\text{min}}^2}$$

Training error decreases **exponentially** in the number of rounds, provided each weak classifier does slightly better than random ($\gamma_t > 0$).

### 3.3 AdaBoost as Coordinate Descent

AdaBoost is coordinate descent on the exponential loss $R^{\text{train}}(\boldsymbol{\lambda})$:
- "Run weak learning algorithm" = "choose coordinate $j$" = "choose weak classifier $h_j$".
- "Update $\alpha_t$" = "move along coordinate $j$ by step size $\alpha_t$".

**Why AdaBoost tends not to overfit:**
- It implicitly maximises the **margin** $y_i f(x_i)/\|f\|$ — pushing predictions away from the decision boundary.
- The iterative procedure itself acts as an implicit regulariser, even with no explicit penalty.

---

## 4. Generalised Additive Models (GAMs)

### 4.1 Structure

A GAM decomposes the prediction as a sum of univariate component functions:

$$g(\hat{y}(x)) = \sum_{j=1}^p f_j(x_{\cdot,j})$$

where $g$ is a link function (e.g., identity for regression, logit for classification). No interaction terms between variables, so each $f_j$ can be inspected independently — comparable interpretability to linear models, but with arbitrary nonlinearity per feature.

### 4.2 Training with AdaBoost (Boosted Stumps)

Each $f_j$ is built from a sum of **decision stumps** (trees with one split):

$$f_j(x_{\cdot,j}) = \sum_\ell \lambda_{j,\ell}\,\text{stump}(j,\ell)(x)$$

At each AdaBoost iteration, a weighted step function is added to one component function. After training, regroup stumps by feature:

$$g(\hat{y}(x)) = \sum_t\alpha_t h_t(x) = \sum_j\sum_\ell \lambda_{j,\ell}\,\text{stump}(j,\ell)(x), \quad \lambda_{j,\ell} = \sum_{t:\, h_t=\text{stump}(j,\ell)}\alpha_t$$

**Backfitting** (alternative training): iterate over features, model the residual $y - \sum_{k\neq j}f_k$, and update $f_j$.

**GA$^2$M (pairwise interactions):** add interaction terms via forward selection:

$$g(\hat{y}(x)) = \sum_j f_j(x_{\cdot,j}) + \sum_{k\neq j} f_{k,j}(x_{\cdot,k}, x_{\cdot,j})$$

### 4.3 FastSparse — Sparse GAMs

FastSparse produces interpretable GAMs with **few** stumps by minimising:

$$R^0_{\text{train}}(\boldsymbol{\lambda}) = R_{\text{train}}(\boldsymbol{\lambda}) + C_0\|\boldsymbol{\lambda}\|_0$$

where $\|\boldsymbol{\lambda}\|_0$ counts non-zero coefficients.

**Update rule — Case 1** ($\lambda_{t,j}=0$, consider adding stump $j$):

Include stump $j$ only if doing so reduces $R^0_{\text{train}}$ by at least $C_0$, i.e.:

$$d_- \notin \left[\frac{1}{2} - \frac{\sqrt{C_0(2R_{\text{train}}-C_0)}}{2R_{\text{train}}},\; \frac{1}{2}\right]$$

If $d_- \leq$ the lower bound, set $\lambda_{t+1,j} = \frac{1}{2}\ln\frac{1-d_-}{d_-}$; otherwise keep $\lambda_{t+1,j}=0$.

**Update rule — Case 2** ($\lambda_{t,j}\neq 0$, consider removing stump $j$):

Temporarily remove stump $j$, compute the modified error rate $d^{\backslash j}_-$, and apply the same interval check using $R^{\backslash j}_{\text{train}}$. If the check fails, set $\lambda_{t+1,j}=0$.

**Key intuition:** An error rate close to 1/2 means stump $j$ is no better than random guessing on the weighted data — not worth the complexity cost $C_0$. An error rate close to 0 means the stump is very useful.

FastSparse cycles through features in order of accuracy, alternating between coordinate descent steps and "swap steps" (replacing one stump with a better one).

### 4.4 When to Use GAMs

**Good for:** tabular data with meaningful continuous features; binary classification or regression; datasets where individual feature effects are important.

**Not suited for:** data where strong pairwise or higher-order interactions dominate; multiclass problems (decision trees are better); datasets with many binary features (step functions become uninformative).

---

## 5. References

1. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5–32. [Original random forests paper.]
2. **Fisher, A., Rudin, C., & Dominici, F.** (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research*, 20(177), 1–81. [MR, CMR, MCR framework.]
3. **Lundberg, S. M. & Lee, S.-I.** (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*. [SHAP values and TreeSHAP.]
4. **Freund, Y. & Schapire, R. E.** (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119–139. [Original AdaBoost paper with training error bound.]
5. **Hastie, T. & Tibshirani, R.** (1990). *Generalized Additive Models*. Chapman & Hall. [Foundational GAM reference.]
6. **Lou, Y., Caruana, R., Gehrke, J., & Ridgeway, G.** (2013). Accurate intelligible models with pairwise interactions. *KDD 2013*. [GA²M with interaction terms.]
7. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Ch. 10 (Boosting and additive trees), Ch. 9 (GAMs).

# 02 — Statistical Learning Theory

> **Keywords:** true risk, empirical risk, Hoeffding's inequality, union bound, Ockham's Razor bound, growth function, VC dimension, VC bound, bias-variance, approximation-estimation decomposition

---

## 1. Setup and Notation

| Symbol | Meaning |
|--------|---------|
| $D$ | True data distribution |
| $f:\mathcal{X}\to\{-1,+1\}$ | Classifier |
| $\mathcal{F}$ | Function class |
| $g(x,y)=\mathbf{1}[f(x)\neq y]$ | 0-1 loss for $f$ |
| $\mathcal{G}=\{g_f:f\in\mathcal{F}\}$ | Loss class |

**True (test) risk:**

$$R^{\text{true}}(f) = \mathbb{P}_{(X,Y)\sim D}(f(X)\neq Y) = \mathbb{E}_{(X,Y)\sim D}[\mathbf{1}_{f(X)\neq Y}]$$

**Empirical (training) risk:**

$$R^{\text{emp}}(f) = \frac{1}{n}\sum_{i=1}^n\mathbf{1}[f(x_i)\neq y_i]$$

**Regression function and Bayes classifier:**

$$\eta(x) = \mathbb{E}[Y|X=x], \qquad t(x) = \text{sign}(\eta(x))$$

**Bayes risk:** $R^* = R^{\text{true}}(t) = \inf_f R^{\text{true}}(f)$ — the irreducible minimum error.

**Approximation-Estimation Decomposition:**

For $f_n = \arg\min_{f\in\mathcal{F}}R^{\text{emp}}(f)$ and $f^*=\arg\min_{f\in\mathcal{F}}R^{\text{true}}(f)$:

$$R^{\text{true}}(f_n) - R^* = \underbrace{(R^{\text{true}}(f^*) - R^*)}_{\text{Approximation Error}} + \underbrace{(R^{\text{true}}(f_n) - R^{\text{true}}(f^*))}_{\text{Estimation Error}}$$

- **Approximation error:** how well the best function in $\mathcal{F}$ approximates the Bayes classifier. Decreases as $\mathcal{F}$ grows richer. Fixed once $\mathcal{F}$ is chosen.
- **Estimation error:** how far $f_n$ (chosen from data) deviates from $f^*$. Increases as $\mathcal{F}$ grows (more complex models overfit). Decreases as $n$ grows.

This is the **bias-variance trade-off** expressed in risk terms.

---

## 2. Hoeffding's Inequality

**Theorem (Hoeffding's Inequality).** Let $Z_1,\ldots,Z_n$ be i.i.d. random variables with $h(Z)\in[a,b]$. Then for all $\epsilon>0$:

$$\mathbb{P}\left(\left|\frac{1}{n}\sum_{i=1}^nh(Z_i)-\mathbb{E}[h(Z)]\right|>\epsilon\right) \leq 2\exp\left(-\frac{2n\epsilon^2}{(b-a)^2}\right)$$

**Inverted (confidence interval form):** Setting the right-hand side to $\delta$:

$$\epsilon = (b-a)\sqrt{\frac{\log(2/\delta)}{2n}}$$

So with probability at least $1-\delta$:

$$\left|R^{\text{emp}}(f) - R^{\text{true}}(f)\right| \leq \sqrt{\frac{\log(2/\delta)}{2n}}$$

(using $b-a=1$ for the 0-1 loss).

**One-sided version:** With probability at least $1-\delta$:

$$R^{\text{true}}(f) - R^{\text{emp}}(f) \leq \sqrt{\frac{\log(1/\delta)}{2n}}$$

**Key limitation:** This bound applies only to a **fixed** $f$ chosen independently of the data. It fails when $f$ is chosen based on the data (which is exactly what learning algorithms do).

---

## 3. Union Bound and the Ockham's Razor Bound

### 3.1 The Problem with Finite Function Classes

For a finite class $\mathcal{F}=\{f_1,\ldots,f_M\}$, we want a bound that holds simultaneously for all $f\in\mathcal{F}$.

Define "bad events" $C_j = \{\text{training set } \mathbf{Z} \text{ where } R^{\text{true}}(g_j)-R^{\text{emp}}(g_j)\geq\epsilon\}$.

**Union bound:** $\mathbb{P}[C_1\cup\cdots\cup C_M]\leq\sum_{j=1}^M\mathbb{P}[C_j]$.

By Hoeffding (one-sided) applied to each $g_j$:

$$\mathbb{P}\left[\exists f\in\mathcal{F}: R^{\text{true}}(f)-R^{\text{emp}}(f)\geq\epsilon\right] \leq \sum_{j=1}^M\exp(-2n\epsilon^2) = M\exp(-2n\epsilon^2)$$

Setting this to $\delta$: $\epsilon=\sqrt{\frac{\log M+\log(1/\delta)}{2n}}$.

### 3.2 Ockham's Razor Bound (Hoeffding + Union Bound)

**Theorem.** For any finite $\mathcal{F}=\{f_1,\ldots,f_M\}$, for all $\delta>0$, with probability at least $1-\delta$:

$$\left|R^{\text{emp}}(f) - R^{\text{true}}(f)\right| \leq \sqrt{\frac{\log M + \log(2/\delta)}{2n}} \quad\forall f\in\mathcal{F}$$

### 3.3 Bounding the True Risk of the Learner

Since $f_n=\arg\min_{f\in\mathcal{F}}R^{\text{emp}}(f)$, we know $R^{\text{emp}}(f^*)-R^{\text{emp}}(f_n)\geq 0$. Therefore:

$$R^{\text{true}}(f_n) = R^{\text{true}}(f_n) - R^{\text{true}}(f^*) + R^{\text{true}}(f^*)$$

$$\leq [R^{\text{emp}}(f^*)-R^{\text{emp}}(f_n)] + R^{\text{true}}(f_n)-R^{\text{true}}(f^*) + R^{\text{true}}(f^*)$$

Rearranging and using the triangle inequality:

$$R^{\text{true}}(f_n) \leq 2\sup_{f\in\mathcal{F}}\left|R^{\text{true}}(f)-R^{\text{emp}}(f)\right| + R^{\text{true}}(f^*)$$

Applying the Ockham's Razor bound, with probability at least $1-\delta$:

$$\boxed{R^{\text{true}}(f_n) \leq 2\sqrt{\frac{\log M+\log(2/\delta)}{2n}} + R^{\text{true}}(f^*)}$$

**Interpretations:**
- Generalisation requires restricting $f$ to a class $\mathcal{F}$ (more knowledge = better bound).
- For a fixed $f$: $R^{\text{true}}(f)-R^{\text{emp}}(f)\approx 1/\sqrt{n}$ for most datasets.
- For finite $\mathcal{F}$ with $|\mathcal{F}|=M$: the supremum grows as $\sqrt{\log M/n}$ — the extra $\log M$ term is the cost of model selection.

---

## 4. Growth Function and VC Dimension

The Ockham's Razor bound requires $|\mathcal{F}|<\infty$. For infinite function classes, we need a different complexity measure.

### 4.1 Growth Function

**Definition.** The **growth function** $S_{\mathcal{F}}(n)$ is the maximum number of distinct labellings that $\mathcal{F}$ can produce on any $n$ points:

$$S_{\mathcal{F}}(n) = \sup_{(z_1,\ldots,z_n)}|\mathcal{F}_{z_1,\ldots,z_n}|$$

where $\mathcal{F}_{z_1,\ldots,z_n} = \{(f(z_1),\ldots,f(z_n)):f\in\mathcal{F}\}$.

Clearly $S_{\mathcal{F}}(n)\leq 2^n$ (at most $2^n$ binary labellings of $n$ points).

**Theorem (Growth Function Bound; Vapnik-Chervonenkis).** For any $\delta>0$, with probability at least $1-\delta$:

$$R^{\text{true}}(f) \leq R^{\text{emp}}(f) + 2\sqrt{\frac{\log S_{\mathcal{F}}(2n)+\log(4/\delta)}{n}} \quad\forall f\in\mathcal{F}$$

### 4.2 VC Dimension

**Definition.** $\mathcal{F}$ **shatters** a set $\{z_1,\ldots,z_n\}$ if $S_{\mathcal{F}}(n)=2^n$ on those points (every labelling is achievable).

**VC dimension** $h=\text{VC}(\mathcal{F})$: the largest $n$ for which there exists a set of $n$ points that $\mathcal{F}$ shatters.

**Example VC dimensions:**

| Class $\mathcal{F}$ | VC dimension $h$ |
|---|---|
| Half-spaces in $\mathbb{R}^d$ (linear classifiers) | $d+1$ |
| Axis-aligned rectangles in $\mathbb{R}^2$ | 4 |
| Convex polygons in $\mathbb{R}^2$ | $\infty$ |
| Decision trees with $k$ leaves | $O(k\log k)$ |

### 4.3 Sauer-Shelah Lemma

**Lemma (Vapnik-Chervonenkis; Sauer-Shelah).** For $\mathcal{F}$ with finite VC dimension $h$ and $n\geq h$:

$$S_{\mathcal{F}}(n) \leq \sum_{i=0}^h\binom{n}{i} \leq \left(\frac{en}{h}\right)^h$$

The growth function is at most polynomial in $n$ (once $n$ exceeds the VC dimension). This is the key insight: even though $\mathcal{F}$ is infinite, it can only "see" $O(n^h)$ distinct labellings on $n$ points.

### 4.4 VC Bound

Combining the Growth Function Bound with the Sauer-Shelah Lemma:

**Theorem (VC Bound).** If $\mathcal{F}$ has VC dimension $h$, then for $n\geq h$, with probability at least $1-\delta$:

$$\boxed{R^{\text{true}}(f) \leq R^{\text{emp}}(f) + 2\sqrt{\frac{h(\ln(2n)+1)+\log(4/\delta)}{n}} \quad\forall f\in\mathcal{F}}$$

**Interpretation:**
- The gap between true and empirical risk is $O\left(\sqrt{h\log n/n}\right)$.
- For the gap to be small, we need $n\gg h$ (many times the VC dimension of the class).
- For $n=h$: the bound is vacuous — we can shatter the training data and learn nothing about the true risk.
- Richer classes (larger $h$) require more data to generalise.

---

## 5. Summary

```
For a FIXED function f:
  |R_true(f) - R_emp(f)| ≤ sqrt(log(2/δ) / 2n)   with prob ≥ 1−δ
  [Hoeffding; valid only for f not depending on data]

For a FINITE function class |F| = M:
  sup_f |R_true - R_emp| ≤ sqrt((log M + log(2/δ)) / 2n)   [Ockham's Razor]
  R_true(f_n) ≤ 2·sqrt(...) + R_true(f*)

For an INFINITE class with VC dimension h:
  S_F(n) ≤ (en/h)^h                                [Sauer-Shelah]
  R_true(f) ≤ R_emp(f) + 2·sqrt((h·ln(2n) + log(4/δ)) / n)   [VC bound]

Key insight: generalisation requires knowledge (restricting to F)
             and scales as sqrt(complexity / n)
```

---

## 6. References

1. **Vapnik, V. N. & Chervonenkis, A. Ya.** (1971). On the uniform convergence of relative frequencies of events to their probabilities. *Theory of Probability and its Applications*, 16(2), 264–280. [VC dimension and growth function.]
2. **Sauer, N.** (1972). On the density of families of sets. *Journal of Combinatorial Theory, Series A*, 13(1), 145–147. [Sauer-Shelah lemma.]
3. **Hoeffding, W.** (1963). Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association*, 58(301), 13–30. [Hoeffding's inequality.]
4. **Kearns, M. & Vazirani, U.** (1994). *An Introduction to Computational Learning Theory*. MIT Press. [Accessible treatment of VC theory and PAC learning.]
5. **Shalev-Shwartz, S. & Ben-David, S.** (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press. [Modern treatment; free PDF available.]
6. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Ch. 7 (Model assessment and selection).

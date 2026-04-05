# 01 — ML Fundamentals

> **Keywords:** supervised learning, unsupervised learning, loss function, risk, regularisation, ROC curve, AUC, cross-validation, data leakage

---

## 1. Taxonomy of Machine Learning Problems

| Task | Type | Representative Methods |
|------|------|----------------------|
| **Classification** | Supervised | Logistic regression, AdaBoost, random forest, SVM, k-NN, decision trees |
| **Conditional probability estimation** | Supervised | Same as classification |
| **Regression** | Supervised | Least squares, ridge, lasso, kernel ridge, boosted trees, GP regression |
| **Ranking** | Supervised | Conditional probability estimation |
| **Pattern finding** | Unsupervised | Apriori, FP-Growth |
| **Clustering** | Unsupervised | K-means, k-medians, hierarchical agglomerative, GMM |
| **Dimension reduction** | Unsupervised | PaCMAP, t-SNE, UMAP, PHATE, LargeVis |
| **Density estimation** | Unsupervised | Adaptive kernel density estimation |
| **Sampling / Generation** | — | GANs, Transformers, Diffusion models (DDPMs, LDMs) |

---

## 2. K-Nearest Neighbours (KNN)

KNN classifies a new point by finding its $k$ nearest training points (by some distance metric) and taking a majority vote.

**Key insight:** the choice of distance metric is critical — it determines which points are considered neighbours. Common choices include standard Euclidean distance and weighted Euclidean distance.

**Adding higher-dimensional features:**

| | Effect |
|--|--|
| **Pro** | Separation is easier in higher dimensions |
| **Con** | Overfitting and increased computational cost |

---

## 3. Loss Functions

A **loss function** $\ell(f(x), y)$ measures how wrong a prediction $f(x)$ is relative to the true label $y$.

**Common loss functions:**

| Name | Formula | Used in |
|------|---------|---------|
| Least squares | $(f(x)-y)^2$ | Regression |
| Misclassification | $\mathbf{1}[\text{sign}(f(x)) \neq y]$ | Classification |
| Logistic | $\log(1+e^{-yf(x)})$ | Logistic regression, neural networks |
| Hinge | $\max(0, 1-yf(x))$ | SVM |
| Exponential | $e^{-yf(x)}$ | AdaBoost |

### 3.1 True Risk vs. Empirical Risk

**True risk (test error):** the expected loss over the true data distribution $D$:

$$R^{\text{test}}(f) = \mathbb{E}_{(x,y)\sim D}\,\ell(f(x), y) = \int \ell(f(x), y)\,dD(x, y)$$

This is what we actually care about but cannot directly measure.

**Empirical risk (training error):** the average loss on the observed training set:

$$R^{\text{train}}(f) = \frac{1}{n}\sum_{i=1}^n \ell(f(x_i), y_i)$$

This is measurable but can be gamed by overfitting.

**Goal:** Make $R^{\text{train}}$ close to $R^{\text{test}}$. This requires (i) a large $n$, and (ii) a simple $f$.

---

## 4. Ockham's Razor and Regularisation

> **Ockham's Razor:** The best models are simple models that fit the data well.

### 4.1 Regularised Learning

Add a complexity penalty $R^{\text{reg}}(f)$ to the training objective:

$$\min_f \sum_i \ell(f(x_i), y_i) + C\cdot R^{\text{reg}}(f)$$

where $C > 0$ controls the trade-off between fit and simplicity. Larger $C$ = stronger regularisation = simpler model.

**Regularisation norms for linear $f(x) = \lambda^\top x$:**

| Norm | Formula | Effect |
|------|---------|--------|
| $\ell_2$ (ridge) | $\|\lambda\|_2^2 = \sum_j \lambda_j^2$ | Shrinks all coefficients |
| $\ell_1$ (Lasso) | $\|\lambda\|_1 = \sum_j |\lambda_j|$ | Sets some coefficients to zero (sparsity) |
| $\ell_0$ | $\|\lambda\|_0 = \sum_j \mathbf{1}[\lambda_j \neq 0]$ | Counts non-zero terms |

**Full objectives:**

$$\text{Ridge:} \quad \frac{1}{n}\sum_i(y_i - f(x_i))^2 + C\|\lambda\|_2^2$$

$$\text{SVM:} \quad \frac{1}{n}\sum_i \max(0, 1-y_if(x_i)) + C\|\lambda\|_2^2$$

$$\text{Reg. Logistic:} \quad \frac{1}{n}\sum_i \log(1+e^{-y_if(x_i)}) + C\|\lambda\|_1 \text{ or } C\|\lambda\|_2^2$$

$$\text{AdaBoost:} \quad \frac{1}{n}\sum_i e^{-y_if(x_i)} \quad \text{(no explicit regularisation)}$$

**Common misconception:** A classifier trained on *less* training data is *more* likely to overfit — not less. With less data, any fixed-complexity model has more latitude to fit noise.

---

## 5. Receiver Operating Characteristic (ROC) Curves

### 5.1 Confusion Matrix

For binary classification with $y \in \{+1, -1\}$ and predicted $\hat{y}$:

| | $\hat{y}=+1$ | $\hat{y}=-1$ |
|--|--|--|
| $y=+1$ | **TP** (True Positive) | **FN** (False Negative, Type II error) |
| $y=-1$ | **FP** (False Positive, Type I error) | **TN** (True Negative) |

### 5.2 Key Metrics

$$\text{Misclassification error} = \frac{FP+FN}{n} = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[y_i \neq \hat{y}_i]$$

$$\text{TPR (Sensitivity, Recall)} = \frac{TP}{N_+} = \frac{\sum_i \mathbf{1}[y_i=\hat{y}_i=1]}{\sum_i \mathbf{1}[y_i=1]}$$

$$\text{TNR (Specificity)} = \frac{TN}{N_-} = \frac{\sum_i \mathbf{1}[y_i=\hat{y}_i=-1]}{\sum_i \mathbf{1}[y_i=-1]}$$

$$\text{FPR} = \frac{FP}{N_-} = 1 - \text{TNR}$$

$$\text{Precision} = \frac{TP}{N_{\hat{+}}}$$

$$F_1 = 2\cdot\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}$$

### 5.3 The ROC Curve

The ROC curve plots **TPR vs. FPR** as the classification threshold varies. It answers: "For a given false positive rate, what true positive rate can we achieve?"

Each threshold gives one point on the curve. A random classifier lies on the diagonal; a perfect classifier passes through $(0, 1)$.

### 5.4 AUC (Area Under the ROC Curve)

AUC is a **rank statistic**: it equals the probability that a randomly chosen positive is ranked higher (by $f$) than a randomly chosen negative:

$$\text{AUC} = \frac{1}{N_+ \cdot N_-}\sum_{i\in\text{Pos}}\sum_{k\in\text{Neg}}\mathbf{1}[f(x_i) > f(x_k)]$$

AUC optimisation is therefore equivalent to **supervised bipartite ranking**.

### 5.5 Handling Imbalanced Data

When positives are rare:
1. **Do not** report plain accuracy — it can be misleadingly high.
2. Adjust the imbalance parameter $C$ to balance TP and FP trade-offs.
3. Examine the confusion matrix separately for FP and FN.
4. Weigh losses for positives and negatives differently:

$$\frac{1}{n}\!\left(C_{\text{imb}}\sum_{i:y_i=1}\ell(y_i,f(x_i)) + \sum_{k:y_k=-1}\ell(y_k,f(x_k))\right) + \text{Reg}(f)$$

---

## 6. Cross-Validation

Cross-validation estimates test error without wasting data on a separate validation set.

### 6.1 $k$-Fold Cross-Validation for Evaluation

1. Divide data into $k$ approximately equal **folds**.
2. Train on $k-1$ folds; evaluate on the held-out fold.
3. Repeat $k$ times, each fold serving as the test fold once.
4. Report **mean** and **std** of the evaluation metric across folds.

Typical choice: $k = 10$.

### 6.2 $k$-Fold Cross-Validation for Hyperparameter Tuning

1. Set aside a **test fold** (never touched during tuning).
2. From the remaining data, repeatedly hold out a **validation fold**.
3. For each candidate hyperparameter value, train on training folds, evaluate on validation fold.
4. Average the validation metric across validation folds for each candidate. Choose the best.
5. Retrain on the full training + validation data with the best hyperparameter.
6. Report final performance on the **test fold**.

### 6.3 Nested Cross-Validation

Use **two nested loops**:
- **Outer loop:** $k$-fold CV for unbiased performance *evaluation*.
- **Inner loop:** $k$-fold CV for hyperparameter *tuning*.

This prevents information from the test fold leaking into the tuning process.

For imbalanced datasets: ensure each fold contains a proportional representation of the minority class (stratified splitting).

### 6.4 Data Leakage — Critical Warnings

Data leakage occurs when information from the test set influences training, giving an over-optimistic estimate of performance.

**Rules to prevent leakage:**
1. Any feature selection or elimination decisions must be made on the **training set only**.
2. **Normalisation** (mean, std) must be computed on the training set. Apply those same parameters to the validation and test sets. Never normalise the test set using its own statistics.
3. Never normalise the full dataset before splitting into train/test.
4. Do not use the same normalisation code path for training and testing.

---

## 7. References

1. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Ch. 2 (Overview of supervised learning), Ch. 7 (Model assessment and selection).
2. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §1.5 (Decision theory), §1.3 (Model selection).
3. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Ch. 6 (Frequentist statistics).
4. **Fawcett, T.** (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. [Definitive tutorial on ROC curves and AUC.]
5. **Arlot, S. & Celisse, A.** (2010). A survey of cross-validation procedures for model selection. *Statistics Surveys*, 4, 40–79.

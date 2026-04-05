# 18 — Conditional Random Field

> **Keywords:** structured prediction, graphical model, MRF, label bias, forward-backward algorithm, Viterbi, feature functions, log-linear model

---

## 1. Motivation & Intuition

Many real-world problems require predicting **structured outputs** — sequences, trees, or graphs — rather than a single label. Examples include:

- **Part-of-speech tagging:** given a sentence $x = (x_1, \ldots, x_T)$, predict the tag sequence $y = (y_1, \ldots, y_T)$.
- **Named entity recognition (NER):** label each word as PERSON, LOCATION, OTHER, etc.
- **Gene sequence labelling, handwriting recognition, speech tagging.**

A naive approach applies an independent classifier at each position, ignoring label dependencies entirely. Better approaches (HMM, MEMM, CRF) model the joint distribution over the entire label sequence.

**The key insight of CRF** is to model $p(Y \mid X)$ directly — the conditional distribution of the label sequence given the observations — rather than modelling the joint $p(X, Y)$. This is the discriminative vs. generative distinction:

| Model | Models | Advantage |
|---|---|---|
| HMM | $p(X, Y) = p(Y)p(X \mid Y)$ | Simple, generative; strong independence assumptions |
| MEMM | $p(Y \mid X) = \prod_t p(y_t \mid y_{t-1}, x_{1:T})$ | Discriminative; but suffers from label bias |
| **CRF** | $p(Y \mid X) = \frac{1}{Z(X)}\prod_t \psi_t(y_{t-1}, y_t, X)$ | Discriminative; **global normalisation** avoids label bias |

CRFs allow arbitrary, overlapping **feature functions** over the entire input $x_{1:T}$ and any pair of adjacent labels. This flexibility — without the generative modelling cost — is the main reason CRFs became the standard for sequence labelling throughout the 2000s and remain relevant today as components of deep models.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1989 | Hidden Markov Models (HMMs) popularised for speech recognition and sequence labelling (Rabiner's tutorial). |
| 1996 | Maximum Entropy Markov Models (MEMMs) introduced as discriminative alternatives to HMMs. |
| 2001 | **Lafferty, McCallum & Pereira** introduce Conditional Random Fields, explicitly diagnosing and fixing the label bias problem of MEMMs. |
| 2003–2008 | CRFs become the dominant model for NLP sequence labelling (NER, POS, chunking). |
| 2015–present | Neural CRFs: the output layer of a BiLSTM (or Transformer) feeds into a CRF layer, combining learned representations with structured decoding. |

---

## 3. Background: Markov Random Fields

A **Markov Random Field (MRF)** — also called an undirected graphical model — defines a joint distribution via **clique potential functions** (also called factors):

$$p(x) = \frac{1}{Z} \prod_{c \in \mathcal{C}} \psi_c(x_c)$$

where $\mathcal{C}$ is the set of maximal cliques, $x_c$ denotes the variables in clique $c$, and $Z = \sum_x \prod_c \psi_c(x_c)$ is the partition function ensuring normalisation.

Each potential $\psi_c \geq 0$ can always be written in log-linear form by defining $\psi_c(x_c) = \exp[-E_c(x_c)]$, where $E_c$ is an "energy" (lower energy = higher probability). Equivalently, defining $F_c := -E_c$:

$$p(x) = \frac{1}{Z} \prod_{c} \exp[F_c(x_c)] = \frac{1}{Z} \exp\!\left[\sum_{c} F_c(x_c)\right]$$

This exponential (Gibbs) form is convenient because the log-probability is a **sum** of local energies, making gradient computations tractable.

---

## 4. MEMM and the Label Bias Problem

### 4.1 MEMM Definition

The **Maximum Entropy Markov Model (MEMM)** factorises the conditional as a product of locally normalised distributions:

$$P(Y \mid X, \lambda) = \prod_{t=1}^{T} P(y_t \mid y_{t-1},\, x_{1:T},\, \lambda)$$

Each factor $P(y_t \mid y_{t-1}, x_{1:T}, \lambda)$ is a local maximum-entropy (log-linear) classifier, normalised over the possible values of $y_t$ given the previous state $y_{t-1}$.

### 4.2 The Label Bias Problem

The label bias problem (first named by Lafferty et al., 2001) is a fundamental flaw of **per-state local normalisation**.

Consider a state $y_{t-1}$ with only **one possible successor** state. The local distribution $P(y_t \mid y_{t-1}, x_{1:T})$ must assign probability 1 to that successor regardless of the observation $x$. Effectively, the transition is **blind to the input** at that step.

States with few outgoing transitions concentrate probability mass on their successors irrespective of the evidence. This biases the model toward hypotheses that pass through low-branching states — a purely structural artifact with no connection to the data.

**Why CRF solves this:** CRF uses a **single global partition function** $Z(X)$ that normalises over the entire label sequence. No per-state local normalisation occurs, so every transition competes globally and must be consistent with the full observed sequence.

---

## 5. Chain-Structured CRF

### 5.1 Deriving CRF from MRF

A CRF models $p(Y \mid X)$ using an MRF structure over the label sequence $Y$, treating the observations $X$ as fixed conditioning variables (not modelled). For a **chain-structured** CRF, the graph is a linear chain $y_1 - y_2 - \cdots - y_T$, with cliques consisting of adjacent pairs $(y_{t-1}, y_t)$.

Applying the MRF factor decomposition to the label sequence, conditioned on $X$:

$$p(Y \mid X) = \frac{1}{Z(X)} \prod_{t=1}^{T} \psi_t(y_{t-1},\, y_t,\, X)$$

where $Z(X) = \sum_{y} \prod_{t=1}^T \psi_t(y_{t-1}, y_t, X)$ now depends on $X$ (unlike the MRF partition function, which is a constant). Each potential $\psi_t$ depends on the entire input $X = x_{1:T}$, giving the model access to non-local context.

Writing each potential in log-linear form:

$$\psi_t(y_{t-1}, y_t, X) = \exp[F_t(y_{t-1}, y_t, X)]$$

so that:

$$p(Y \mid X) = \frac{1}{Z(X)} \exp\!\left[\sum_{t=1}^{T} F_t(y_{t-1}, y_t, X)\right]$$

### 5.2 Feature Functions and Parameterisation

Each $F_t$ is decomposed into two types of **feature functions**:

**Transition features** $f_k(y_{t-1}, y_t, X)$: capture dependencies between adjacent labels and the input. Example: "the current word is capitalised AND the previous label is O AND the current label is B-PER."

**State (emission) features** $g_l(y_t, X)$: capture dependencies between a single label and the input. Example: "the current word ends in -ing AND the current label is VBG."

The full parameterisation (with weight vectors $\lambda \in \mathbb{R}^K$ for transition features and $\eta \in \mathbb{R}^L$ for state features):

$$F_t(y_{t-1}, y_t, X) = \sum_{k=1}^{K} \lambda_k f_k(y_{t-1}, y_t, X) + \sum_{l=1}^{L} \eta_l g_l(y_t, X)$$

Substituting into $p(Y \mid X)$:

$$p(Y \mid X) = \frac{1}{Z(X,\lambda,\eta)} \exp\!\left[\sum_{t=1}^{T}\!\left(\sum_{k=1}^{K} \lambda_k f_k(y_{t-1}, y_t, X) + \sum_{l=1}^{L} \eta_l g_l(y_t, X)\right)\right]$$

Since the double sum can be rearranged (weights are independent of $t$):

$$= \frac{1}{Z(X,\lambda,\eta)} \exp\!\left[\lambda^\top \underbrace{\sum_{t=1}^{T} f(y_{t-1}, y_t, X)}_{\text{total transition feature count}} + \eta^\top \underbrace{\sum_{t=1}^{T} g(y_t, X)}_{\text{total state feature count}}\right]$$

### 5.3 Compact Notation

Define the concatenated parameter vector and sufficient statistic:

$$\theta = \begin{pmatrix} \lambda \\ \eta \end{pmatrix} \in \mathbb{R}^{K+L}, \qquad H(y, X) = \begin{pmatrix} \displaystyle\sum_{t=1}^{T} f(y_{t-1}, y_t, X) \\[6pt] \displaystyle\sum_{t=1}^{T} g(y_t, X) \end{pmatrix} \in \mathbb{R}^{K+L}$$

Then the CRF distribution takes the elegant form of an **exponential family**:

$$\boxed{p(Y = y \mid X = x) = \frac{1}{Z(x,\theta)} \exp\!\left[\theta^\top H(y, x)\right]}$$

where $Z(x, \theta) = \sum_{y'} \exp[\theta^\top H(y', x)]$.

This is a log-linear model over the entire sequence. The sufficient statistic $H(y, x)$ is the vector of **accumulated feature counts** along the labelling $y$.

---

## 6. Learning (Parameter Estimation)

### 6.1 Maximum Likelihood Objective

Given $N$ i.i.d. training examples $\{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$, we maximise the conditional log-likelihood:

$$\hat{\theta} = \arg\max_{\theta} \sum_{i=1}^{N} \log p(y^{(i)} \mid x^{(i)})$$

Expanding with the CRF formula:

$$\sum_{i=1}^{N} \log p(y^{(i)} \mid x^{(i)}) = \sum_{i=1}^{N} \left[\theta^\top H(y^{(i)}, x^{(i)}) - \log Z(x^{(i)}, \theta)\right]$$

$$= \sum_{i=1}^{N} \left[\lambda^\top \sum_{t=1}^{T} f(y_{t-1}^{(i)}, y_t^{(i)}, x^{(i)}) + \eta^\top \sum_{t=1}^{T} g(y_t^{(i)}, x^{(i)}) - \log Z(x^{(i)}, \lambda, \eta)\right]$$

This objective is **concave** in $\theta$ (since $\log Z$ is convex in $\theta$), so gradient ascent or L-BFGS finds the global maximum.

### 6.2 Computing the Gradient

The gradient w.r.t. $\lambda$ is:

$$\nabla_\lambda \mathcal{L} = \sum_{i=1}^{N} \left[\sum_{t=1}^{T} f(y_{t-1}^{(i)}, y_t^{(i)}, x^{(i)}) - \nabla_\lambda \log Z(x^{(i)}, \lambda, \eta)\right]$$

**Computing $\nabla_\lambda \log Z$:** We differentiate $\log Z(x, \lambda, \eta) = \log \sum_y \exp[\theta^\top H(y, x)]$:

$$\nabla_\lambda \log Z = \frac{\nabla_\lambda Z}{Z} = \frac{\sum_y \exp[\theta^\top H(y,x)] \cdot \sum_t f(y_{t-1},y_t,x)}{\sum_y \exp[\theta^\top H(y,x)]}$$

$$= \sum_y p(y \mid x) \sum_{t=1}^{T} f(y_{t-1}, y_t, x) = \mathbb{E}_{p(y \mid x)}\!\left[\sum_{t=1}^{T} f(y_{t-1}, y_t, x)\right]$$

**Exchanging sum and expectation** (both are finite):

$$= \sum_{t=1}^{T} \sum_y p(y \mid x)\, f(y_{t-1}, y_t, x)$$

For each $t$, we can marginalise out all labels except $y_{t-1}$ and $y_t$:

$$\sum_y p(y \mid x)\, f(y_{t-1}, y_t, x) = \sum_{y_{t-1}} \sum_{y_t} \underbrace{\left(\sum_{y \setminus \{y_{t-1}, y_t\}} p(y \mid x)\right)}_{= p(y_{t-1}, y_t \mid x)} f(y_{t-1}, y_t, x)$$

$$= \sum_{y_{t-1}} \sum_{y_t} p(y_{t-1}, y_t \mid x)\, f(y_{t-1}, y_t, x)$$

Therefore the full gradient is:

$$\boxed{\nabla_\lambda \mathcal{L} = \sum_{i=1}^{N} \left[\sum_{t=1}^{T} f(y_{t-1}^{(i)}, y_t^{(i)}, x^{(i)}) - \sum_{t=1}^{T} \sum_{j, k} p(y_{t-1}^{(i)} = j,\, y_t^{(i)} = k \mid x^{(i)})\, f(j, k, x^{(i)})\right]}$$

**Interpretation:** The gradient is **observed feature counts minus expected feature counts** (under the current model). At convergence, these two are equal — the model's expected sufficient statistics match the empirical ones. This is the standard result for exponential family MLE.

The gradient w.r.t. $\eta$ has an identical structure with $g$ in place of $f$ and $p(y_t \mid x)$ in place of $p(y_{t-1}, y_t \mid x)$.

**Key observation:** Computing the gradient requires the marginals $p(y_{t-1}, y_t \mid x)$ and $p(y_t \mid x)$ for every $t$ and every training example. These are computed efficiently via the **forward-backward algorithm** (Section 7.2).

---

## 7. Inference

Given a trained CRF, there are three standard inference tasks:

1. **Marginal inference:** compute $p(y_t = i \mid x)$ for each position $t$ and label $i \in \mathcal{S}$.
2. **Pairwise marginal inference:** compute $p(y_{t-1} = j, y_t = i \mid x)$ — needed for learning.
3. **MAP inference (decoding):** find $\hat{y} = \arg\max_{y} p(y \mid x)$ — the most probable label sequence.

### 7.1 Factored Form of $p(Y \mid X)$

Write the joint as:

$$p(y \mid x) = \frac{1}{Z} \prod_{t=1}^{T} \psi_t(y_{t-1}, y_t, x)$$

where $\psi_t(y_{t-1}, y_t, x) = \exp[\lambda^\top f(y_{t-1}, y_t, x) + \eta^\top g(y_t, x)]$.

### 7.2 Forward-Backward Algorithm (Marginal Inference)

To compute $p(y_t = i \mid x)$, we marginalise over all other labels:

$$p(y_t = i \mid x) = \sum_{y : y_t = i} p(y \mid x) = \frac{1}{Z} \sum_{y : y_t = i} \prod_{t'=1}^{T} \psi_{t'}(y_{t'-1}, y_{t'}, x)$$

The key observation is that this sum **decomposes** into a product of a "left part" and a "right part":

$$p(y_t = i \mid x) = \frac{1}{Z}\; \underbrace{\left(\sum_{y_{1:t-1}} \prod_{t'=1}^{t} \psi_{t'}(y_{t'-1}, y_{t'}, x)\bigg|_{y_t=i}\right)}_{\alpha_t(i)} \cdot \underbrace{\left(\sum_{y_{t+1:T}} \prod_{t'=t+1}^{T} \psi_{t'}(y_{t'-1}, y_{t'}, x)\bigg|_{y_t=i}\right)}_{\beta_t(i)}$$

**Forward variable** $\alpha_t(i)$: the (unnormalised) probability of all sequences $y_{1:t}$ ending in $y_t = i$:

$$\alpha_1(i) = \psi_1(y_0, i,\, x) \qquad \text{(}y_0 \text{ is a special START token)}$$

$$\alpha_t(i) = \sum_{j \in \mathcal{S}} \psi_t(j,\, i,\, x)\; \alpha_{t-1}(j), \quad t = 2, \ldots, T$$

**Backward variable** $\beta_t(i)$: the (unnormalised) probability of all sequences $y_{t+1:T}$ given $y_t = i$:

$$\beta_T(i) = 1 \qquad \forall\, i \in \mathcal{S}$$

$$\beta_t(i) = \sum_{j \in \mathcal{S}} \psi_{t+1}(i,\, j,\, x)\; \beta_{t+1}(j), \quad t = T-1, \ldots, 1$$

**Partition function** (from forward variables at the last step):

$$Z = \sum_{i \in \mathcal{S}} \alpha_T(i)$$

**Marginals:**

$$p(y_t = i \mid x) = \frac{\alpha_t(i)\,\beta_t(i)}{Z}$$

$$p(y_{t-1} = j,\; y_t = i \mid x) = \frac{\alpha_{t-1}(j)\;\psi_t(j, i, x)\;\beta_t(i)}{Z}$$

**Time complexity:** $O(T \cdot |\mathcal{S}|^2)$ — linear in sequence length, quadratic in the number of labels.

### 7.3 MAP Inference — Viterbi Algorithm

MAP decoding finds:

$$\hat{y} = \arg\max_{y_1, \ldots, y_T} p(y \mid x) = \arg\max_{y_1, \ldots, y_T} \sum_{t=1}^{T} \log \psi_t(y_{t-1}, y_t, x)$$

This is solved by **Viterbi dynamic programming**. Define:

$$v_t(i) = \max_{y_{1:t-1}} \sum_{t'=1}^{t} \log \psi_{t'}(y_{t'-1}, y_{t'}, x)\bigg|_{y_t = i}$$

the score of the best partial path ending in state $i$ at time $t$. Recursion:

$$v_1(i) = \log \psi_1(y_0, i, x)$$

$$v_t(i) = \max_{j \in \mathcal{S}} \left[\log \psi_t(j, i, x) + v_{t-1}(j)\right], \quad t = 2, \ldots, T$$

Track back-pointers $\text{bp}_t(i) = \arg\max_j [\log \psi_t(j,i,x) + v_{t-1}(j)]$ to recover the optimal sequence. The final score is $\max_i v_T(i)$ and the sequence is recovered by tracing back-pointers from $\hat{y}_T = \arg\max_i v_T(i)$.

**Time complexity:** $O(T \cdot |\mathcal{S}|^2)$ — same as forward-backward.

---

## 8. Summary

```
Input: x = (x_1, ..., x_T)
Output: y = (y_1, ..., y_T)

CRF:  p(y|x) = (1/Z(x)) * exp[ θᵀ H(y,x) ]

                    ┌─────────────────────────┐
                    │  Global normalisation   │
                    │  Z(x) = Σ_y exp[...]   │  ← avoids label bias
                    └─────────────────────────┘

Learning:  gradient = observed counts − expected counts
           expected counts via forward-backward  (O(T|S|²))

Decoding:  Viterbi  (O(T|S|²))
```

The CRF's power comes from three things working together:
1. **Global normalisation** eliminates label bias.
2. **Arbitrary feature functions** over the full input and label pairs enable rich, overlapping features.
3. **Efficient dynamic programming** (forward-backward, Viterbi) makes both learning and inference tractable despite the exponential number of label sequences.

---

## 9. References

1. **Lafferty, J., McCallum, A., & Pereira, F.** (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data. *Proceedings of ICML 2001*, 282–289. [Original CRF paper; introduces label bias diagnosis.]
2. **Sutton, C. & McCallum, A.** (2012). An introduction to conditional random fields. *Foundations and Trends in Machine Learning*, 4(4), 267–373. [Comprehensive tutorial; free PDF available.]
3. **Lafferty, J.** (1996). Bernstein polynomials and the label bias problem. [Background on label bias in MEMMs.]
4. **Rabiner, L. R.** (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286. [Forward-backward and Viterbi algorithms.]
5. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §8.3 (Markov Random Fields), §8.4 (Inference in graphical models).
6. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. §19 (Undirected graphical models), §19.6 (CRFs).
7. **Huang, Z., Xu, W., & Yu, K.** (2015). Bidirectional LSTM-CRF models for sequence labeling. *arXiv:1508.01991*. [Neural CRF; state-of-the-art NER with CRF output layer.]

# Boltzmann Machine and Restricted Boltzmann Machine (RBM)

> **Keywords:** energy-based model, Gibbs distribution, restricted Boltzmann machine, contrastive divergence, deep Boltzmann machine, partition function, positive phase, negative phase

---

## 1. Motivation & Intuition

Most probabilistic models we have seen so far are **directed** (Bayesian networks): probability flows from parents to children, and sampling is easy. But many natural constraints on data — mutual exclusivity, soft symmetry, pairwise compatibility — are more naturally expressed as **undirected** dependencies without a causal ordering.

**Energy-based models** assign to each configuration $x$ an energy $E(x)$: low energy = high probability. They capture complex, globally-consistent distributions without requiring a normalised conditional at every node. The Boltzmann Machine is the canonical deep energy-based model.

The **Restricted Boltzmann Machine (RBM)** imposes a bipartite graph structure between visible units $v$ (observed) and hidden units $h$ (latent): no connections within a layer. This restriction makes inference tractable — hidden units are conditionally independent given visibles and vice versa — while retaining the expressiveness of an energy-based model.

RBMs were the building blocks of the first successful deep generative models (Deep Belief Nets, Deep Boltzmann Machines) and helped launch the deep learning renaissance around 2006.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1982 | **Hopfield** introduces the Hopfield network — an energy-based associative memory. |
| 1985 | **Hinton & Sejnowski** introduce the Boltzmann Machine: an MRF with hidden units trained by a stochastic gradient rule. |
| 1986 | **Smolensky** introduces the Harmonium — equivalent to what later becomes the RBM. |
| 2002 | **Hinton** introduces **Contrastive Divergence (CD)** — a practical, efficient approximation to the log-likelihood gradient, making RBM training feasible. |
| 2006 | **Hinton, Osindero & Teh** use stacked RBMs to pretrain Deep Belief Nets, achieving state-of-the-art on MNIST and reigniting interest in deep learning. |
| 2009 | **Salakhutdinov & Hinton** introduce **Deep Boltzmann Machines (DBM)**: a fully undirected deep model with approximate variational inference. |
| 2012–present | RBMs largely superseded by VAEs and GANs for generative modelling, but remain influential in recommender systems (Netflix Prize) and as conceptual building blocks. |

---

## 3. Boltzmann Machine: General Framework

### 3.1 MRF with Hidden Nodes

A Boltzmann Machine is a **Markov Random Field (MRF)** in which variables are partitioned into **visible units** $v$ (observed data) and **hidden units** $h$ (latent factors). The joint distribution follows the MRF factor decomposition over cliques $\{C_i\}$:

$$p(x) = \frac{1}{Z}\prod_{i=1}^K \psi_i(x_{C_i}), \qquad Z = \sum_x \prod_{i=1}^K \psi_i(x_{C_i})$$

Each potential is written in **log-linear (energy) form**:

$$\psi_i(x_{C_i}) = \exp\{-E_i(x_{C_i})\}$$

So:

$$p(x) = \frac{1}{Z}\exp\!\left\{-\sum_{i=1}^K E_i(x_{C_i})\right\} = \frac{1}{Z}\exp\{-E(x)\}$$

This is the **Boltzmann (Gibbs) distribution**: $p(x) \propto e^{-E(x)}$. Low energy configurations have high probability.

The full state is $x = (h^\top, v^\top)^\top \in \mathbb{R}^{m+n}$.

**Challenge:** The partition function $Z = \sum_x e^{-E(x)}$ requires summing over all $2^{m+n}$ configurations (for binary units) — exponentially intractable for large models.

---

## 4. Restricted Boltzmann Machine (RBM)

### 4.1 Bipartite Graph Structure

An RBM restricts the Boltzmann Machine to a **bipartite graph**: connections exist only between visible units $v\in\{0,1\}^n$ and hidden units $h\in\{0,1\}^m$. No visible-visible or hidden-hidden connections.

$$x = \begin{pmatrix}h \\ v\end{pmatrix}, \quad h = \begin{pmatrix}h_1\\\vdots\\h_m\end{pmatrix}, \quad v = \begin{pmatrix}v_1\\\vdots\\v_n\end{pmatrix}$$

### 4.2 Energy Function and Joint Distribution

The RBM energy function is:

$$E(v, h) = -\left(h^\top W v + \alpha^\top v + \beta^\top h\right) = -\left(\sum_{i=1}^m\sum_{j=1}^n h_i w_{ij} v_j + \sum_{j=1}^n \alpha_j v_j + \sum_{i=1}^m \beta_i h_i\right)$$

where:
- $W \in \mathbb{R}^{m\times n}$: weight matrix (visible-hidden connections)
- $\alpha \in \mathbb{R}^n$: visible unit biases
- $\beta \in \mathbb{R}^m$: hidden unit biases

The joint distribution:

$$p(v, h) = \frac{1}{Z}\exp\{-E(v,h)\} = \frac{1}{Z}\exp\!\left\{h^\top W v + \alpha^\top v + \beta^\top h\right\}$$

$$= \frac{1}{Z}\prod_{i=1}^m\prod_{j=1}^n \exp(h_i w_{ij} v_j)\cdot\prod_{j=1}^n\exp(\alpha_j v_j)\cdot\prod_{i=1}^m\exp(\beta_i h_i)$$

### 4.3 Inference: Conditional Distributions

#### $p(h|v)$ — Posterior over Hidden Units

Because the RBM has no hidden-hidden connections, all hidden units are **conditionally independent given the visible units**. Formally, by the local Markov property of MRFs:

$$p(h_l \mid h_{-l}, v) = p(h_l \mid v)$$

Therefore:

$$\boxed{p(h|v) = \prod_{l=1}^m p(h_l|v)}$$

**Deriving $p(h_l = 1 \mid v)$:** Decompose the energy into terms involving $h_l$ and terms that do not:

$$E(h, v) = -\Big[h_l\underbrace{\left(\sum_{j=1}^n w_{lj}v_j + \beta_l\right)}_{=:\,H_l(v)} + \underbrace{\sum_{i\neq l}\sum_{j=1}^n h_i w_{ij}v_j + \sum_{j=1}^n\alpha_jv_j + \sum_{i\neq l}\beta_ih_i}_{=:\,\bar{H}_l(h_{-l},v)}\Big]$$

The conditional probability is:

$$p(h_l=1|v) = \frac{p(h_l=1, h_{-l}, v)}{p(h_l=0, h_{-l}, v) + p(h_l=1, h_{-l}, v)}$$

$$= \frac{\exp\{H_l(v) + \bar{H}_l\}}{\exp\{\bar{H}_l\} + \exp\{H_l(v) + \bar{H}_l\}} = \frac{\exp\{H_l(v)\}}{1 + \exp\{H_l(v)\}}$$

$$\boxed{p(h_l=1|v) = \sigma\!\left(\sum_{j=1}^n w_{lj}v_j + \beta_l\right)}$$

where $\sigma(x) = 1/(1+e^{-x})$ is the sigmoid function. Each hidden unit activates independently as a sigmoid function of its weighted input from the visible layer.

By symmetry, the conditional $p(v_j=1|h)$ has the identical form:

$$p(v_j=1|h) = \sigma\!\left(\sum_{i=1}^m w_{ij}h_i + \alpha_j\right)$$

### 4.4 Marginal $p(v)$ via Hidden Unit Summation

Marginalising over hidden units:

$$p(v) = \sum_h p(v,h) = \frac{1}{Z}\sum_h \exp\!\left\{h^\top Wv + \alpha^\top v + \beta^\top h\right\}$$

Factor out terms not involving $h$:

$$= \frac{1}{Z}\exp(\alpha^\top v)\sum_{h_1,\ldots,h_m}\exp\!\left\{\sum_{i=1}^m h_i\left(W_i^\top v + \beta_i\right)\right\}$$

where $W_i^\top v = \sum_j w_{ij}v_j$ is the $i$-th row of $W$ dotted with $v$. Since the exponent decomposes as a sum over $i$, the sum over all $h\in\{0,1\}^m$ factorises into independent sums over each $h_i$:

$$= \frac{1}{Z}\exp(\alpha^\top v)\prod_{i=1}^m\left[\sum_{h_i\in\{0,1\}}\exp\!\left\{h_i(W_i^\top v + \beta_i)\right\}\right]$$

Each inner sum over binary $h_i$:

$$\sum_{h_i\in\{0,1\}}\exp\!\left\{h_i(W_i^\top v + \beta_i)\right\} = \exp(0) + \exp(W_i^\top v + \beta_i) = 1 + \exp(W_i^\top v + \beta_i)$$

Therefore:

$$\boxed{p(v) = \frac{1}{Z}\exp\!\left(\alpha^\top v + \sum_{i=1}^m \log\!\left(1 + \exp(W_i^\top v + \beta_i)\right)\right) = \frac{1}{Z}\exp\!\left(\alpha^\top v + \sum_{i=1}^m \text{Softplus}(W_i^\top v + \beta_i)\right)}$$

where $\text{Softplus}(x) = \log(1+e^x)$. The hidden units contribute a sum of Softplus terms — a smooth, log-convex function of $v$.

---

## 5. Learning: The Partition Function Problem

### 5.1 Maximum Likelihood Objective

Given data $\{v^{(i)}\}_{i=1}^N$, maximum likelihood estimation:

$$\hat{\theta} = \arg\max_\theta \frac{1}{N}\sum_{i=1}^N\log p(v^{(i)};\theta) = \arg\max_\theta\;\mathbb{E}_{P_{\text{data}}}[\log P_{\text{model}}(v;\theta)]$$

This is equivalent to minimising the KL divergence from data to model:

$$= \arg\min_\theta\; D_{\text{KL}}(P_{\text{data}}\,\|\,P_{\text{model}})$$

since $\int P_{\text{data}}\log P_{\text{data}}$ is constant w.r.t. $\theta$.

### 5.2 Gradient of the Log-Likelihood

For the general energy-based model $p(x;\theta) = \hat{p}(x;\theta)/Z(\theta)$:

$$l(\theta) = \frac{1}{N}\sum_{i=1}^N\log\hat{p}(x_i;\theta) - \log Z(\theta)$$

**Gradient of $\log Z(\theta)$:** Using $\nabla_\theta\log Z = \frac{\nabla_\theta Z}{Z}$ and swapping differentiation and integration (justified under regularity conditions):

$$\nabla_\theta\log Z(\theta) = \frac{1}{Z(\theta)}\int\nabla_\theta\hat{p}(x;\theta)\,dx = \int\frac{\hat{p}(x;\theta)}{Z(\theta)}\cdot\frac{\nabla_\theta\hat{p}(x;\theta)}{\hat{p}(x;\theta)}\,dx$$

$$= \int p(x;\theta)\,\nabla_\theta\log\hat{p}(x;\theta)\,dx = \mathbb{E}_{p(x;\theta)}\!\left[\nabla_\theta\log\hat{p}(x;\theta)\right]$$

Therefore the full log-likelihood gradient decomposes into **two phases**:

$$\boxed{\nabla_\theta l(\theta) = \underbrace{\mathbb{E}_{P_{\text{data}}}\!\left[\nabla_\theta\log\hat{p}(x;\theta)\right]}_{\text{positive phase}} - \underbrace{\mathbb{E}_{P_{\text{model}}}\!\left[\nabla_\theta\log\hat{p}(x;\theta)\right]}_{\text{negative phase}}}$$

**Positive phase:** Average gradient of the unnormalised log-probability over the **training data** — pushes up the probability of observed data. Tractable (sum over training set).

**Negative phase:** Average gradient over samples from the **current model** — pushes down the probability of model hallucinations. Intractable in general (requires sampling from $p(x;\theta)$, which has an unknown $Z$).

### 5.3 RBM-Specific Gradient

For the RBM with $\theta = \{W, \alpha, \beta\}$, the energy $E(h,v) = -(h^\top Wv + \alpha^\top v + \beta^\top h)$:

$$\frac{\partial E(h,v)}{\partial w_{ij}} = -h_iv_j$$

The log-likelihood gradient w.r.t. $w_{ij}$:

$$\frac{\partial}{\partial w_{ij}}\log p(v) = \sum_h p(h|v)\,h_iv_j - \sum_{h,v}p(h,v)\,h_iv_j$$

Marginalising out $h$ in each term:

- **Positive phase** (first term): $\sum_{h_i}p(h_i|v)\,h_i\,v_j = p(h_i=1|v)\cdot v_j$. Closed-form via sigmoid — **tractable**.
- **Negative phase** (second term): $\sum_v p(v)\,p(h_i=1|v)\,v_j = \mathbb{E}_{p(v)}[p(h_i=1|v)\,v_j]$. Requires sampling from $p(v)$ — **intractable**.

$$\boxed{\frac{\partial}{\partial w_{ij}}\log p(v) = p(h_i=1|v)\cdot v_j - \mathbb{E}_{p(v)}\!\left[p(h_i=1|v)\cdot v_j\right]}$$

---

## 6. Contrastive Divergence (CD-$k$)

### 6.1 The Problem with MCMC

The exact negative phase requires running Gibbs sampling on $p(v,h;\theta)$ until convergence — which can take thousands of steps. This makes each gradient step extremely slow.

### 6.2 CD Objective

Hinton (2002) proposes to approximate the negative phase by running Gibbs sampling for only $k$ steps (typically $k=1$), starting from the training data. The **CD objective** is:

$$\hat{\theta}_{CD} = \arg\min_\theta\;D_{\text{KL}}(P^{(0)}\,\|\,P^{(\infty)}) - D_{\text{KL}}(P^{(k)}\,\|\,P^{(\infty)})$$

where $P^{(0)}$ is the data distribution, $P^{(k)}$ is the distribution after $k$ Gibbs steps from the data, and $P^{(\infty)} = P_{\text{model}}$.

Minimising this encourages $P^{(k)}$ to be close to $P^{(\infty)}$ — i.e., the chain mixes quickly from data to the model. CD-1 (one Gibbs step) works well in practice despite this crude approximation.

### 6.3 CD-$k$ Algorithm for RBM

```
For each minibatch {v^(1), ..., v^(N)}:

  Initialise:  v^(0) ← v  (from training data)

  For l = 0, 1, ..., k-1:
    For i = 1, ..., m:  h_i^(l) ~ p(h_i | v^(l))   = Bernoulli(σ(W_i·v^(l) + β_i))
    For j = 1, ..., n:  v_j^(l+1) ~ p(v_j | h^(l)) = Bernoulli(σ(W_j^T·h^(l) + α_j))

  CD-k gradient approximation:
    ∂log p(v)/∂w_{ij} ≈ p(h_i=1 | v^(0))·v_j^(0)  −  p(h_i=1 | v^(k))·v_j^(k)
                         [positive phase]                [negative phase]

  Update:  w_{ij} ← w_{ij} + η · (1/N)·Σ ∂log p(v)/∂w_{ij}
```

**Intuition:**
- The **positive phase** $p(h_i=1|v^{(0)})v_j^{(0)}$ is the correlation between hidden unit $i$ and visible unit $j$ under the data: it reflects what the model "wants" to explain.
- The **negative phase** $p(h_i=1|v^{(k)})v_j^{(k)}$ is the correlation under model fantasies after $k$ steps: it reflects what the model "currently believes".
- The update increases weights where data and model agree that units should co-activate, and decreases weights where the model's fantasies differ from the data.

---

## 7. Deep Boltzmann Machine (DBM)

A **Deep Boltzmann Machine** stacks multiple RBM layers into a single undirected model:

$$p(v, h^{(1)}, h^{(2)}) = \frac{1}{Z}\exp\!\left\{v^\top W^{(1)}h^{(1)} + h^{(1)\top}W^{(2)}h^{(2)}\right\}$$

Exact inference is intractable; variational inference with a factored approximation $q(h|v) = \prod_l q(h_l^{(1)}|v)\prod_k q(h_k^{(2)}|v)$ gives a tractable ELBO.

### 7.1 ELBO Derivation

For an arbitrary variational distribution $q(h|v)$:

$$\log p(v) = \log\frac{p(v,h)}{p(h|v)}$$

Multiply and divide by $q(h|v)$:

$$= \log\frac{p(v,h)}{q(h|v)} + \log\frac{q(h|v)}{p(h|v)}$$

Integrate both sides against $q(h|v)$ (note $\log p(v)$ is constant w.r.t. $h$):

$$\log p(v) = \underbrace{\mathbb{E}_{q(h|v)}\!\left[\log\frac{p(v,h)}{q(h|v)}\right]}_{\text{ELBO}(q,\theta)} + \underbrace{D_{\text{KL}}(q(h|v)\,\|\,p(h|v))}_{\geq\,0}$$

$$= \mathbb{E}_{q(h|v)}\!\left[\log p(v,h)\right] + H[q] + D_{\text{KL}}(q(h|v)\,\|\,p(h|v))$$

Since $D_{\text{KL}} \geq 0$: $\text{ELBO}(q,\theta) \leq \log p(v)$. Training maximises the ELBO over both $q$ (E-step) and $\theta$ (M-step).

---

## 8. Summary

```
Boltzmann Machine:
  p(x) = (1/Z) exp{-E(x)}     [Gibbs distribution]

RBM (bipartite structure, no intra-layer connections):
  E(v,h) = -(h^T W v + α^T v + β^T h)

Key inference results:
  p(h_l=1|v) = σ(W_l^T v + β_l)     [sigmoid; tractable]
  p(v_j=1|h) = σ(W_j^T h + α_j)     [sigmoid; tractable]
  p(v) = (1/Z) exp(α^T v + Σ_i Softplus(W_i^T v + β_i))

Learning gradient = positive phase − negative phase
  Positive (tractable):   E_data[∇ log p̂]
  Negative (intractable): E_model[∇ log p̂]

CD-k: approximate negative phase with k Gibbs steps from data
  k=1 works well in practice
```

---

## 9. References

1. **Hinton, G. E. & Sejnowski, T. J.** (1986). Learning and relearning in Boltzmann machines. In Rumelhart, D. E. & McClelland, J. L. (eds.), *Parallel Distributed Processing*, Vol. 1, pp. 282–317. MIT Press. [Original Boltzmann Machine.]
2. **Hinton, G. E.** (2002). Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 1771–1800. [Contrastive divergence algorithm; makes RBM training practical.]
3. **Hinton, G. E., Osindero, S., & Teh, Y. W.** (2006). A fast learning algorithm for deep belief nets. *Neural Computation*, 18(7), 1527–1554. [Greedy layer-wise pretraining with RBMs; sparked the deep learning revival.]
4. **Salakhutdinov, R. & Hinton, G. E.** (2009). Deep Boltzmann machines. *Proceedings of AISTATS 2009*, 448–455. [DBM with mean-field variational inference.]
5. **Fischer, A. & Igel, C.** (2012). An introduction to restricted Boltzmann machines. In *Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications (CIARP)*, Lecture Notes in Computer Science, Vol. 7441, pp. 14–36. Springer. [Accessible tutorial covering energy functions, inference, and CD learning.]
6. **Bengio, Y.** (2009). Learning deep architectures for AI. *Foundations and Trends in Machine Learning*, 2(1), 1–127. [Broader context for RBMs in deep learning.]
7. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §8.3 (Markov random fields), §8.3.3 (Restricted Boltzmann machines).
8. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. Ch. 18 (Confronting the partition function), Ch. 20.2 (Restricted Boltzmann machines).

# 20 — Restricted Boltzmann Machine

> **Keywords:** energy-based model, Boltzmann distribution, restricted Boltzmann machine, visible units, hidden units, contrastive divergence, persistent CD, free energy, deep belief network

---

## 1. Energy-Based Models

### 1.1 General Form

An **energy-based model** defines a probability distribution via an energy function $E(x; \theta)$:

$$p(x; \theta) = \frac{1}{Z(\theta)}\exp(-E(x; \theta)), \qquad Z(\theta) = \sum_x \exp(-E(x; \theta))$$

where $Z(\theta)$ is the **partition function** (normalising constant). Lower energy states are more probable.

The appeal is flexibility: any non-negative function on $\mathcal{X}$ can be represented this way. The challenge is that $Z(\theta)$ is a sum/integral over all configurations of $x$, which is intractable for most models of interest.

### 1.2 Boltzmann Machine

A **Boltzmann Machine** (Hinton & Sejnowski, 1983) is an energy-based model over binary units $s = (s_1, \ldots, s_p) \in \{0,1\}^p$:

$$E(s; \theta) = -\sum_i b_i s_i - \sum_{i < j} w_{ij} s_i s_j$$

where $b_i$ are **biases** (node potentials) and $w_{ij}$ are **weights** (pairwise interactions).

Learning requires computing $\mathbb{E}_p[s_i s_j]$ — the correlation under the model distribution — which involves summing over $2^p$ configurations. Exact learning is exponentially expensive.

---

## 2. Restricted Boltzmann Machine

### 2.1 Architecture

A **Restricted Boltzmann Machine (RBM)** restricts the Boltzmann machine to a **bipartite graph**: a layer of **visible** units $v \in \{0,1\}^D$ and a layer of **hidden** units $h \in \{0,1\}^K$, with connections **only between layers** (no within-layer connections).

The energy function is:

$$E(v, h; \theta) = -b^\top v - c^\top h - v^\top W h = -\sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} W_{ij} v_i h_j$$

where:
- $b \in \mathbb{R}^D$: visible biases
- $c \in \mathbb{R}^K$: hidden biases
- $W \in \mathbb{R}^{D \times K}$: weight matrix

Parameters: $\theta = (W, b, c)$.

### 2.2 Joint Distribution

$$p(v, h; \theta) = \frac{1}{Z}\exp(-E(v, h; \theta))$$

$$Z = \sum_{v \in \{0,1\}^D}\sum_{h \in \{0,1\}^K}\exp(-E(v, h; \theta))$$

### 2.3 Conditional Independence

The **bipartite** structure is the key restriction. Given $v$, all hidden units are **conditionally independent**:

$$p(h \mid v) = \prod_{j=1}^K p(h_j \mid v)$$

**Derivation.** Factoring from the joint:

$$p(h \mid v) = \frac{p(v,h)}{\sum_{h'} p(v,h')} \propto \exp\left(\sum_j c_j h_j + \sum_{i,j} W_{ij}v_i h_j\right) = \prod_j \exp\left(h_j\bigl(c_j + \sum_i W_{ij}v_i\bigr)\right)$$

This factorises over $j$, confirming conditional independence. Each factor gives a Bernoulli:

$$p(h_j = 1 \mid v) = \sigma\left(c_j + \sum_i W_{ij} v_i\right) = \sigma(c_j + W_{\cdot j}^\top v)$$

where $\sigma(x) = 1/(1+e^{-x})$ is the sigmoid function.

By the same argument, given $h$, all visible units are conditionally independent:

$$p(v_i = 1 \mid h) = \sigma\left(b_i + \sum_j W_{ij} h_j\right) = \sigma(b_i + W_{i\cdot} h)$$

These conditional independence properties make **block Gibbs sampling** tractable: alternate between sampling all of $h$ given $v$, and all of $v$ given $h$.

---

## 3. Marginal Distribution and Free Energy

### 3.1 Marginal Over Hidden Units

Marginalising out $h$ (summing over $2^K$ binary configurations of $h$):

$$p(v; \theta) = \frac{1}{Z}\sum_h \exp(-E(v,h)) = \frac{1}{Z}\exp(-b^\top v)\prod_{j=1}^K \sum_{h_j \in \{0,1\}} \exp\left(h_j(c_j + W_{\cdot j}^\top v)\right)$$

Each factor in the product is $1 + \exp(c_j + W_{\cdot j}^\top v)$. Therefore:

$$p(v; \theta) = \frac{1}{Z}\exp(-b^\top v)\prod_{j=1}^K \bigl[1 + \exp(c_j + W_{\cdot j}^\top v)\bigr]$$

### 3.2 Free Energy

Define the **free energy** $F(v)$ via $p(v) = e^{-F(v)}/Z$:

$$F(v) = -b^\top v - \sum_{j=1}^K \log\bigl[1 + \exp(c_j + W_{\cdot j}^\top v)\bigr]$$

This is a useful quantity because we can compute it exactly (without sampling), and the gradient of $\log p(v)$ can be expressed in terms of gradients of $F$.

---

## 4. Maximum Likelihood Learning

### 4.1 Log-Likelihood Gradient

Given a dataset $\mathcal{D} = \{v^{(n)}\}_{n=1}^N$, maximise:

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_n \log p(v^{(n)}; \theta) = \frac{1}{N}\sum_n \Bigl[-F(v^{(n)}) - \log Z\Bigr]$$

The gradient with respect to $W_{ij}$:

$$\frac{\partial \log p(v)}{\partial W_{ij}} = \underbrace{\mathbb{E}_{p(h|v)}[v_i h_j]}_{\text{data-driven}} - \underbrace{\mathbb{E}_{p(v,h)}[v_i h_j]}_{\text{model expectation}}$$

The first term (positive phase) is the correlation under the data posterior: $v_i \cdot p(h_j=1\mid v) = v_i\,\sigma(c_j + W_{\cdot j}^\top v)$ — this is **computable exactly**.

The second term (negative phase) is the correlation under the model joint distribution — this requires summing over all $v$, which is **intractable**. Similarly for $b$ and $c$:

$$\frac{\partial \log p(v)}{\partial b_i} = v_i - \mathbb{E}_{p(v)}[v_i], \qquad \frac{\partial \log p(v)}{\partial c_j} = p(h_j=1 \mid v) - \mathbb{E}_{p(v)}[p(h_j=1\mid v)]$$

### 4.2 Contrastive Divergence (CD-$k$)

**Contrastive Divergence** (Hinton, 2002) approximates the intractable negative phase using a short Gibbs chain started from the data:

1. **Positive phase:** Start with a data sample $v^{(0)} = v^{\text{data}}$. Compute $p(h \mid v^{(0)})$.
2. **Negative phase (approximation):** Run $k$ steps of block Gibbs:
   - Sample $h^{(0)} \sim p(h \mid v^{(0)})$
   - Sample $v^{(1)} \sim p(v \mid h^{(0)})$
   - $\ldots$ repeat $k$ times to get $v^{(k)}$
3. **Update:**

$$\Delta W_{ij} = \eta\,\Bigl[\underbrace{v_i^{(0)} p(h_j=1 \mid v^{(0)})}_{\text{positive phase}} - \underbrace{v_i^{(k)} p(h_j=1 \mid v^{(k)})}_{\text{negative phase (approx.)}}\Bigr]$$

**CD-1** ($k=1$) is the most common. It is a biased estimator of the gradient but works well in practice and is computationally cheap.

**Persistent Contrastive Divergence (PCD):** Instead of restarting from data, maintain a set of persistent "fantasy particles" $\{v^{(\text{pcd})}\}$ that are updated across mini-batches (Tieleman, 2008). This gives a better approximation of the model distribution but at higher cost.

---

## 5. Relation to Other Models

### 5.1 RBM as a Latent Variable Model

The RBM marginal $p(v)$ can be seen as a mixture:

$$p(v) = \sum_{h \in \{0,1\}^K} p(h)\, p(v \mid h)$$

It is a mixture of $2^K$ factorial distributions over $v$. With $K$ hidden units, an RBM can represent exponentially many mixture components.

### 5.2 Deep Belief Network (DBN)

A **Deep Belief Network** stacks RBMs greedily. Train an RBM on the data; treat the hidden activations as a new "visible" layer and train another RBM on top. Repeat.

**Training procedure (Hinton et al., 2006):**
1. Train RBM$_1$ on data $v$; extract features $h^{(1)} \sim p(h^{(1)}\mid v)$.
2. Train RBM$_2$ on $h^{(1)}$; extract $h^{(2)}$; continue stacking.
3. Fine-tune the entire network discriminatively (e.g., backpropagation).

This greedy layer-wise pretraining was historically important for training deep networks before modern initialisation and normalisation techniques (batch norm, ReLU) made direct end-to-end training practical.

### 5.3 Connection to VAE

Both the RBM and the VAE are latent variable models with the generative form $p(v) = \sum_h p(h)p(v\mid h)$. The key differences:
- RBM: discrete $h$, learned by CD (not gradient descent through the latent).
- VAE: continuous $h$, learned by the reparameterisation trick.

---

## 6. Summary

```
RBM: bipartite energy-based model
  E(v,h) = −bᵀv − cᵀh − vᵀWh
  p(v,h) = exp(−E(v,h)) / Z

Conditional independence (from bipartite structure):
  p(h_j=1 | v) = σ(c_j + W_{·j}ᵀ v)
  p(v_i=1 | h) = σ(b_i + W_{i·} h)

Log-likelihood gradient:
  ∂ log p(v)/∂W_{ij} = E_{p(h|v)}[v_i h_j] − E_{p(v,h)}[v_i h_j]
                        ↑ computable            ↑ intractable

CD-k approximation:
  Replace model expectation by k-step Gibbs chain started from data
  CD-1: positive phase − negative phase (one Gibbs step)
```

---

## 7. References

1. **Hinton, G. E. & Sejnowski, T. J.** (1983). Optimal perceptual inference. *Proceedings of CVPR*, 448–453. [Original Boltzmann Machine.]
2. **Smolensky, P.** (1986). Information processing in dynamical systems: Foundations of harmony theory. In Rumelhart, D. E. & McClelland, J. L. (eds.), *Parallel Distributed Processing*, Vol. 1. MIT Press. [Original RBM.]
3. **Hinton, G. E.** (2002). Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 1771–1800. [Contrastive Divergence.]
4. **Hinton, G. E., Osindero, S., & Teh, Y.-W.** (2006). A fast learning algorithm for deep belief nets. *Neural Computation*, 18(7), 1527–1554. [Deep Belief Networks.]
5. **Tieleman, T.** (2008). Training restricted Boltzmann machines using approximations to the likelihood gradient. *Proceedings of ICML 2008*, 1064–1071. [Persistent CD.]
6. **Fischer, A. & Igel, C.** (2012). An introduction to restricted Boltzmann machines. In *Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications*, Lecture Notes in Computer Science, Vol. 7441, 14–36. Springer. [Comprehensive review.]
7. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. Ch. 20 (Deep generative models, RBMs).

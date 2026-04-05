# 19 — Generative Models

> **Keywords:** generative model, explicit density, implicit density, latent variable, reparameterisation trick, stochastic backpropagation, VAE, GAN, normalising flow, diffusion model

---

## 1. Motivation & Intuition

A **generative model** learns the underlying probability distribution $p(x)$ of data. Once learned, it can:

- **Generate** new, realistic samples $x \sim p_\theta(x)$.
- **Evaluate** the likelihood $p_\theta(x)$ of a given observation.
- **Infer** latent structure — discovering interpretable low-dimensional representations of high-dimensional data.
- **Complete** missing data, perform anomaly detection, or enable data augmentation.

Generative models sit at the heart of modern machine learning: language models (autoregressive), image synthesis (diffusion, GAN), representation learning (VAE), and density estimation (flows) are all instances.

**Generative vs. Discriminative.** A discriminative model learns $p(y|x)$ — the label given the input. A generative model learns $p(x)$ or $p(x, y)$ — the joint data distribution. Generative models are harder to train but more expressive: they can answer any probabilistic query about $x$, not just classification.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1950s–80s | Classical statistical generative models: Gaussian mixtures, HMMs, factor analysis, Kalman filters. |
| 1986 | **Hinton & Sejnowski** introduce the Boltzmann Machine — the first deep (undirected) generative model. |
| 1995 | **Helmholtz Machine** (Dayan et al.) introduces the wake-sleep algorithm; precursor to VAEs. |
| 2001 | **Blei, Ng & Jordan** introduce Latent Dirichlet Allocation (LDA), a mixed-membership model for text. |
| 2013–14 | **VAE** (Kingma & Welling; Rezende, Mohamed & Wierstra): variational inference + reparameterisation trick enable scalable deep generative models. |
| 2014 | **GAN** (Goodfellow et al.): adversarial training as an alternative to likelihood maximisation. |
| 2015 | **Normalising flows** (Rezende & Mohamed; Dinh et al.): exact likelihood with invertible neural networks. |
| 2016 | **PixelCNN / WaveNet** (van den Oord et al.): autoregressive models achieve state-of-the-art image and audio generation. |
| 2020–22 | **Diffusion models** (Ho et al.; Song et al.) surpass GANs on image quality; flow matching (Lipman et al.) offers a cleaner training framework. |

---

## 3. Taxonomy of Generative Models

### 3.1 Model Families

| Family | Representative Models | Key Idea |
|--------|----------------------|----------|
| **Naive Bayes** | Naive Bayes classifier | $p(x) = \prod_j p(x_j)$; feature independence |
| **Mixture models** | GMM, topic models | Discrete latent cluster variable $z$ |
| **Time-series models** | HMM, Kalman Filter, Particle Filter | Sequential latent state $z_t$ |
| **Non-parametric Bayes** | Gaussian Process, Dirichlet Process | Infinite-dimensional priors |
| **Mixed-membership** | LDA, pLSA | Each observation belongs to multiple latent topics |
| **Factorial models** | FA, P-PCA, ICA, Sparse Coding | Continuous latent factor $z$; linear decoder |
| **Energy-based** | Boltzmann Machine, RBM | $p(x) \propto e^{-E(x)}$; unnormalised density |
| **VAE** | VAE, $\beta$-VAE, IWAE | Amortised variational inference; learned encoder/decoder |
| **GAN** | GAN, DCGAN, StyleGAN, WGAN | Adversarial game; implicit density |
| **Autoregressive** | PixelCNN, GPT, WaveNet | $p(x) = \prod_t p(x_t \mid x_{<t})$; exact likelihood |
| **Flow-based** | RealNVP, Glow, FFJORD | Invertible map; exact likelihood via change-of-variables |
| **Diffusion / Score** | DDPM, DDIM, Flow Matching | Iterative denoising; score matching |

### 3.2 Model Evolution

Several families are best understood as progressions from simpler classical methods:

```
PCA  ──▶  Probabilistic PCA (P-PCA)  ──▶  Factor Analysis (FA)
                                               ↕
K-means  ──────────────────────────▶  Gaussian Mixture Model (GMM)

Autoencoder  ──────────────────────▶  Variational Autoencoder (VAE)

LSA  ──▶  pLSA  ──▶  Latent Dirichlet Allocation (LDA)
```

These progressions share a common theme: start with a hard-assignment or deterministic method, then introduce a probabilistic (Bayesian) treatment to gain uncertainty quantification and principled inference.

---

## 4. Key Dimensions for Classifying Generative Models

### 4.1 Explicit vs. Implicit Density

| Type | Definition | Examples | Advantage |
|------|-----------|----------|-----------|
| **Explicit density** | $p_\theta(x)$ can be evaluated directly | VAE (approximate), flows (exact), autoregressive (exact) | Can compute likelihood; useful for anomaly detection, compression |
| **Implicit density** | $p_\theta(x)$ is not directly accessible; only sampling is possible | GAN | Greater flexibility; no need to normalise |

### 4.2 Directed vs. Undirected Graphical Model

| Type | Definition | Examples |
|------|-----------|----------|
| **Directed (Bayesian network)** | $p(x) = \prod_i p(x_i \mid \text{parents}(x_i))$; clear ancestral sampling | HMM, VAE, diffusion |
| **Undirected (MRF)** | $p(x) = \frac{1}{Z}\prod_c \psi_c(x_c)$; no ordering, partition function $Z$ intractable | RBM, Boltzmann Machine, MRF/CRF |

### 4.3 Latent Variable vs. Fully-Observed

| Type | Definition | Examples |
|------|-----------|----------|
| **Latent variable** | Part of the generative story is unobserved ($z$) | GMM, HMM, VAE, LDA |
| **Fully-observed** | All variables in the model are observed | Autoregressive models, flows |

Latent variable models are more compact and interpretable but require integrating out $z$, which is often intractable.

### 4.4 Tractable vs. Intractable Inference

| Type | Definition | Examples |
|------|-----------|----------|
| **Tractable** | Posterior $p(z\mid x)$ or likelihood $p(x)$ has closed form | Linear-Gaussian models (Kalman filter), P-PCA |
| **Intractable** | Must approximate via variational inference (ELBO) or MCMC | Deep VAEs, LDA, Boltzmann machines |

### 4.5 Shallow vs. Deep

| Type | Definition | Examples |
|------|-----------|----------|
| **Shallow** | Single hidden layer or single stochastic layer | GMM, RBM, FA |
| **Deep** | Multiple stochastic layers or deep neural network decoders | Deep Boltzmann Machine, VAE with deep encoder/decoder |

### 4.6 Parametric vs. Non-Parametric

| Type | Definition | Examples |
|------|-----------|----------|
| **Parametric** | Fixed number of parameters | GMM ($K$ components), neural network |
| **Non-parametric** | Number of parameters grows with data | Gaussian Process, Dirichlet Process mixture |

### 4.7 Likelihood-Based vs. Likelihood-Free

| Type | Training Signal | Examples |
|------|----------------|----------|
| **Likelihood-based** | Maximise $\log p_\theta(x)$ or ELBO | VAE, flows, autoregressive, diffusion |
| **Likelihood-free** | Adversarial signal, score matching, moment matching | GAN, MMD-based models |

---

## 5. Stochastic Backpropagation and the Reparameterisation Trick

### 5.1 The Problem: Gradients Through Random Nodes

Suppose we want to minimise a loss $J_\theta = \mathbb{E}_{Y \sim p_\theta(Y)}[L(Y)]$ by gradient descent on $\theta$. Naively computing $\nabla_\theta \mathbb{E}_{Y\sim p_\theta}[L(Y)]$ requires differentiating through the sampling operation, which is not directly possible.

A log-derivative trick exists (REINFORCE) but has high variance. The **reparameterisation trick** gives a lower-variance, computationally efficient alternative when the distribution $p_\theta$ is reparametrisable.

### 5.2 Reparameterisation of a Gaussian

For $Y \sim \mathcal{N}(\mu, \sigma^2)$ with parameters $\theta = (\mu, \sigma)$, introduce an auxiliary noise variable:

$$Z \sim \mathcal{N}(0, 1), \qquad Y = \mu + \sigma Z$$

The random variable $Y$ is now a **deterministic function** of $\theta$ and the noise $Z$:

$$Y = f(\mu, \sigma, Z)$$

Sampling $Y \sim p_\theta$ is equivalent to: (1) draw $Z\sim\mathcal{N}(0,1)$; (2) compute $Y = \mu + \sigma Z$ deterministically. The randomness is isolated in $Z$, which does **not depend on $\theta$**.

Monte Carlo estimate of $\mathbb{E}_{Y}[L(Y)]$ with $N$ samples:

$$J_\theta(Y) \approx \frac{1}{N}\sum_{i=1}^N L(Y^{(i)}), \qquad Z^{(i)} \sim \mathcal{N}(0,1), \quad Y^{(i)} = \mu + \sigma Z^{(i)}$$

### 5.3 Gradient via the Chain Rule

Since $Y = f(\mu, \sigma, Z)$ is differentiable in $(\mu, \sigma)$ with $Z$ fixed:

$$\nabla_\theta J_\theta = \nabla_Y J_\theta \cdot \nabla_\theta Y = \nabla_Y J_\theta \cdot \frac{\partial(\mu + \sigma Z)}{\partial\theta}$$

Separating the two parameters:

$$\frac{\partial Y}{\partial \mu} = 1, \qquad \frac{\partial Y}{\partial \sigma} = Z$$

$$\boxed{\nabla_\theta J_\theta = \nabla_Y J_\theta \cdot \underbrace{\frac{\partial Y}{\partial \mu}}_{1} \cdot \nabla_\theta \mu + \nabla_Y J_\theta \cdot \underbrace{\frac{\partial Y}{\partial \sigma}}_{Z} \cdot \nabla_\theta \sigma}$$

This is a standard backpropagation chain rule — no special treatment of stochastic nodes needed. Modern autodiff frameworks (PyTorch, JAX) implement this automatically.

### 5.4 Conditional Reparameterisation

For a conditional distribution $p(Y|X) = \mathcal{N}(\mu_\theta(X),\, \sigma_\theta^2(X))$ — where the mean and variance are outputs of a neural network with parameters $\theta$ — the reparameterisation is:

$$Z \sim \mathcal{N}(0, 1), \qquad Y = \mu_\theta(X) + \sigma_\theta(X)\cdot Z$$

The gradient w.r.t. $\theta$ is:

$$\nabla_\theta J_\theta = \nabla_Y J_\theta \cdot \frac{\partial Y}{\partial \mu_\theta}\cdot\nabla_\theta \mu_\theta(X) + \nabla_Y J_\theta \cdot \frac{\partial Y}{\partial \sigma_\theta}\cdot\nabla_\theta \sigma_\theta(X)$$

with $\frac{\partial Y}{\partial \mu_\theta} = 1$ and $\frac{\partial Y}{\partial \sigma_\theta} = Z$.

This is the **core mechanism of the VAE encoder**: $\mu_\theta(x)$ and $\sigma_\theta(x)$ are the encoder outputs, $Z\sim\mathcal{N}(0,I)$ is sampled once per training step, and $z = \mu_\theta(x) + \sigma_\theta(x)\odot Z$ is the latent code passed to the decoder. The gradient flows through $\mu_\theta$ and $\sigma_\theta$ via standard backpropagation.

### 5.5 Generalisation: Other Reparametrisable Distributions

The trick extends to any distribution from which we can draw samples via a deterministic transformation of a fixed base noise:

| Distribution | Reparameterisation |
|---|---|
| $\mathcal{N}(\mu, \Sigma)$ | $Y = \mu + L Z$, $Z\sim\mathcal{N}(0,I)$, $L = \text{chol}(\Sigma)$ |
| $\text{Exp}(\lambda)$ | $Y = -\frac{\log U}{\lambda}$, $U\sim\text{Uniform}(0,1)$ |
| $\text{Concrete/Gumbel-Softmax}$ | Continuous relaxation of discrete distributions via Gumbel noise |

For distributions that are not reparametrisable (e.g., discrete), alternatives include the REINFORCE estimator or the straight-through estimator.

---

## 6. Comparative Overview

| Model | Density | Training | Inference | Strengths | Weaknesses |
|-------|---------|----------|-----------|-----------|------------|
| **GMM** | Explicit | EM | Exact | Simple, interpretable | Limited expressiveness |
| **VAE** | Approximate | ELBO + reparam. | Amortised | Stable training, smooth latent space | Blurry samples |
| **GAN** | Implicit | Adversarial | None needed | Sharp samples | Mode collapse, unstable training |
| **Flow** | Exact | MLE | Exact (invertible) | Exact likelihood, efficient sampling | Architecture constrained to be invertible |
| **Autoregressive** | Exact | MLE | Sequential ($O(d)$ steps) | State-of-the-art density estimation | Slow sampling |
| **Diffusion** | Approximate | Score matching / ELBO | Iterative ($T$ steps) | High-quality samples, stable training | Slow sampling (mitigated by DDIM/distillation) |

---

## 7. References

1. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. Ch. 20 (Deep Generative Models). [Free online: deeplearningbook.org]
2. **Kingma, D. P. & Welling, M.** (2014). Auto-encoding variational Bayes. *ICLR 2014*. [VAE and reparameterisation trick.]
3. **Rezende, D. J., Mohamed, S., & Wierstra, D.** (2014). Stochastic backpropagation and approximate inference in deep generative models. *ICML 2014*. [Stochastic backpropagation / reparameterisation, introduced independently from Kingma & Welling.]
4. **Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y.** (2014). Generative adversarial nets. *NeurIPS 2014*.
5. **Blei, D. M., Ng, A. Y., & Jordan, M. I.** (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022.
6. **Tipping, M. E. & Bishop, C. M.** (1999). Probabilistic principal component analysis. *Journal of the Royal Statistical Society: Series B*, 61(3), 611–622. [P-PCA; connects PCA to FA via the EM algorithm.]
7. **Dinh, L., Sohl-Dickstein, J., & Bengio, S.** (2017). Density estimation using Real-valued Non-Volume Preserving (Real NVP) transformations. *ICLR 2017*. [Flow-based generative models.]
8. **van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K.** (2016). Pixel recurrent neural networks. *ICML 2016*. [PixelRNN/PixelCNN; autoregressive image generation.]
9. **Ho, J., Jain, A., & Abbeel, P.** (2020). Denoising diffusion probabilistic models. *NeurIPS 2020*. [DDPM.]
10. **Bond-Taylor, S., Leach, A., Long, Y., & Willcocks, C. G.** (2022). Deep generative modelling: A comparative review of VAEs, GANs, normalizing flows, energy-based and autoregressive models. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(11), 7327–7347. [Comprehensive survey covering all major generative model families.]
11. **Hinton, G. E. & Sejnowski, T. J.** (1986). Learning and relearning in Boltzmann machines. In Rumelhart, D. E. & McClelland, J. L. (eds.), *Parallel Distributed Processing*, Vol. 1, pp. 282–317. MIT Press.

# Generative Adversarial Networks (GAN)

> **Keywords:** implicit density model, minimax game, discriminator, generator, Jensen-Shannon divergence, optimal discriminator, mode collapse, Wasserstein GAN

---

## 1. Motivation & Intuition

A **generative model** learns a distribution $P_g$ that approximates the data distribution $P_{\text{data}}$. Classical approaches — like VAEs or energy-based models — define an explicit density and maximise the likelihood. This requires a tractable normalisation constant and often imposes restrictive distributional assumptions.

GANs take a fundamentally different approach: they are **implicit density models**. The generator $G$ never defines $P_g$ explicitly; it simply learns a mapping from noise $z$ to data space $x = G(z)$. The induced distribution $P_g$ is implicitly defined by the pushforward of $P_Z$ through $G$. Sampling from $P_g$ is trivial (just draw $z \sim P_Z$ and apply $G$), even though evaluating $P_g(x)$ is not.

The training signal comes from a **discriminator** $D$ that tries to tell real data apart from generated data. The generator is trained to fool the discriminator. This adversarial dynamic — a two-player minimax game — drives $P_g$ toward $P_{\text{data}}$.

**Analogy.** Think of the generator as a counterfeiter printing fake banknotes and the discriminator as a forensic detective. The counterfeiter gets better by studying the detective's decisions; the detective gets better by seeing more counterfeits. At equilibrium, the counterfeits are indistinguishable from real notes — $P_g = P_{\text{data}}$.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 2014 | **Goodfellow et al.** introduce GANs, proposing the minimax objective and proving the global optimum is $P_g = P_{\text{data}}$. |
| 2015 | **DCGAN** (Radford et al.): convolutional architecture stabilises GAN training; produces coherent images. |
| 2017 | **WGAN** (Arjovsky et al.): replaces JS divergence with Wasserstein distance, solving the vanishing gradient problem and mode collapse theoretically. |
| 2018 | **BigGAN** (Brock et al.): class-conditional GANs at scale produce photorealistic images on ImageNet. |
| 2019 | **StyleGAN** (Karras et al.): style-based generator architecture enables fine-grained control over image synthesis. |
| 2020–present | Diffusion models emerge as strong competitors to GANs for image generation, though GANs remain preferred for speed (single forward pass). |

---

## 3. Setup

**Elements:**

| Symbol | Role |
|--------|------|
| $\{x_i\}_{i=1}^N \sim P_{\text{data}}$ | Real training data |
| $z \sim P_Z$ | Noise input (e.g., $\mathcal{N}(0,I)$) |
| $G(z;\theta_g)$ | **Generator**: maps noise to fake data; $x_{\text{fake}} = G(z)$ |
| $P_g$ | Distribution of $G(z)$ — the implicit generated distribution |
| $D(x;\theta_d) \in [0,1]$ | **Discriminator**: outputs probability that $x$ is from $P_{\text{data}}$ |

The goal is to learn $G$ such that $P_g \to P_{\text{data}}$, i.e., the generated samples become indistinguishable from real data.

---

## 4. Objectives

### 4.1 Discriminator Objective

The discriminator is trained to **maximise** its ability to classify real vs. fake:

$$\max_D\;V_D = \mathbb{E}_{x \sim P_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim P_Z}[\log(1 - D(G(z)))]$$

- **First term** $\mathbb{E}[\log D(x)]$: real data should be classified as real ($D(x) \to 1$, so $\log D(x) \to 0$).
- **Second term** $\mathbb{E}[\log(1-D(G(z)))]$: fake data should be classified as fake ($D(G(z)) \to 0$, so $\log(1-D(G(z))) \to 0$).

Both terms are maximised when $D$ correctly identifies real and fake samples.

### 4.2 Generator Objective

The generator is trained to **minimise** the discriminator's ability to detect fakes:

$$\min_G\;V_G = \mathbb{E}_{z \sim P_Z}[\log(1 - D(G(z)))]$$

When the generator successfully fools the discriminator, $D(G(z)) \to 1$, so $1 - D(G(z)) \to 0$ and $\log(1-D(G(z))) \to -\infty$ — a large negative number, which is what the generator wants to minimise (i.e., push as negative as possible by making $D(G(z))$ large).

### 4.3 Combined Minimax Game

Combining both objectives:

$$\min_G\max_D\;V(D,G) = \mathbb{E}_{x \sim P_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim P_Z}[\log(1 - D(G(z)))]$$

Equivalently, writing in terms of $P_g$ (since $z \sim P_Z$ and $x = G(z)$ means $G(z) \sim P_g$):

$$V(D,G) = \mathbb{E}_{x \sim P_{\text{data}}}[\log D(x)] + \mathbb{E}_{x \sim P_g}[\log(1 - D(x))]$$

The discriminator maximises $V$; the generator minimises $V$. This is a two-player zero-sum game.

### 4.4 Non-Saturating Generator Loss (Practical Fix)

In practice the original generator loss $\min_G \mathbb{E}[\log(1-D(G(z)))]$ suffers from **vanishing gradients** early in training: when $D$ easily rejects all fakes, $D(G(z)) \approx 0$, so $\log(1-D(G(z))) \approx 0$ and its gradient w.r.t. $\theta_g$ is near zero.

The standard fix is the **non-saturating loss**:
$$\max_G\;\mathbb{E}_{z \sim P_Z}[\log D(G(z))]$$

This has the same Nash equilibrium but provides stronger gradients early in training (when $D(G(z))$ is small, $\log D(G(z))$ has a large negative gradient, pushing the generator to improve quickly).

---

## 5. Optimal Discriminator

### 5.1 Derivation of $D^*_G$

For a **fixed** generator $G$ (and hence fixed $P_g$), find the discriminator $D$ maximising $V(D,G)$:

$$\max_D\;V(D,G) = \int \Big[P_{\text{data}}(x)\log D(x) + P_g(x)\log(1-D(x))\Big]\,dx$$

Since the integrand decomposes pointwise, we can maximise under the integral for each $x$ independently. For a fixed $x$, maximise $f(D) = a\log D + b\log(1-D)$ where $a = P_{\text{data}}(x) \geq 0$, $b = P_g(x) \geq 0$, $a+b > 0$:

$$\frac{df}{dD} = \frac{a}{D} - \frac{b}{1-D} = 0 \;\Rightarrow\; a(1-D) = bD \;\Rightarrow\; a = (a+b)D$$

$$\boxed{D^*_G(x) = \frac{P_{\text{data}}(x)}{P_{\text{data}}(x) + P_g(x)}}$$

**Verification:** $D^*_G \in [0,1]$ since $P_{\text{data}}, P_g \geq 0$. When $P_{\text{data}}(x) \gg P_g(x)$ (the point is much more likely to be real), $D^* \approx 1$. When $P_g(x) \gg P_{\text{data}}(x)$, $D^* \approx 0$. ✓

**Second-order check:** $\frac{d^2f}{dD^2} = -\frac{a}{D^2} - \frac{b}{(1-D)^2} < 0$, confirming this is a maximum. ✓

---

## 6. Global Optimum and Jensen-Shannon Divergence

### 6.1 Substituting $D^*_G$

Substituting the optimal discriminator into $V$:

$$\min_G V(D^*_G, G) = \min_G\;\mathbb{E}_{x\sim P_{\text{data}}}\!\left[\log\frac{P_{\text{data}}}{P_{\text{data}}+P_g}\right] + \mathbb{E}_{x\sim P_g}\!\left[\log\frac{P_g}{P_{\text{data}}+P_g}\right]$$

Factor out $\frac{1}{2}$ by writing $P_{\text{data}}+P_g = 2\cdot\frac{P_{\text{data}}+P_g}{2}$:

$$= \min_G\;\mathbb{E}_{x\sim P_{\text{data}}}\!\left[\log\frac{P_{\text{data}}}{\frac{P_{\text{data}}+P_g}{2}} - \log 2\right] + \mathbb{E}_{x\sim P_g}\!\left[\log\frac{P_g}{\frac{P_{\text{data}}+P_g}{2}} - \log 2\right]$$

$$= \min_G\;\underbrace{D_{\text{KL}}\!\left(P_{\text{data}}\;\Big\|\;\frac{P_{\text{data}}+P_g}{2}\right)}_{\geq\,0} + \underbrace{D_{\text{KL}}\!\left(P_g\;\Big\|\;\frac{P_{\text{data}}+P_g}{2}\right)}_{\geq\,0} - 2\log 2$$

### 6.2 Connection to Jensen-Shannon Divergence

The **Jensen-Shannon divergence** between two distributions $P$ and $Q$ is:
$$\text{JSD}(P \| Q) = \frac{1}{2}D_{\text{KL}}\!\left(P\,\Big\|\,\frac{P+Q}{2}\right) + \frac{1}{2}D_{\text{KL}}\!\left(Q\,\Big\|\,\frac{P+Q}{2}\right) \geq 0$$

with $\text{JSD}(P\|Q) = 0$ if and only if $P = Q$. Unlike $D_{\text{KL}}$, JSD is **symmetric** and always finite (bounded in $[0, \log 2]$). Therefore:

$$\min_G V(D^*_G, G) = 2\,\text{JSD}(P_{\text{data}} \| P_g) - 2\log 2 \geq -2\log 2$$

### 6.3 Global Optimum

The lower bound $-2\log 2$ is achieved when $\text{JSD}(P_{\text{data}}\|P_g) = 0$, i.e., when:

$$\boxed{P_g^* = P_{\text{data}}, \qquad D^*_{G^*}(x) = \frac{P_{\text{data}}}{P_{\text{data}}+P_{\text{data}}} = \frac{1}{2}}$$

At the global optimum, the generator perfectly replicates the data distribution. The discriminator can do no better than random guessing ($D = \frac{1}{2}$ everywhere), confirming the equilibrium is a Nash equilibrium: neither player can improve unilaterally.

---

## 7. Training Procedure

In practice, $G$ and $D$ are both neural networks trained by alternating stochastic gradient steps:

```
For each training iteration:
  1. Sample minibatch of real data:  {x_i} ~ P_data
  2. Sample minibatch of noise:      {z_i} ~ P_Z
  3. Update D (gradient ascent on V):
       θ_d ← θ_d + η · ∇_{θ_d} [E[log D(x)] + E[log(1 - D(G(z)))]]
     (repeat k times per generator step, typically k=1 or 5)
  4. Sample fresh noise:             {z_i} ~ P_Z
  5. Update G (gradient descent, non-saturating loss):
       θ_g ← θ_g - η · ∇_{θ_g} [-E[log D(G(z))]]
```

**Why alternate?** The minimax objective cannot be optimised jointly because it is non-convex in $\theta_g$ and the landscape changes as $D$ improves.

---

## 8. Failure Modes and Remedies

### 8.1 Mode Collapse

The generator may learn to map all noise vectors $z$ to a single (or few) mode(s) of $P_{\text{data}}$, producing little diversity. This happens because the generator finds a local strategy to fool $D$ without covering all of $P_{\text{data}}$.

**Remedies:** minibatch discrimination, unrolled GANs, mode-seeking divergences.

### 8.2 Training Instability

When $P_{\text{data}}$ and $P_g$ have disjoint supports (common early in training), the JS divergence is $\log 2$ everywhere (constant), giving zero gradients to $G$.

**Remedy:** **Wasserstein GAN (WGAN)** replaces JSD with the Wasserstein-1 distance:
$$W(P_{\text{data}}, P_g) = \sup_{\|f\|_L \leq 1}\;\mathbb{E}_{x\sim P_{\text{data}}}[f(x)] - \mathbb{E}_{x\sim P_g}[f(x)]$$
where the supremum is over 1-Lipschitz functions $f$. $W$ is continuous and differentiable almost everywhere even when distributions have disjoint support, providing stable gradients throughout training. The discriminator is replaced by a **critic** $f_w$ (not constrained to $[0,1]$), trained with gradient penalty (WGAN-GP) to enforce the Lipschitz constraint.

### 8.3 Discriminator Overpowering Generator

If $D$ becomes too strong before $G$ has learned, the generator receives no useful gradient signal. This is why the discriminator is often trained with fewer steps than the generator, or with label smoothing.

---

## 9. Summary

```
Minimax game:
  min_G max_D  E_{x~P_data}[log D(x)] + E_{x~P_g}[log(1-D(x))]

Step 1 — Optimal D (for fixed G):
  D*_G(x) = P_data(x) / (P_data(x) + P_g(x))

Step 2 — Substitute D*:
  min_G V(D*,G) = 2·JSD(P_data ‖ P_g) − 2 log 2  ≥  −2 log 2

Step 3 — Global optimum:
  P_g* = P_data   →   D* = 1/2 everywhere   →   JSD = 0
```

---

## 10. References

1. **Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y.** (2014). Generative adversarial nets. *NeurIPS 2014*. [Original GAN paper; minimax formulation and JSD connection.]
2. **Radford, A., Metz, L., & Chintala, S.** (2015). Unsupervised representation learning with deep convolutional generative adversarial networks (DCGAN). *ICLR 2016*. [Convolutional GAN architecture; training best practices.]
3. **Arjovsky, M., Chintala, S., & Bottou, L.** (2017). Wasserstein GAN. *ICML 2017*. [Replaces JSD with Wasserstein distance; addresses mode collapse and vanishing gradients.]
4. **Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A.** (2017). Improved training of Wasserstein GANs. *NeurIPS 2017*. [Gradient penalty (WGAN-GP) for enforcing Lipschitz constraint.]
5. **Brock, A., Donahue, J., & Simonyan, K.** (2019). Large scale GAN training for high fidelity natural image synthesis (BigGAN). *ICLR 2019*. [Class-conditional GANs at ImageNet scale.]
6. **Karras, T., Laine, S., & Aila, T.** (2019). A style-based generator architecture for generative adversarial networks (StyleGAN). *CVPR 2019*.
7. **Goodfellow, I.** (2016). NIPS 2016 tutorial: Generative adversarial networks. *arXiv:1701.00160*. [Comprehensive tutorial by the original author; covers theory, failure modes, and training tricks.]
8. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §8.1 (Graphical models background); see also Murphy (2023) *Probabilistic Machine Learning: Advanced Topics*, Ch. 26 (GANs).

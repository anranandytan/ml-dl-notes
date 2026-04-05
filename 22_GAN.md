# DL 13 — Generative Adversarial Networks (GAN)

> **Keywords:** implicit density, generator, discriminator, minimax game, optimal discriminator, Jensen-Shannon divergence, mode collapse, Wasserstein GAN, gradient penalty

---

## 1. Overview

A **GAN** (Goodfellow et al., 2014) is an **implicit density model**: the generator $G$ never defines $p_g(x)$ explicitly but learns a mapping from noise $z \sim p_Z$ to data space $x = G(z)$. The induced distribution $p_g$ is the pushforward of $p_Z$ through $G$. Sampling is trivial; evaluating $p_g(x)$ is not.

Training uses an adversarial signal from a **discriminator** $D$ that classifies real vs. generated samples. The generator is trained to fool the discriminator; the discriminator is trained to resist being fooled. At equilibrium, $p_g = p_{\mathrm{data}}$ and $D = 1/2$ everywhere.

---

## 2. Historical Notes

| Year | Event |
|------|-------|
| 2014 | **Goodfellow et al.** introduce GANs; prove the global optimum is $p_g = p_{\mathrm{data}}$. |
| 2015 | **DCGAN** (Radford et al.): convolutional architecture stabilises training. |
| 2017 | **WGAN** (Arjovsky et al.): Wasserstein distance replaces JSD; solves vanishing gradients and provides a meaningful training metric. |
| 2018 | **BigGAN** (Brock et al.): class-conditional generation at ImageNet scale. |
| 2019 | **StyleGAN** (Karras et al.): style-based generator for fine-grained control. |

---

## 3. Setup and Objectives

**Symbols:**

| Symbol | Role |
|--------|------|
| $p_{\mathrm{data}}$ | True data distribution |
| $p_Z$ | Noise prior (e.g., $\mathcal{N}(0,I)$) |
| $G(z;\theta_g)$ | Generator: maps $z$ to fake data |
| $p_g$ | Distribution of $G(z)$ |
| $D(x;\theta_d)\in[0,1]$ | Discriminator: probability $x$ is real |

**Discriminator objective** (maximise ability to classify):

$$\max_D\;V(D,G) = \mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_Z}[\log(1-D(G(z)))]$$

**Generator objective** (minimise discriminator's success):

$$\min_G\;\mathbb{E}_{z\sim p_Z}[\log(1-D(G(z)))]$$

**Combined minimax:**

$$\min_G\max_D\;V(D,G) = \mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)] + \mathbb{E}_{x\sim p_g}[\log(1-D(x))]$$

**Non-saturating generator loss.** In practice, early in training $D(G(z))\approx 0$, so $\log(1-D(G(z)))\approx 0$ with near-zero gradient. The standard fix is to maximise $\mathbb{E}[\log D(G(z))]$ instead, which has large gradient when $D(G(z))$ is small. This has the same Nash equilibrium but better gradient flow.

---

## 4. Optimal Discriminator

**Claim.** For fixed $G$:

$$D^*_G(x) = \frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x) + p_g(x)}$$

**Derivation.** The value function decomposes pointwise:

$$V(D,G) = \int [p_{\mathrm{data}}(x)\log D(x) + p_g(x)\log(1-D(x))]\,dx$$

For each $x$, maximise $f(D) = a\log D + b\log(1-D)$ where $a=p_{\mathrm{data}}(x)\geq 0$, $b=p_g(x)\geq 0$, $a+b>0$:

$$\frac{df}{dD} = \frac{a}{D} - \frac{b}{1-D} = 0 \quad\Rightarrow\quad D^*_G(x) = \frac{a}{a+b} = \frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}$$

This is a maximum since $d^2f/dD^2 = -a/D^2 - b/(1-D)^2 < 0$.

---

## 5. Global Optimum and Jensen-Shannon Divergence

Substitute $D^*_G$ into $V(D^*_G, G)$:

$$V(D^*_G, G) = \mathbb{E}_{p_{\mathrm{data}}}\!\left[\log\frac{p_{\mathrm{data}}}{p_{\mathrm{data}}+p_g}\right] + \mathbb{E}_{p_g}\!\left[\log\frac{p_g}{p_{\mathrm{data}}+p_g}\right]$$

Add and subtract $\log 2$ in each term:

$$= -\log 4 + D_{\mathrm{KL}}\!\left(p_{\mathrm{data}}\,\Big\|\,\frac{p_{\mathrm{data}}+p_g}{2}\right) + D_{\mathrm{KL}}\!\left(p_g\,\Big\|\,\frac{p_{\mathrm{data}}+p_g}{2}\right)$$

The **Jensen-Shannon divergence** is $\mathrm{JSD}(p\|q) = \frac{1}{2}D_{\mathrm{KL}}(p\|\frac{p+q}{2}) + \frac{1}{2}D_{\mathrm{KL}}(q\|\frac{p+q}{2}) \geq 0$, with equality iff $p=q$. Therefore:

$$\min_G V(D^*_G, G) = -\log 4 + 2\,\mathrm{JSD}(p_{\mathrm{data}} \| p_g) \geq -\log 4$$

The minimum is $-\log 4$ iff $\mathrm{JSD}=0$ iff $p_g = p_{\mathrm{data}}$. At the global optimum, $D^*_{G^*}(x) = 1/2$ everywhere.

---

## 6. Training Procedure

```
For each training iteration:
  1. Sample minibatch: {x_i} ~ p_data,  {z_i} ~ p_Z
  2. [Discriminator update]  (gradient ascent on V)
       θ_d ← θ_d + η ∇_{θ_d} [E[log D(x)] + E[log(1-D(G(z)))]]
     Repeat k times (k=1 typical; k=5 for WGAN)

  3. Sample fresh: {z_i} ~ p_Z
  4. [Generator update]  (gradient descent, non-saturating loss)
       θ_g ← θ_g + η ∇_{θ_g} [E[log D(G(z))]]
```

---

## 7. Failure Modes

### 7.1 Mode Collapse

The generator maps all noise inputs to a small subset of modes in $p_{\mathrm{data}}$. This satisfies the discriminator locally — a single mode may fool a poorly trained discriminator — but $p_g$ fails to cover all of $p_{\mathrm{data}}$.

**Why it happens:** The minimax objective measures the discriminator's ability to distinguish at the current $G$, not whether $G$ covers all modes. A generator that fools the current $D$ on a single mode can have low loss even if other modes are entirely absent.

**Remedies:** minibatch discrimination (encourages within-batch diversity), unrolled GANs, mode-seeking objectives.

### 7.2 Vanishing Gradients

When $p_g$ and $p_{\mathrm{data}}$ have disjoint support (common early in training), $D$ can perfectly classify and $\mathrm{JSD}=\log 2$ (constant). The gradient $\nabla_{\theta_g}V \approx 0$: the generator receives no training signal.

### 7.3 Wasserstein GAN (WGAN)

**Wasserstein-1 distance:**

$$W(p_{\mathrm{data}}, p_g) = \sup_{\|f\|_L\leq 1}\;\mathbb{E}_{x\sim p_{\mathrm{data}}}[f(x)] - \mathbb{E}_{x\sim p_g}[f(x)]$$

where the supremum is over 1-Lipschitz functions. Unlike JSD, $W$ is continuous and provides non-zero gradients even when distributions have disjoint support.

The discriminator is replaced by a **critic** $f_w$ (not constrained to $[0,1]$), trained to approximate the Wasserstein distance. The Lipschitz constraint is enforced via **gradient penalty** (WGAN-GP):

$$\mathcal{L}_D = \mathbb{E}_{p_g}[f_w] - \mathbb{E}_{p_{\mathrm{data}}}[f_w] + \lambda\mathbb{E}_{\hat{x}}\!\left[(\|\nabla_{\hat{x}}f_w(\hat{x})\|_2 - 1)^2\right]$$

where $\hat{x}$ is sampled uniformly along straight lines between real and generated samples.

---

## 8. Summary

```
Minimax:
  min_G max_D  E_{p_data}[log D(x)] + E_{p_g}[log(1-D(x))]

Optimal D (for fixed G):
  D*_G(x) = p_data(x) / (p_data(x) + p_g(x))

Global optimum:
  min_G V(D*_G, G) = -log 4 + 2·JSD(p_data ‖ p_g)
  Achieved at p_g = p_data,  D* = 1/2

Failure modes:
  Mode collapse:        G ignores parts of p_data
  Vanishing gradients:  disjoint support → JSD = log 2 (constant)

WGAN fix:
  Replace JSD with Wasserstein-1 distance
  Train critic (not constrained to [0,1]) with gradient penalty
```

---

## 9. References

1. **Goodfellow, I. et al.** (2014). Generative adversarial nets. *NeurIPS 2014*. [Original GAN; minimax, optimal discriminator, JSD.]
2. **Radford, A., Metz, L., & Chintala, S.** (2015). Unsupervised representation learning with DCGAN. *ICLR 2016*.
3. **Arjovsky, M., Chintala, S., & Bottou, L.** (2017). Wasserstein GAN. *ICML 2017*.
4. **Gulrajani, I. et al.** (2017). Improved training of Wasserstein GANs. *NeurIPS 2017*. [Gradient penalty (WGAN-GP).]
5. **Brock, A., Donahue, J., & Simonyan, K.** (2019). Large scale GAN training for high fidelity natural image synthesis (BigGAN). *ICLR 2019*.
6. **Karras, T., Laine, S., & Aila, T.** (2019). A style-based generator architecture for GANs (StyleGAN). *CVPR 2019*.
7. **Goodfellow, I.** (2016). NIPS 2016 tutorial: Generative adversarial networks. *arXiv:1701.00160*.

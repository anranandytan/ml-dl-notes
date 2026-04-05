# 21 — Variational Autoencoder

> **Keywords:** VAE, ELBO, amortised inference, reparameterisation trick, encoder, decoder, latent space, generative model

---

## 1. Motivation & Intuition

The **Variational Autoencoder** unifies two ideas:

1. **Deep latent variable models** — learn a low-dimensional latent representation $z$ that captures the underlying structure of data $x$, with a neural network decoder $p_\theta(x|z)$ mapping latent codes to observations.
2. **Variational inference** — since the true posterior $p_\theta(z|x)$ is intractable, approximate it with a learned encoder $q_\phi(z|x)$.

**Relationship to GMM.** A Gaussian Mixture Model (GMM) can be seen as a simple latent variable model:
- $z \sim \text{Categorical}(1,\ldots,K)$: discrete cluster assignment.
- $x|z=k \sim \mathcal{N}(\mu_k, \Sigma_k)$: Gaussian within each cluster.

A VAE generalises this to a **continuous, infinite mixture**:
- $z \sim \mathcal{N}(0, I)$: continuous, Gaussian latent variable.
- $x|z \sim \mathcal{N}(\mu_\theta(z), \Sigma_\theta(z))$: mean and covariance are **neural network outputs**.

This allows the model to capture far richer distributions than a finite Gaussian mixture.

**Why is $p_\theta(z|x)$ intractable?** The marginal likelihood:

$$p_\theta(x) = \int p_\theta(x|z)\,p(z)\,dz$$

requires integrating over the entire latent space — an intractable high-dimensional integral. Without $p_\theta(x)$, the posterior $p_\theta(z|x) = p_\theta(x|z)p(z)/p_\theta(x)$ cannot be evaluated.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1995 | **Helmholtz Machine** (Dayan, Hinton, Neal, Zemel): a directed latent variable model trained by the wake-sleep algorithm — a precursor to the VAE. |
| 2013–14 | **Kingma & Welling** and **Rezende, Mohamed & Wierstra** independently introduce the VAE, combining amortised variational inference with the reparameterisation trick. |
| 2017 | **Higgins et al.** introduce $\beta$-VAE: increasing the KL weight $\beta > 1$ encourages **disentangled** latent representations. |
| 2019 | **Burgess et al.** analyse the capacity-controlled $\beta$-VAE; **Tolstikhin et al.** introduce WAE (Wasserstein Autoencoder). |
| 2020–present | VAEs extended to images (VQVAE-2), text (optimus), graphs (GVAE); combined with diffusion models (latent diffusion). |

---

## 3. The Generative Model

**Prior:** $p(z) = \mathcal{N}(0, I)$

**Likelihood (decoder):** $p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), \Sigma_\theta(z))$

where $\mu_\theta$ and $\Sigma_\theta$ are neural networks with parameters $\theta$.

**Joint distribution:**

$$p_\theta(x) = \int p_\theta(x,z)\,dz = \int p(z)\,p_\theta(x|z)\,dz$$

This marginal is **intractable** for flexible decoders — no closed form exists.

---

## 4. ELBO Decomposition

Following the standard variational inference derivation, introduce an approximate posterior (encoder) $q_\phi(z|x)$. The ELBO decomposition gives:

$$\log p_\theta(x) = \underbrace{\mathcal{L}(\theta, \phi; x)}_{\text{ELBO}} + D_{\text{KL}}\!\big(q_\phi(z|x)\,\|\,p_\theta(z|x)\big)$$

Since $D_{\text{KL}} \geq 0$:

$$\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x,z) - \log q_\phi(z|x)\right]$$

The ELBO can be written in two equivalent forms:

**Form 1 — Joint and entropy:**

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x, z)\right] + H\!\left[q_\phi(z|x)\right]$$

**Form 2 — Reconstruction minus KL:**

$$\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x|z)\right]}_{\text{reconstruction term}} - \underbrace{D_{\text{KL}}\!\big(q_\phi(z|x)\,\|\,p(z)\big)}_{\text{regularisation term}}$$

**Derivation of Form 2:** Starting from Form 1:

$$\mathbb{E}_{q_\phi}[\log p_\theta(x,z)] + H[q_\phi] = \mathbb{E}_{q_\phi}[\log p_\theta(x|z) + \log p(z)] - \mathbb{E}_{q_\phi}[\log q_\phi(z|x)]$$

$$= \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] + \mathbb{E}_{q_\phi}[\log p(z) - \log q_\phi(z|x)]$$

$$= \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x)\,\|\,p(z))$$

**Interpretation of the two terms:**

- **Reconstruction term** $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]$: How well the decoder reconstructs $x$ from latent samples $z\sim q_\phi(z|x)$. Maximising this pushes $q_\phi$ to produce latent codes from which the decoder can recover $x$.
- **KL regularisation** $D_{\text{KL}}(q_\phi(z|x)\|p(z))$: How far the approximate posterior is from the prior $\mathcal{N}(0,I)$. Minimising this forces $q_\phi(z|x)$ to stay close to the prior, preventing the encoder from collapsing to deterministic point masses and regularising the latent space.

---

## 5. Relationship to EM

The ELBO optimisation has a natural EM interpretation:

**E-step:** Fix $\theta$; tighten the bound by choosing $q_\phi(z|x) = p_\theta(z|x)$, making $D_{\text{KL}} = 0$. The ELBO then equals $\log p_\theta(x)$ and the ELBO reduces to:

$$\mathcal{L} = \mathbb{E}_{p_\theta(z|x)}\!\left[\log p_\theta(x,z)\right] + \text{const}$$

This is the EM Q-function.

**M-step:** Fix $q$; maximise ELBO over $\theta$:

$$\theta^* = \arg\max_\theta\;\mathbb{E}_{p_\theta(z|x)}\!\left[\log p_\theta(x,z)\right]$$

**VAE vs. EM:**

| | EM | VAE |
|---|---|---|
| **E-step** | Compute exact posterior $p_\theta(z|x)$ | Fit neural network encoder $q_\phi(z|x) \approx p_\theta(z|x)$ |
| **M-step** | Optimise $\theta$ analytically | Gradient ascent on $\theta$ |
| **Scales to** | Small models with conjugacy | Large neural networks |
| **Posterior** | Exact (expensive) | Approximate (fast, amortised) |

The VAE replaces the exact E-step with **amortised inference**: train $q_\phi$ once across all data, so that inference for a new $x$ is a single encoder forward pass.

---

## 6. VAE Objective and Training

The full VAE objective jointly optimises $\theta$ (decoder) and $\phi$ (encoder):

$$\langle\hat{\theta}, \hat{\phi}\rangle = \arg\max_{\theta, \phi}\;\mathcal{L}(\theta, \phi; x) = \arg\max_{\theta,\phi}\;\mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x|z)\right] - D_{\text{KL}}\!\big(q_\phi(z|x)\,\|\,p(z)\big)$$

### 6.1 Encoder (Inference Network)

The encoder outputs the parameters of the approximate posterior:

$$q_\phi(z|x) = \mathcal{N}\!\big(z\,|\,\mu_\phi(x),\,\text{diag}(\sigma_\phi^2(x))\big)$$

where $\mu_\phi(x) \in \mathbb{R}^d$ and $\sigma_\phi^2(x) \in \mathbb{R}^d_{>0}$ are neural network outputs.

### 6.2 Reparameterisation

To backpropagate through the sampling step $z \sim q_\phi(z|x)$, use the reparameterisation trick:

$$\varepsilon \sim \mathcal{N}(0, I), \qquad z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$$

The randomness is isolated in $\varepsilon$, which has no parameters. The gradient flows deterministically through $\mu_\phi$ and $\sigma_\phi$.

### 6.3 KL in Closed Form

For Gaussian $q_\phi$ and standard Gaussian prior $p(z)=\mathcal{N}(0,I)$, the KL has an analytic form:

$$D_{\text{KL}}\!\big(\mathcal{N}(\mu_\phi, \text{diag}(\sigma_\phi^2))\,\|\,\mathcal{N}(0,I)\big) = \frac{1}{2}\sum_{j=1}^d\!\left(\mu_{\phi,j}^2 + \sigma_{\phi,j}^2 - \log\sigma_{\phi,j}^2 - 1\right)$$

This requires no sampling — it is computed analytically and backpropagated exactly.

### 6.4 Training Algorithm

```
For each minibatch {x^(1), ..., x^(m)}:

  For each x^(i):
    1. [Encode]   Compute mu_phi(x^(i)), sigma_phi(x^(i)) via encoder network
    2. [Sample]   eps ~ N(0, I),   z^(i) = mu_phi + sigma_phi * eps
    3. [Decode]   Compute p_theta(x^(i) | z^(i)) via decoder network

  4. [ELBO]
       L = (1/m) * sum_i [
             E_q[log p_theta(x^(i) | z^(i))]      [reconstruction loss]
           - KL(q_phi(z|x^(i)) || p(z))            [KL term, closed form]
         ]

  5. [Update]
       theta, phi <- theta, phi + eta * grad_{theta,phi} L
```

---

## 7. The Latent Space

A well-trained VAE has several desirable properties of its latent space:

- **Continuity:** Nearby points in latent space decode to similar outputs.
- **Completeness:** Any point sampled from the prior $p(z)=\mathcal{N}(0,I)$ decodes to a plausible output — not just points that were encoded from training data.
- **Interpolation:** A linear interpolation between two latent codes produces a smooth visual transition between the corresponding decoded images.

These properties follow from the KL regularisation: by forcing $q_\phi(z|x)$ toward $\mathcal{N}(0,I)$, the encoder cannot map different inputs to widely separated or disjoint regions of the latent space.

---

## 8. Extensions

| Extension | Idea | Reference |
|---|---|---|
| **$\beta$-VAE** | Upweight KL: $\mathcal{L} = \mathbb{E}[\log p_\theta(x|z)] - \beta\cdot D_{\text{KL}}$ for $\beta>1$. Encourages disentangled representations. | Higgins et al. (2017) |
| **IWAE** | Replace ELBO with a tighter importance-weighted bound: $\log\mathbb{E}[\frac{p_\theta(x,z)}{q_\phi(z|x)}]$. Lower variance, tighter bound. | Burda et al. (2016) |
| **VQ-VAE** | Discrete latent space via vector quantisation; avoids posterior collapse. | van den Oord et al. (2017) |
| **Latent Diffusion** | Run a diffusion model in the VAE's latent space for high-quality generation. | Rombach et al. (2022) |

---

## 9. Summary

```
Generative model:
  p(z)      = N(0, I)                  [prior]
  p_theta(x|z) = N(mu_theta(z), ...)   [decoder]

Intractability: p_theta(x) = integral p(z)p_theta(x|z)dz is intractable

Variational bound:
  log p_theta(x) = ELBO + KL(q_phi || p_theta(z|x))
                 >= ELBO

ELBO (two forms):
  L = E_q[log p_theta(x,z)] + H[q]
    = E_q[log p_theta(x|z)] - KL(q_phi(z|x) || p(z))

Training:
  Encoder: q_phi(z|x) = N(mu_phi(x), diag(sigma_phi^2(x)))
  Reparameterisation: z = mu_phi(x) + sigma_phi(x) * eps, eps ~ N(0,I)
  KL: closed form for Gaussian q and p
  Optimise theta (decoder) and phi (encoder) jointly by gradient ascent on ELBO
```

---

## 10. References

1. **Kingma, D. P. & Welling, M.** (2014). Auto-encoding variational Bayes. *ICLR 2014*. [Original VAE paper; reparameterisation trick and amortised inference.]
2. **Rezende, D. J., Mohamed, S., & Wierstra, D.** (2014). Stochastic backpropagation and approximate inference in deep generative models. *ICML 2014*. [Independent derivation of VAE and reparameterisation.]
3. **Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A.** (2017). beta-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR 2017*. [Disentangled representations via increased KL weight.]
4. **Burda, Y., Grosse, R., & Salakhutdinov, R.** (2016). Importance weighted autoencoders. *ICLR 2016*. [IWAE: tighter bound via importance sampling.]
5. **van den Oord, A., Vinyals, O., & Kavukcuoglu, K.** (2017). Neural discrete representation learning (VQ-VAE). *NeurIPS 2017*. [Discrete latent space; avoids posterior collapse.]
6. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). High-resolution image synthesis with latent diffusion models. *CVPR 2022*. [Stable Diffusion; runs diffusion in VAE latent space.]
7. **Doersch, C.** (2016). Tutorial on variational autoencoders. *arXiv:1606.05908*. [Accessible introductory tutorial.]
8. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. §10.1 (Variational inference), §12.1 (PCA, as a simpler latent variable model).

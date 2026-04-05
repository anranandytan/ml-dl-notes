# Diffusion Models

> **Keywords:** score matching, stochastic differential equation, Fokker-Planck, Langevin dynamics, DDPM, DDIM, classifier guidance, classifier-free guidance, flow matching

---

## 1. Motivation & Intuition

The central goal of a generative model is to learn a distribution $p_0$ from data and then draw new samples from it. Classical approaches (VAE, GAN) parametrise the distribution directly, which can be hard to train stably.

**Diffusion models** take a different route: they define a **forward process** that gradually destroys the data by adding noise until the data becomes pure Gaussian noise, and then learn to **reverse** this process to recover data from noise.

The intuition is elegant:

- **Forward process:** $x_0$ (data) $\to$ $x_1$ (slightly noisy) $\to \cdots\to$ $x_T$ ($\approx \mathcal{N}(0, \sigma^2 I)$). Each step is simple: add a small Gaussian perturbation.
- **Reverse process:** Learn a neural network to undo each noisy step. At inference, start from pure noise and apply the learned reverse steps to generate data.

The key insight connecting all diffusion models is the **score function** $\nabla\log p_t(x)$ — the gradient of the log-density. Knowing the score at every noise level is equivalent to knowing how to denoise, how to sample, and how to compute the reverse SDE.

---

## 2. Historical Context

| Year | Event |
|------|-------|
| 1905 | Einstein's theory of Brownian motion; Langevin dynamics introduced for physical systems. |
| 1931 | Fokker-Planck equation derived, describing the time evolution of probability densities under stochastic dynamics. |
| 1950s | Itô stochastic calculus formalised by Kiyosi Itô. |
| 2005 | **Hyvärinen** introduces score matching — a method for estimating $\nabla\log p(x)$ without computing the partition function. |
| 2011 | **Vincent** shows denoising score matching is equivalent to score matching — the key bridge to practical training. |
| 2019 | **Song & Ermon** introduce Noise Conditional Score Networks (NCSN): train a single network to match scores at many noise levels. |
| 2020 | **Ho et al.** introduce DDPM (Denoising Diffusion Probabilistic Models), connecting diffusion to score matching via a simple $\epsilon$-prediction objective. |
| 2020 | **Song et al.** unify all diffusion models under a single **SDE framework**, with DDPM as VP-SDE and NCSN as VE-SDE. |
| 2021 | **Song et al.** introduce DDIM (deterministic sampler), drastically reducing sampling steps. |
| 2022 | **Dhariwal & Nichol** introduce classifier guidance; **Ho & Salimans** introduce classifier-free guidance (CFG). |
| 2022–present | **Flow Matching** (Lipman et al., Liu et al.) as a cleaner, simulation-free alternative to score-based diffusion. |

---

## 3. Mathematical Prerequisites

### 3.1 Gaussian Distribution

A $d$-dimensional Gaussian $x \sim \mathcal{N}(\mu, \Sigma)$ has density:

$$p(x) = \frac{1}{(2\pi)^{d/2}\det(\Sigma)^{1/2}} \exp\!\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right)$$

**Key property (convolution):** If $x \sim p_0$ and $\varepsilon \sim \mathcal{N}(0, tI)$ independently, then $x + \varepsilon \sim p_0 * \mathcal{N}(0, tI)$, i.e., the density of $x+\varepsilon$ is the convolution of $p_0$ with a Gaussian kernel.

### 3.2 Brownian Motion

**Definition.** A stochastic process $\{B_t\}_{t \geq 0}$ is a **Brownian motion** (Wiener process) if:
1. $B_0 = 0$.
2. **Independent increments:** For $0 \leq t_1 < t_2 < \cdots < t_n$, the increments $B_{t_2}-B_{t_1}, \ldots, B_{t_n}-B_{t_{n-1}}$ are mutually independent.
3. **Stationary increments:** For any $s, t \geq 0$, $B_{t+s} - B_s \sim \mathcal{N}(0, tI)$. (The distribution depends only on the elapsed time $t$, not the starting time $s$.)
4. Paths $t \mapsto B_t$ are almost surely continuous.

**Consequence.** If $x_0 \sim p_0$ and $dx_t = dB_t$ (pure noise injection), then:
$$x_t - x_0 \sim \mathcal{N}(0,\, tI), \qquad x_t \sim p_t = p_0 * \mathcal{N}(0,\, tI)$$
At large $t$, $p_t$ approaches a Gaussian regardless of $p_0$.

### 3.3 Ordinary Differential Equations and the Chain Rule

For the ODE $dx_t = v_t(x_t)\,dt$ and a smooth function $F : \mathbb{R}^d \to \mathbb{R}$, the standard chain rule gives:

$$\frac{d}{dt}F(x_t) = \langle \nabla F(x_t),\, v_t(x_t) \rangle$$

To derive this from first principles using Euler discretisation: $x_{t+\delta} \approx x_t + \delta v(x_t)$, so:
$$F(x_{t+\delta}) \approx F(x_t) + \langle \nabla F(x_t), \delta v(x_t)\rangle + \frac{\delta^2}{2} v(x_t)^\top \nabla^2 F(x_t)\, v(x_t) + O(\delta^3)$$

Dividing by $\delta$ and letting $\delta \to 0$, the $O(\delta)$ term dominates and:
$$\frac{d}{dt}F(x_t) = \langle \nabla F(x_t), v(x_t)\rangle$$

**Example.** If $v(x_t) = -\nabla F(x_t)$ (gradient flow), then $\frac{d}{dt}F(x_t) = -\|\nabla F(x_t)\|^2 \leq 0$: the function decreases along trajectories, so this ODE finds a minimum of $F$.

### 3.4 Stochastic Differential Equations and Itô's Lemma

When $dx_t = dB_t$ (pure Brownian motion), the standard chain rule breaks down because Brownian increments are of order $\sqrt{\delta}$, not $\delta$. Let $\xi := \frac{B_{t+\delta}-B_t}{\sqrt{\delta}} \sim \mathcal{N}(0, I)$, so $x_{t+\delta} \approx x_t + \sqrt{\delta}\,\xi$.

Taylor-expanding $F$ around $x_t$:
$$F(x_{t+\delta}) \approx F(x_t) + \langle \nabla F(x_t),\, \sqrt{\delta}\,\xi \rangle + \frac{1}{2}(\sqrt{\delta}\,\xi)^\top \nabla^2 F(x_t)(\sqrt{\delta}\,\xi) + O(\delta^{3/2})$$

Taking expectations (using $\mathbb{E}[\xi] = 0$ and $\mathbb{E}[\xi\xi^\top] = I$):

$$\frac{d}{dt}\mathbb{E}[F(x_t)] = \lim_{\delta \to 0}\frac{\mathbb{E}[F(x_{t+\delta})]-\mathbb{E}[F(x_t)]}{\delta}$$

The first-order term vanishes: $\mathbb{E}[\langle \nabla F(x_t), \sqrt{\delta}\,\xi\rangle] = \sqrt{\delta}\langle \nabla F(x_t), \mathbb{E}[\xi]\rangle = 0$.

The second-order term survives:
$$\frac{1}{2\delta}\mathbb{E}\!\left[\delta\,\xi^\top \nabla^2 F(x_t)\,\xi\right] = \frac{1}{2}\mathbb{E}\!\left[\text{tr}(\nabla^2 F(x_t)\,\mathbb{E}[\xi\xi^\top])\right] = \frac{1}{2}\mathbb{E}[\text{tr}(\nabla^2 F(x_t))]$$

Therefore, **Itô's lemma** for $dx_t = dB_t$:

$$\boxed{\frac{d}{dt}\mathbb{E}[F(x_t)] = \frac{1}{2}\mathbb{E}[\text{tr}(\nabla^2 F(x_t))]}$$

The extra $\frac{1}{2}\text{tr}(\nabla^2 F)$ term compared to the ODE case is the **Itô correction** — a direct consequence of Brownian motion having quadratic variation $\langle B, B\rangle_t = t$.

**Example.** For $F(x) = \|x\|^2$: $\nabla^2 F = 2I$, so $\text{tr}(\nabla^2 F) = 2d$. With $dx_t = dB_t$:
$$\frac{d}{dt}\mathbb{E}[\|x_t\|^2] = \frac{1}{2}\mathbb{E}[2d] = d$$
Integrating: $\mathbb{E}[\|x_T\|^2] = \mathbb{E}[\|x_0\|^2] + dT$. This is consistent with $x_T = x_0 + B_T$ and $\mathbb{E}[\|B_T\|^2] = dT$. ✓

**General SDE.** For $dx_t = v_t(x_t)\,dt + g(t)\,dB_t$ (drift $v_t$ plus diffusion $g(t)$):

$$\boxed{\frac{d}{dt}\mathbb{E}[F(x_t)] = \mathbb{E}[\langle \nabla F(x_t), v_t(x_t)\rangle] + \frac{g(t)^2}{2}\mathbb{E}[\text{tr}(\nabla^2 F(x_t))]}$$

The drift contributes via the standard chain rule; the diffusion contributes via the Itô correction.

### 3.5 Variance-Exploding vs. Variance-Preserving SDEs

Two fundamental SDE types for diffusion models:

**Variance-Exploding (VE) SDE:** $dx_t = dB_t$

$$\frac{d}{dt}\mathbb{E}[\|x_t\|^2] = d \quad \Rightarrow \quad \mathbb{E}[\|x_t\|^2] = \mathbb{E}[\|x_0\|^2] + dt \to \infty$$

The variance grows without bound. Simple but the scale of $x_t$ drifts far from data.

**Variance-Preserving (VP) SDE:** $d\bar{x}_t = -\frac{1}{2}\bar{x}_t\,dt + dB_t$

Using Itô's lemma with $F(x) = \|x\|^2$, $\nabla F = 2x$, $\nabla^2 F = 2I$:

$$\frac{d}{dt}\mathbb{E}[\|\bar{x}_t\|^2] = \mathbb{E}\!\left[\left\langle 2\bar{x}_t, -\tfrac{1}{2}\bar{x}_t\right\rangle\right] + \frac{1}{2}\mathbb{E}[2d] = -\mathbb{E}[\|\bar{x}_t\|^2] + d$$

This ODE (in $m(t) = \mathbb{E}[\|\bar{x}_t\|^2]$) has solution:
$$m(t) = e^{-t}(m(0) - d) + d$$

As $t \to \infty$, $m(t) \to d$, meaning the variance stabilises — hence "variance-preserving" in steady state. Crucially, for well-initialised data the variance stays approximately constant throughout the process.

The explicit solution of the VP SDE is:
$$\bar{x}_t = e^{-t/2}x_0 + \int_0^t e^{-(t-s)/2}\,dB_s$$

The stochastic integral $\int_0^t e^{-(t-s)/2}\,dB_s$ is Gaussian with variance $\int_0^t e^{-(t-s)}\,ds = 1 - e^{-t}$. So:

$$\bar{x}_t = e^{-t/2}x_0 + \sqrt{1-e^{-t}}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This is a weighted mixture of the clean data and noise, with the data term decaying exponentially. In the DDPM literature, defining $\alpha_t = \frac{1}{1+\sigma_t^2}$ (where $\sigma_t = \sqrt{\frac{1-\alpha_t}{\alpha_t}}$), this is written as:

$$\bar{x}_t = \sqrt{\alpha_t}\,x_0 + \sqrt{1-\alpha_t}\,\epsilon$$

---

## 4. Fokker-Planck Equation

The **Fokker-Planck equation** describes how the probability density $p_t(x)$ of the process $x_t$ evolves over time.

### 4.1 Integration by Parts (Key Tool)

For smooth functions $f : \mathbb{R}^d \to \mathbb{R}$ and $g : \mathbb{R}^d \to \mathbb{R}^d$ decaying sufficiently fast at infinity, the Gauss divergence theorem gives $\int_{\mathbb{R}^d} \text{div}(fg)\,dx = 0$. Expanding the product rule:

$$\text{div}(f(x)g(x)) = \langle \nabla f(x), g(x)\rangle + f(x)\,\text{div}(g(x))$$

Integrating and using $\int \text{div}(fg)\,dx = 0$:

$$\boxed{\int_{\mathbb{R}^d} f(x)\,\text{div}(g(x))\,dx = -\int_{\mathbb{R}^d} \langle \nabla f(x), g(x)\rangle\,dx}$$

This is the **integration by parts formula** (Green's identity in $\mathbb{R}^d$) and is used repeatedly below.

### 4.2 Derivation of Fokker-Planck

Consider $dx_t = v_t(x_t)\,dt$. For any smooth test function $f$:

**Side A** — differentiate the integral:
$$\frac{d}{dt}\mathbb{E}[f(x_t)] = \frac{d}{dt}\int f(x)\,p_t(x)\,dx = \int f(x)\,\frac{\partial}{\partial t}p_t(x)\,dx$$

**Side B** — apply the ODE chain rule and use integration by parts:
$$\frac{d}{dt}\mathbb{E}[f(x_t)] = \mathbb{E}[\langle \nabla f(x_t), v_t(x_t)\rangle] = \int \langle \nabla f(x), v_t(x)\rangle p_t(x)\,dx = -\int f(x)\,\text{div}(p_t(x)v_t(x))\,dx$$

Since $f$ is arbitrary, equating the integrands:
$$\frac{\partial}{\partial t}p_t(x) = -\text{div}(p_t(x)v_t(x)) \qquad \textbf{(Continuity equation)}$$

For $dx_t = dB_t$, Side B uses Itô's lemma instead:
$$\frac{d}{dt}\mathbb{E}[f(x_t)] = \frac{1}{2}\mathbb{E}[\text{tr}(\nabla^2 f)] = \frac{1}{2}\int \text{div}(\nabla f(x))\,p_t(x)\,dx = \frac{1}{2}\int f(x)\,\text{div}(\nabla p_t(x))\,dx$$

where the last step applies integration by parts twice. So for $dx_t = dB_t$:
$$\frac{\partial}{\partial t}p_t(x) = \frac{1}{2}\text{div}(\nabla p_t(x)) = \frac{1}{2}\Delta p_t(x) \qquad \textbf{(Heat equation)}$$

Combining both cases for the general SDE $dx_t = v_t(x_t)\,dt + g(t)\,dB_t$:

$$\boxed{\frac{\partial}{\partial t}p_t(x) = -\text{div}(p_t(x)v_t(x)) + \frac{g(t)^2}{2}\text{div}(\nabla p_t(x))}$$

This is the **Fokker-Planck equation**. It is a PDE for $p_t$ given the SDE coefficients $v_t$ and $g(t)$.

---

## 5. Score Function and Score Matching

### 5.1 Definition of the Score

The **score function** of a distribution with density $p_t$ is:
$$\nabla_x \log p_t(x)$$

This is a vector field pointing in the direction of increasing log-density — i.e., toward regions of higher probability. If we can evaluate the score everywhere, we can run gradient ascent on $\log p_t$ to move toward high-density regions, which is the basis of Langevin sampling.

### 5.2 Score for the VE SDE

Under $dx_t = dB_t$, the marginal density at time $t$ is:
$$p_t(x) = \int p_0(x_0)\,q_t(x \mid x_0)\,dx_0, \qquad q_t(x \mid x_0) = \mathcal{N}(x \mid x_0,\, tI) = \frac{1}{(2\pi t)^{d/2}} e^{-\frac{\|x-x_0\|^2}{2t}}$$

Differentiating $\log p_t(x)$ with respect to $x$:
$$\nabla \log p_t(x) = \frac{\nabla_x p_t(x)}{p_t(x)} = \frac{\int p_0(x_0)\,\nabla_x q_t(x \mid x_0)\,dx_0}{p_t(x)}$$

Computing $\nabla_x q_t(x \mid x_0) = q_t(x \mid x_0) \cdot \frac{x_0 - x}{t}$, so:
$$\nabla \log p_t(x) = \frac{\int p_0(x_0)\,q_t(x \mid x_0)\,\frac{x_0-x}{t}\,dx_0}{p_t(x)} = \mathbb{E}\!\left[\frac{x_0 - x_t}{t}\,\bigg|\,x_t = x\right]$$

$$\boxed{\nabla \log p_t(x) = \frac{\mathbb{E}[x_0 \mid x_t = x] - x}{t}}$$

**Three key takeaways:**

1. The score points toward $\mathbb{E}[x_0 \mid x_t]$: it always points from the noisy observation back toward the (expected) clean data. Denoising and score estimation are the **same task**.

2. The $\frac{1}{t}$ factor scales the magnitude: at small $t$ (little noise), the score has large magnitude; at large $t$ (high noise), the score is small. This scale variation complicates training.

3. Writing the noise explicitly: $x_t = x_0 + \sqrt{t}\,\varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$, so $x_0 - x_t = -\sqrt{t}\,\varepsilon$:
$$\nabla \log p_t(x) = \frac{\mathbb{E}[-\sqrt{t}\,\varepsilon \mid x_t = x]}{t} = -\frac{\mathbb{E}[\varepsilon \mid x_t = x]}{\sqrt{t}}$$

### 5.3 Denoising Score Matching Loss

We want to train a network $s_\theta(x_t, t)$ to approximate $\nabla\log p_t(x_t)$. The naive objective:
$$\min_\theta\;\mathbb{E}\!\left[\|s_\theta(x_t, t) - \nabla\log p_t(x_t)\|^2\right]$$
is intractable because $\nabla\log p_t(x_t)$ requires integrating over all $x_0$.

**Denoising score matching** (Vincent, 2011) replaces this with a tractable surrogate. Define $h_{x_0}(v) = \|v - \frac{x_0-x_t}{t}\|^2$. Then:
$$L(v) = \mathbb{E}_{x_0 \mid x_t}\!\left[\left\|v - \nabla\log p_t(x_t)\right\|^2\right] = \mathbb{E}_{x_0 \mid x_t}[h_{x_0}(v)] + \text{const}$$

Since each $h_{x_0}(v)$ is convex in $v$ and the expectation of a convex function is convex, $L(v)$ is convex. Setting $\nabla_v L(v^*) = 0$:
$$\mathbb{E}_{x_0 \mid x_t}\!\left[2(v^* - \tfrac{x_0-x_t}{t})\right] = 0 \;\Rightarrow\; v^* = \frac{\mathbb{E}[x_0 \mid x_t] - x_t}{t} = \nabla\log p_t(x_t)$$

Therefore, minimising over $(x_0, x_t)$ jointly:

$$\boxed{\min_\theta\;\mathbb{E}_{x_0,\, \varepsilon \sim \mathcal{N}(0, tI),\, x_t = x_0+\varepsilon}\!\left[\left\|s_\theta(x_t, t) - \frac{x_0 - x_t}{t}\right\|^2\right]}$$

Under infinite capacity, the minimiser equals $\nabla\log p_t(x_t)$. This expectation is tractable: sample $x_0$ from data, sample $\varepsilon$, form $x_t = x_0 + \varepsilon$, then evaluate the squared error.

### 5.4 Reparameterised Training Objectives

Training $s_\theta$ to match $\nabla\log p_t$ directly is unstable because $\|\nabla\log p_t\|$ grows as $t\to 0$. Three equivalent parameterisations, each numerically better in different regimes:

| Target | Network | Loss | Relationship |
|--------|---------|------|-------------|
| Score prediction | $s_\theta(x_t, t) \approx \nabla\log p_t$ | $\mathbb{E}[\|s_\theta - \frac{x_0-x_t}{t}\|^2]$ | Direct |
| Noise prediction | $\varepsilon_\theta(x_t, t) \approx \varepsilon$ | $\mathbb{E}[\|\varepsilon_\theta + \varepsilon\|^2]$ | $\varepsilon_\theta = -\sqrt{t}\,s_\theta$ |
| $x_0$ prediction | $D_\theta(x_t, t) \approx x_0$ | $\mathbb{E}[\|D_\theta - x_0\|^2]$ | $D_\theta = x_t + t\,s_\theta$ |

To see the noise-prediction form: write $x_t = x_0 + \sqrt{t}\,\varepsilon$ (with $\varepsilon \sim \mathcal{N}(0,I)$). Then $\frac{x_0-x_t}{t} = \frac{-\sqrt{t}\,\varepsilon}{t} = -\frac{\varepsilon}{\sqrt{t}}$. Define $\varepsilon_\theta$ so that $\varepsilon_\theta(x_t,t) = -\sqrt{t}\,s_\theta(x_t,t)$, i.e. $s_\theta \approx \sqrt{t}\nabla\log p_t$; then the loss becomes $\mathbb{E}[\|\varepsilon_\theta + \varepsilon\|^2]$. This is the **DDPM** loss.

---

## 6. General SDE Framework

### 6.1 General Forward SDE

Consider the general SDE with noise schedule $g(t)$:
$$dx_t = g(t)\,dB_t$$

The induced marginal density satisfies $p_t = p_0 * \mathcal{N}(0, \sigma_t^2 I)$ where $\sigma_t^2 = \int_0^t g(s)^2\,ds$.

The score generalises to:
$$\nabla\log p_t(x) = \frac{\mathbb{E}[x_0 \mid x_t = x] - x}{\sigma_t^2}$$

and the score matching loss is:
$$\min_\theta\;\mathbb{E}_{x_0,\,\epsilon\sim\mathcal{N}(0,I),\,x_t = x_0+\sigma_t\epsilon}\!\left[\left\|s_\theta(x_t, t) - \frac{x_0-x_t}{\sigma_t^2}\right\|^2\right]$$

### 6.2 Probability Flow ODE

For the SDE $dx_t = g(t)\,dB_t$, the **probability flow ODE** (Song et al., 2020) has the same marginal distributions $p_t$ as the SDE but is deterministic:

$$dx_t = -\frac{g(t)^2}{2}\nabla\log p_t(x_t)\,dt$$

To verify: the Fokker-Planck equation for this ODE is $\partial_t p_t = \frac{g(t)^2}{2}\text{div}(p_t\nabla\log p_t) = \frac{g(t)^2}{2}\text{div}(\nabla p_t)$, which matches the Fokker-Planck of the original SDE ✓.

In practice, writing $g(t)^2 = 2\sigma_t\dot{\sigma}_t$ (where $\dot{\sigma}_t = d\sigma_t/dt$):
$$dx_t = -\dot{\sigma}_t\sigma_t\nabla\log p_t(x_t)\,dt = \frac{\dot{\sigma}_t}{\sigma_t}(x_t - \mathbb{E}[x_0|x_t])\,dt$$

### 6.3 Reverse Process: DDIM Sampler

To generate samples, run the probability flow ODE backward from $t = T$ (noise) to $t = 0$ (data). With the noise-prediction network $\varepsilon_\theta(x_t, t) \approx \varepsilon$ and $\varepsilon_\theta = \sigma_t\nabla\log p_t(x_t)$ (noting $\sigma_t^2\nabla\log p_t = \mathbb{E}[x_0|x_t]-x_t$), Euler discretisation from $x_t$ to $x_{t-\delta}$:

$$x_{t-\delta} = x_t + (\sigma_{t-\delta} - \sigma_t)\cdot\frac{\varepsilon_\theta(x_t, t)}{\sigma_t}\cdot\sigma_t = x_t + \frac{\sigma_{t-\delta}-\sigma_t}{\sigma_t}\cdot\varepsilon_\theta(x_t,t)\cdot\sigma_t$$

More precisely, from the ODE $dx_t = \dot{\sigma}_t\sigma_t\nabla\log p_t(x_t)\,dt$ (going forward), the reverse step integrates:
$$x_{t-\delta} = x_t + \nabla\log p_t(x_t)\int_{t-\delta}^{t}\dot{\sigma}_s\sigma_s\,ds = x_t + \nabla\log p_t(x_t)\cdot\frac{\sigma_t^2 - \sigma_{t-\delta}^2}{2}$$

In terms of $\varepsilon_\theta(x_t, t) = \sigma_t\nabla\log p_t(x_t)$:

$$\boxed{x_{t-\delta} = x_t + (\sigma_{t-\delta}-\sigma_t)\,\varepsilon_\theta(x_t, t)}$$

This is the **DDIM** update rule. Since it is deterministic (no added noise), we can take large steps ($\delta$ large) and need far fewer steps than stochastic DDPM sampling.

**VP/DDPM form.** In the variance-preserving parameterisation, define $\bar{x}_t = x_t/\sqrt{1+\sigma_t^2}$, $\alpha_t = 1/(1+\sigma_t^2)$. The DDIM update becomes:
$$\bar{x}_{t-\delta} = \sqrt{\alpha_{t-\delta}}\left(\frac{\bar{x}_t}{\sqrt{\alpha_t}} + \varepsilon_\theta(\bar{x}_t)\left(\sqrt{\frac{1-\alpha_{t-\delta}}{\alpha_{t-\delta}}} - \sqrt{\frac{1-\alpha_t}{\alpha_t}}\right)\right)$$

### 6.4 Stochastic Sampler: Reverse SDE + Langevin

For a stochastic reverse process, mix in Langevin noise with coefficient $c(r)$. Let $y_r = x_{T-r}$ (reverse time), $q_r = p_{T-r}$. The reverse SDE is:
$$dy_r = \frac{1}{2}g(T-r)^2\nabla\log q_r(y_r)\,dr + \frac{1}{2}c(r)^2\nabla\log q_r(y_r)\,dr + c(r)\,dW_r$$

where $W_r$ is a fresh Brownian motion independent of the forward process. Setting $c(r) = 0$ recovers the deterministic reverse ODE.

---

## 7. Langevin Dynamics and Stationary Distribution

Given a target distribution $\pi(x) \propto e^{-U(x)}$, **Langevin dynamics** are defined by the SDE:
$$dx_t = \frac{1}{2}\nabla\log\pi(x_t)\,dt + dB_t$$

**Claim:** $\pi$ is the stationary distribution of this SDE.

**Proof via Fokker-Planck.** Assume $q_t = \pi$. Then:
$$\frac{\partial}{\partial t}q_t(x) = -\text{div}\!\left(q_t(x)\,\frac{1}{2}\nabla\log\pi(x)\right) + \frac{1}{2}\text{div}(\nabla q_t(x))$$

$$= -\frac{1}{2}\text{div}(\pi(x)\nabla\log\pi(x)) + \frac{1}{2}\text{div}(\nabla\pi(x))$$

Since $\pi\nabla\log\pi = \pi\cdot\frac{\nabla\pi}{\pi} = \nabla\pi$:

$$= -\frac{1}{2}\text{div}(\nabla\pi(x)) + \frac{1}{2}\text{div}(\nabla\pi(x)) = 0 \checkmark$$

The time derivative of $q_t$ is zero when $q_t = \pi$, confirming stationarity.

### 7.1 Time Reparameterisation

A more general Langevin SDE uses a time-varying diffusion coefficient $c(t)$:
$$dx_t = \frac{c(t)^2}{2}\nabla\log\pi(x_t)\,dt + c(t)\,dB_t$$

The stationary distribution is still $\pi$ (by the same verification), but the speed of convergence is scaled by $c(t)^2$.

Euler–Maruyama discretisation:
$$x_{t+\delta} \approx x_t + \nabla\log\pi(x_t)\int_t^{t+\delta}\frac{c(s)^2}{2}\,ds + \sqrt{\int_t^{t+\delta}c(s)^2\,ds}\;\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 7.2 Convergence Rate (Log-Sobolev Inequality)

If $\pi \propto e^{-U(x)}$ satisfies the **$\rho$-Log-Sobolev Inequality (LSI)** — meaning for any smooth $f$ with $\mathbb{E}_\pi[f^2]=1$:
$$\text{Ent}_\pi(f^2) := \mathbb{E}_\pi[f^2\log f^2] \leq \frac{2}{\rho}\mathbb{E}_\pi[\|\nabla f\|^2]$$

then the Langevin dynamics converge exponentially fast:

$$D_{\text{KL}}(q_t \| \pi) \leq e^{-2\rho t}\,D_{\text{KL}}(q_0 \| \pi)$$

By Pinsker's inequality ($d_{\text{TV}}^2 \leq \frac{1}{2}D_{\text{KL}}$) and the Otto–Villani theorem ($W_2^2 \leq \frac{2}{\rho}D_{\text{KL}}$):
$$d_{\text{TV}}(q_t, \pi) \leq Ce^{-\rho t}, \qquad W_2(q_t, \pi) \leq C'e^{-\rho t}$$

**Sufficient condition (Bakry–Émery criterion).** If $U(x) = -\log\pi(x)$ is $\rho$-strongly convex ($\nabla^2 U(x) \succeq \rho I$ for all $x$), then $\pi$ satisfies $\rho$-LSI. For Gaussians $\pi = \mathcal{N}(\mu, \Sigma)$, $U(x) = \frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)$ and $\rho = \lambda_{\min}(\Sigma^{-1})$.

---

## 8. Guidance

After training an unconditional score model $\nabla\log p_t(x)$, we often want to steer generation toward a condition $c$ (a class label, a text prompt, etc.).

### 8.1 Classifier Guidance

**Bayes' rule** applied to scores:
$$\nabla_x\log p_t(x \mid c) = \nabla_x\log p_t(x) + \nabla_x\log p_t(c \mid x)$$

This decomposes the conditional score into the unconditional score (from a pre-trained diffusion model) plus the classifier score $\nabla_x\log p_t(c \mid x)$ (from a classifier trained on noisy inputs).

The corresponding ODE becomes:
$$dx_t = -\frac{g(t)^2}{2}\Big[\nabla\log p_t(x_t) + \nabla\log p_t(c \mid x_t)\Big]\,dt$$

**Guidance scale.** Introduce a scale $\omega \geq 0$ to amplify or dampen the classifier signal:
$$\nabla\log p_t(x \mid c;\,\omega) = \nabla\log p_t(x) + \omega\,\nabla\log p_t(c \mid x)$$

This corresponds to a tempered distribution $p_t(x \mid c;\,\omega) \propto p_t(x)\,p_t(c \mid x)^\omega$.

| $\omega$ | Effect |
|----------|--------|
| $1$ | True conditional score |
| $>1$ | Amplified conditioning — higher fidelity to $c$, reduced diversity |
| $\in[0,1)$ | Down-weighted conditioning — more diversity |
| $<0$ | Negative guidance — steers **away** from $c$ |

**Limitation.** Requires training a separate noise-aware classifier $p_t(c \mid x_t)$ for each noise level $t$ — expensive for open-ended conditioning (e.g., text prompts).

### 8.2 Classifier-Free Guidance (CFG)

**Key idea.** Eliminate the separate classifier by noting (via Bayes' rule) that:
$$\nabla\log p_t(c \mid x_t) = \nabla\log p_t(x_t \mid c) - \nabla\log p_t(x_t)$$

Substituting into the guided score:
$$\nabla\log p_t(x \mid c;\,\omega) = \nabla\log p_t(x) + \omega\Big[\nabla\log p_t(x \mid c) - \nabla\log p_t(x)\Big]$$

$$= (1-\omega)\,\nabla\log p_t(x) + \omega\,\nabla\log p_t(x \mid c)$$

**Training:** Train a single network $\varepsilon_\theta(x_t, t, c)$ that handles both conditional and unconditional generation. During training, randomly drop the condition (replace $c$ with a null token $\varnothing$) with some probability (typically 10–20%). The same network then approximates both $\varepsilon_\theta(x_t,t,c)$ and $\varepsilon_\theta(x_t,t,\varnothing) \approx$ unconditional noise.

**Inference:** Combine the two outputs with scale $\omega$:
$$\tilde{\varepsilon}_\theta(x_t, t, c) = (1-\omega)\,\varepsilon_\theta(x_t, t, \varnothing) + \omega\,\varepsilon_\theta(x_t, t, c)$$

In practice, $\omega \in [1.5, 7.5]$ gives a good trade-off between sample quality and diversity. CFG does not require a separate classifier and scales naturally to open-domain conditioning.

---

## 9. Flow Matching

### 9.1 Motivation

Score-based diffusion models train by matching the gradient of a density — an indirect objective. **Flow matching** (Lipman et al., 2022; Liu et al., 2022) takes a more direct approach: directly regress a **vector field** that moves mass from one distribution to another.

### 9.2 Setup and Continuity Equation

Let $p_0$ (data) and $p_1 = \mathcal{N}(0, I)$ (noise) be two distributions. Define a linear interpolation between samples $x_0 \sim p_0$ and $x_1 \sim p_1$:
$$x_t = (1-t)x_0 + tx_1, \quad t \in [0,1]$$

Let $p_t$ be the density of $x_t$. We want a vector field $v_t(x)$ such that the ODE $dx_t = v_t(x_t)\,dt$ has marginals $p_t$ — i.e., starting from $p_0$ and following $v_t$ for time 1 arrives at $p_1$.

By the **continuity equation** (Fokker-Planck without diffusion):
$$\frac{\partial}{\partial t}p_t(x) = -\text{div}(p_t(x)\,v_t(x))$$

### 9.3 Computing the Marginal Velocity

For any smooth test function $f$:

$$\frac{d}{dt}\mathbb{E}[f(x_t)] = \int\int p_0(x_0)p_1(x_1)\,\langle\nabla f(x_0 + t(x_1-x_0)),\, x_1-x_0\rangle\,dx_0\,dx_1$$

(differentiating through the definition $x_t = (1-t)x_0 + tx_1$). After a change of variables $x_1 \to x_t = (1-t)x_0 + tx_1$ (Jacobian $t^{-d}$), the density of $x_t$ is:
$$p_t(x) = \int p_0(x_0)\,p_1\!\left(\frac{x-(1-t)x_0}{t}\right)t^{-d}\,dx_0$$

Substituting and simplifying the expectation:
$$\frac{d}{dt}\mathbb{E}[f(x_t)] = \int\nabla f(x_t)\cdot p_t(x_t)\,\frac{x_t - \mathbb{E}[x_0 \mid x_t]}{t}\,dx_t$$

Comparing with $\frac{d}{dt}\mathbb{E}[f(x_t)] = \int\nabla f(x)\,p_t(x)\,v_t(x)\,dx$ (from the continuity equation), we identify:

$$\boxed{v_t(x) = \frac{x - \mathbb{E}[x_0 \mid x_t = x]}{t}}$$

**Verification:** Since $x_t = (1-t)x_0 + tx_1$, we have $x_1 - x_0 = \frac{x_t - x_0}{t}$, so $v_t = \mathbb{E}[x_1-x_0 \mid x_t=x]$ — the expected direction of travel for trajectories passing through $x$ at time $t$.

### 9.4 Training Objective

The conditional velocity given $(x_0, x_1)$ is simply $\dot{x}_t = x_1 - x_0$ (constant along each trajectory). The **flow matching loss** regresses the network $u_t(x;\theta)$ toward this conditional target:

$$L(\theta) = \mathbb{E}_{x_0,\,x_1,\,t}\!\left[\left\|u_t(x_t;\,\theta) - (x_1 - x_0)\right\|_2^2\right], \quad x_t = (1-t)x_0 + tx_1$$

This is **simulation-free**: no ODE integration is required during training — just sample $t, x_0, x_1$, form $x_t$, and evaluate the squared error.

At optimality: $u_t^*(x) = \mathbb{E}[x_1 - x_0 \mid x_t = x] = v_t(x)$.

### 9.5 Connection to Score Matching

Under the specific choice $x_1 = \epsilon \sim \mathcal{N}(0,I)$ (flow from data to noise), the marginal $p_t = p_0 * \mathcal{N}(0, t^2 I)$. Applying **Tweedie's formula** for the linear interpolation path $x_t = (1-t)x_0 + t\epsilon$:

Since $q_t(x \mid x_0) = \mathcal{N}((1-t)x_0, t^2 I)$, the marginal score is:
$$\nabla\log p_t(x) = \frac{(1-t)\mathbb{E}[x_0 \mid x_t=x] - x}{t^2} \;\Rightarrow\; \mathbb{E}[x_0 \mid x_t=x] = \frac{x + t^2\nabla\log p_t(x)}{1-t}$$

Substituting into $v_t(x) = \frac{x-\mathbb{E}[x_0|x_t=x]}{t}$:
$$v_t(x) = \frac{x}{t} - \frac{1}{t}\cdot\frac{x+t^2\nabla\log p_t(x)}{1-t} = \frac{x(1-t)-x-t^2\nabla\log p_t(x)}{t(1-t)} = -\frac{x}{1-t} - \frac{t}{1-t}\nabla\log p_t(x)$$

For comparison, under the VE path $y_s = y_0 + s\epsilon$ (noise schedule $\sigma=s$), the probability flow ODE is:
$$dy_s = \frac{y_s - \mathbb{E}[y_0 \mid y_s]}{s}\,ds = -s\nabla\log q_s(y_s)\,ds$$

The two formulations are related by the time-variance mapping $\sigma = \frac{t}{1-t}$ (i.e., $t = \frac{\sigma}{1+\sigma}$). Both describe the same physics — moving mass from data to noise — but with different time parameterisations. The **velocity field and the score function are two sides of the same coin**, connected by the denoising expectation $\mathbb{E}[x_0 \mid x_t]$.

---

## 10. Summary

```
DATA (x₀) ──[forward SDE: add noise]──> NOISE (x_T ~ N(0,σ²I))

KEY QUANTITY: score  ∇log p_t(x) = (E[x₀|x_t=x] - x) / σ_t²

TRAINING:   fit ε_θ(x_t, t) ≈ ε  via  E[‖ε_θ + ε‖²]  (DDPM loss)

SAMPLING:
  Stochastic (DDPM):   x_{t-1} = denoise(x_t) + noise
  Deterministic (DDIM): x_{t-1} = denoise(x_t)           [fewer steps]
  Guided (CFG):         ε̃ = (1-ω)ε_unc + ω·ε_cond

FLOW MATCHING (alternative): regress v_θ(x_t,t) ≈ x₁ - x₀
  – simulation-free, straight trajectories, fewer NFE
```

---

## 11. References

1. **Hyvärinen, A.** (2005). Estimation of non-normalized statistical models by score matching. *JMLR*, 6, 695–709. [Foundational score matching paper.]
2. **Vincent, P.** (2011). A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7), 1661–1674. [Denoising score matching ≡ score matching.]
3. **Song, Y. & Ermon, S.** (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS 2019*. [NCSN / noise-conditional score networks.]
4. **Ho, J., Jain, A., & Abbeel, P.** (2020). Denoising diffusion probabilistic models (DDPM). *NeurIPS 2020*. [DDPM: simple $\epsilon$-prediction objective.]
5. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B.** (2021). Score-based generative modeling through stochastic differential equations. *ICLR 2021*. [Unified SDE framework: VE/VP, probability flow ODE, reverse SDE.]
6. **Song, J., Meng, C., & Ermon, S.** (2021). Denoising diffusion implicit models (DDIM). *ICLR 2021*. [Deterministic sampler; drastically fewer NFE.]
7. **Dhariwal, P. & Nichol, A.** (2022). Diffusion models beat GANs on image synthesis. *NeurIPS 2021*. [Classifier guidance.]
8. **Ho, J. & Salimans, T.** (2022). Classifier-free diffusion guidance. *NeurIPS 2021 Workshop*. [CFG.]
9. **Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M.** (2022). Flow matching for generative modeling. *ICLR 2023*. [Flow matching framework.]
10. **Liu, X., Gong, C., & Liu, Q.** (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR 2023*. [Rectified flow / linear interpolation.]
11. **Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T.** (2022). Elucidating the design space of diffusion-based generative models. *NeurIPS 2022*. [EDM: clean unification of DDPM/DDIM/VE/VP with Heun's 2nd-order sampler.]

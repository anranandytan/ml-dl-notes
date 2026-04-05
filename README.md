# Machine Learning & Deep Learning Notes

A comprehensive set of notes covering probabilistic machine learning, deep generative models, and classical machine learning methods. All derivations are step-by-step with historical context and verified references.

---

## Part I — Deep Learning & Probabilistic Models

*Based on graduate-level deep learning course notes.*

### I-A: Bayesian Methods and Inference

| File | Topic | Key Concepts |
|------|-------|--------------|
| `DL_01_BayesianRegression.md` | Bayesian Regression | MAP, conjugate prior, posterior, predictive distribution, kernel trick |
| `DL_02_GaussianProcess.md` | Gaussian Process | Weight-space view, function-space view, GP regression, marginal likelihood |
| `DL_03_GaussianGraphicalModels.md` | Gaussian Graphical Models | Precision matrix, conditional independence, Gaussian BN, Gaussian MRF |
| `DL_04_EM.md` | EM Algorithm | ELBO, E-step, M-step, monotonic convergence, GMM example |
| `DL_05_SGVI.md` | Variational Inference & SGVI | Mean-field, CAVI, score function estimator, reparameterisation |
| `DL_06_MCMC.md` | MCMC | Metropolis-Hastings, Gibbs sampling, detailed balance, burn-in |

### I-B: Sequential Models

| File | Topic | Key Concepts |
|------|-------|--------------|
| `DL_07_HMM.md` | Hidden Markov Model | Forward-backward, Baum-Welch, Viterbi, filtering, smoothing |
| `DL_08_KalmanFilter.md` | Kalman Filter | Linear Gaussian SSM, predict-update, Kalman gain, RTS smoother |
| `DL_09_ParticleFilter.md` | Particle Filter | SIS, bootstrap filter, weight degeneracy, resampling |

### I-C: Generative Models

| File | Topic | Key Concepts |
|------|-------|--------------|
| `DL_10_GenerativeModels.md` | Generative Models (Overview) | Taxonomy, explicit vs. implicit density, reparameterisation trick |
| `DL_11_RBM.md` | Boltzmann Machine & RBM | Energy function, CD learning, DBM, ELBO |
| `DL_12_VAE.md` | Variational Autoencoder (VAE) | ELBO, encoder/decoder, KL regularisation, latent space |
| `DL_13_GAN.md` | GAN | Minimax game, optimal discriminator, Jensen-Shannon divergence |
| `DL_14_Diffusion.md` | Diffusion Models | Score matching, SDE/ODE, DDPM, DDIM, CFG, flow matching |

### I-D: Structured Prediction & Graph-Based Methods

| File | Topic | Key Concepts |
|------|-------|--------------|
| `DL_15_CRF.md` | Conditional Random Field | Label bias, MRF, feature functions, forward-backward, Viterbi |
| `DL_16_SpectralClustering.md` | Spectral Clustering | Graph Laplacian, normalised cut, Rayleigh quotient, eigenvectors |

---

## Part II — Classical Machine Learning

*Based on undergraduate/graduate machine learning course notes.*

### II-A: Foundations

| File | Topic | Key Concepts |
|------|-------|--------------|
| `ML_01_Fundamentals.md` | Fundamentals | Problem taxonomy, loss functions, risk, ROC/AUC, cross-validation |

### II-B: Tree Methods and Ensembles

| File | Topic | Key Concepts |
|------|-------|--------------|
| `ML_02_DecisionTrees.md` | Decision Trees | Information gain, Gini, C4.5, CART, GOSDT, branch-and-bound |
| `ML_03_Ensembles.md` | Ensembles, Boosting & GAMs | Random forest, MR/CMR/SHAP/MCR, AdaBoost, GAM, FastSparse |

### II-C: Neural Networks

| File | Topic | Key Concepts |
|------|-------|--------------|
| `ML_04_NeuralNetworks.md` | Neural Networks | Backpropagation, cross-entropy, CNN, ResNet, dropout, contrastive learning |

### II-D: Linear Methods and Optimisation

| File | Topic | Key Concepts |
|------|-------|--------------|
| `ML_05_LinearMethods.md` | Logistic Regression, Convex Opt., SVM | Logit model, MLE, KKT, strong duality, support vectors, SMO |

### II-E: Kernel Methods

| File | Topic | Key Concepts |
|------|-------|--------------|
| `ML_06_Kernels.md` | Kernel Methods | Polynomial/RBF kernels, RKHS, Mercer, Representer theorem, kernel ridge/regression |
| `ML_07_Regression.md` | Regression | OLS, Ridge ($\ell_2$), Lasso ($\ell_1$), Bayesian interpretation |

### II-F: Learning Theory

| File | Topic | Key Concepts |
|------|-------|--------------|
| `ML_08_LearningTheory.md` | Statistical Learning Theory | Hoeffding, Ockham's Razor bound, growth function, VC dimension |

### II-G: Unsupervised Methods

| File | Topic | Key Concepts |
|------|-------|--------------|
| `ML_09_Clustering.md` | Clustering | K-means (coordinate descent), hierarchical agglomerative, linkage |

### II-H: Modern Methods

| File | Topic | Key Concepts |
|------|-------|--------------|
| `ML_10_Transformers.md` | Transformers, RLHF & Ethics | Self-attention, MHA, positional encoding, RLHF, PPO, AI ethics |

---

## Recommended Reading Order

### For Probabilistic ML (Part I)
```
DL_01 → DL_02 → DL_04 → DL_05 → DL_06     (Bayesian chain)
DL_07 → DL_08 → DL_09                       (Sequential models)
DL_10 → DL_11 → DL_12 → DL_13 → DL_14     (Generative models)
DL_03 → DL_15 → DL_16                       (Graphical models)
```

### For Classical ML (Part II)
```
ML_01 → ML_02 → ML_03                       (Supervised learning foundations)
ML_05 → ML_06 → ML_07                       (Linear and kernel methods)
ML_04 → ML_10                               (Neural networks & transformers)
ML_08                                        (Theory)
ML_09 + DL_16                               (Clustering)
```

### Cross-Part Connections
```
DL_01 (Bayesian Regression) ←→ ML_07 (Ridge/Lasso)      [MAP = regularisation]
DL_02 (GP) ←→ ML_06 (Kernel methods)                    [kernel trick / RKHS]
DL_04 (EM) ←→ DL_11 (RBM) ←→ DL_12 (VAE)              [latent variable models]
DL_12 (VAE) ←→ DL_05 (SGVI)                             [amortised inference]
DL_13 (GAN) ←→ ML_04 (Neural networks)                  [adversarial training]
DL_16 (Spectral) ←→ ML_09 (Clustering)                  [unsupervised methods]
DL_15 (CRF) ←→ DL_07 (HMM)                             [sequence labelling]
```

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $p(x)$, $p_\theta(x)$ | Probability density / mass function |
| $q(z)$, $q_\phi(z\|x)$ | Variational / approximate posterior |
| $\mathcal{L}(q)$ | ELBO (Evidence Lower Bound) |
| $D_{\text{KL}}(P\|Q)$ | KL divergence from $Q$ to $P$ |
| $\mathcal{N}(\mu,\Sigma)$ | Multivariate Gaussian |
| $\sigma(x)$ | Sigmoid function $1/(1+e^{-x})$ |
| $R^{\text{true}}(f)$ | True (test) risk |
| $R^{\text{emp}}(f)$ | Empirical (training) risk |
| $h = \text{VC}(\mathcal{F})$ | VC dimension of function class $\mathcal{F}$ |
| $L = D-W$ | Graph Laplacian |
| $\Lambda = \Sigma^{-1}$ | Precision matrix |
| $\alpha_t(i)$, $\beta_t(i)$ | Forward/backward variables (HMM) |
| $K_t$ | Kalman gain |
| $w_t^{(i)}$, $\hat{w}_t^{(i)}$ | Particle weights (unnormalised/normalised) |
| $Q,K,V$ | Query, Key, Value matrices (Transformer) |

---

## Core References

### Textbooks

- **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
- **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. [[Free](https://www.deeplearningbook.org)]
- **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [[Free PDF](https://hastie.su.domains/ElemStatLearn/)]
- **Rasmussen, C. E. & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press. [[Free PDF](http://www.gaussianprocess.org/gpml/)]
- **Koller, D. & Friedman, N.** (2009). *Probabilistic Graphical Models*. MIT Press.
- **Shalev-Shwartz, S. & Ben-David, S.** (2014). *Understanding Machine Learning*. Cambridge. [[Free PDF](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)]
- **Boyd, S. & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge. [[Free PDF](https://stanford.edu/~boyd/cvxbook/)]

---

## Complete File Index

```
DL_01_BayesianRegression.md          ML_01_Fundamentals.md
DL_02_GaussianProcess.md             ML_02_DecisionTrees.md
DL_03_GaussianGraphicalModels.md     ML_03_Ensembles.md
DL_04_EM.md                          ML_04_NeuralNetworks.md
DL_05_SGVI.md                        ML_05_LinearMethods.md
DL_06_MCMC.md                        ML_06_Kernels.md
DL_07_HMM.md                         ML_07_Regression.md
DL_08_KalmanFilter.md                ML_08_LearningTheory.md
DL_09_ParticleFilter.md              ML_09_Clustering.md
DL_10_GenerativeModels.md            ML_10_Transformers.md
DL_11_RBM.md
DL_12_VAE.md
DL_13_GAN.md
DL_14_Diffusion.md
DL_15_CRF.md
DL_16_SpectralClustering.md

README.md  (this file)
```

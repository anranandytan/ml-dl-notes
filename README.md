# Machine Learning & Deep Learning Notes

Personal course notes covering probabilistic machine learning, deep generative models, and classical machine learning. All derivations are step-by-step, with historical context and references to primary sources.

---

## Contents

### Part I — Deep Learning & Probabilistic Models

#### I-A: Bayesian Methods and Inference

| File | Topic | Key Concepts |
|------|-------|--------------|
| [DL_01_BayesianRegression.md](DL_01_BayesianRegression.md) | Bayesian Regression | MAP, conjugate prior, posterior, marginal likelihood, kernel trick |
| [DL_02_GaussianProcess.md](DL_02_GaussianProcess.md) | Gaussian Process | Weight-space view, GP regression, Mercer kernel, marginal likelihood |
| [DL_03_GaussianGraphicalModels.md](DL_03_GaussianGraphicalModels.md) | Gaussian Graphical Models | Precision matrix, conditional independence, BN, MRF, graphical Lasso |
| [DL_04_EM.md](DL_04_EM.md) | EM Algorithm | ELBO, E-step, M-step, monotone convergence, Fisher's missing information |
| [DL_05_SGVI.md](DL_05_SGVI.md) | Variational Inference | CAVI, score function estimator, control variates, reparameterisation trick |
| [DL_06_MCMC.md](DL_06_MCMC.md) | MCMC | Metropolis-Hastings, Gibbs sampling, detailed balance, HMC, NUTS |

#### I-B: Sequential Models

| File | Topic | Key Concepts |
|------|-------|--------------|
| [DL_07_HMM.md](DL_07_HMM.md) | Hidden Markov Model | Forward-backward, Baum-Welch, Viterbi, log-space scaling, multiple sequences |
| [DL_08_KalmanFilter.md](DL_08_KalmanFilter.md) | Kalman Filter | Linear Gaussian SSM, predict-update, Kalman gain, RTS smoother |
| [DL_09_ParticleFilter.md](DL_09_ParticleFilter.md) | Particle Filter | SIS, bootstrap filter, ESS, resampling, auxiliary particle filter |

#### I-C: Generative Models

| File | Topic | Key Concepts |
|------|-------|--------------|
| [DL_10_GenerativeModels.md](DL_10_GenerativeModels.md) | Generative Models (Overview) | Taxonomy, explicit vs. implicit density, reparameterisation trick |
| [DL_11_RBM.md](DL_11_RBM.md) | Boltzmann Machine & RBM | Energy function, conditional independence, CD, persistent CD, DBN |
| [DL_12_VAE.md](DL_12_VAE.md) | VAE | ELBO, encoder/decoder, reparameterisation, posterior collapse, β-VAE |
| [DL_13_GAN.md](DL_13_GAN.md) | GAN | Minimax game, optimal discriminator, JSD, mode collapse, WGAN |
| [DL_14_Diffusion.md](DL_14_Diffusion.md) | Diffusion Models | Score matching, VE/VP-SDE, DDPM, DDIM, classifier-free guidance |

#### I-D: Structured Prediction & Graph Methods

| File | Topic | Key Concepts |
|------|-------|--------------|
| [DL_15_CRF.md](DL_15_CRF.md) | Conditional Random Field | Label bias, log-linear model, forward-backward, Viterbi, vs. HMM |
| [DL_16_SpectralClustering.md](DL_16_SpectralClustering.md) | Spectral Clustering | Graph Laplacian, normalised cut, random-walk Laplacian, eigengap heuristic |

---

### Part II — Classical Machine Learning

#### II-A: Foundations

| File | Topic | Key Concepts |
|------|-------|--------------|
| [ML_01_Fundamentals.md](ML_01_Fundamentals.md) | Fundamentals | Problem taxonomy, bias-variance tradeoff, loss functions, ROC/PR curves, CV |

#### II-B: Tree Methods and Ensembles

| File | Topic | Key Concepts |
|------|-------|--------------|
| [ML_02_DecisionTrees.md](ML_02_DecisionTrees.md) | Decision Trees | Entropy/Gini, C4.5, CART, GOSDT, branch-and-bound |
| [ML_03_Ensembles.md](ML_03_Ensembles.md) | Ensembles, Boosting & GAMs | Random forest, MR/CMR/SHAP/MCR, AdaBoost, gradient boosting, GAM/FastSparse |

#### II-C: Neural Networks

| File | Topic | Key Concepts |
|------|-------|--------------|
| [ML_04_NeuralNetworks.md](ML_04_NeuralNetworks.md) | Neural Networks | Backprop, batch norm, Adam, ResNet, dropout, contrastive learning |

#### II-D: Linear Methods and Optimisation

| File | Topic | Key Concepts |
|------|-------|--------------|
| [ML_05_LinearMethods.md](ML_05_LinearMethods.md) | Logistic Regression, Convex Opt., SVM | Logit, MLE, KKT, kernel SVM, SMO |

#### II-E: Kernel Methods

| File | Topic | Key Concepts |
|------|-------|--------------|
| [ML_06_Kernels.md](ML_06_Kernels.md) | Kernel Methods | RKHS, Mercer, representer theorem, kernel ridge regression |
| [ML_07_Regression.md](ML_07_Regression.md) | Regression | OLS, Ridge, Lasso, elastic net, effective degrees of freedom |

#### II-F: Learning Theory

| File | Topic | Key Concepts |
|------|-------|--------------|
| [ML_08_LearningTheory.md](ML_08_LearningTheory.md) | Statistical Learning Theory | Hoeffding, Ockham bound, PAC learning, VC dimension |

#### II-G: Unsupervised Methods

| File | Topic | Key Concepts |
|------|-------|--------------|
| [ML_09_Clustering.md](ML_09_Clustering.md) | Clustering | K-means, K-means++, DBSCAN, hierarchical agglomerative |

#### II-H: Modern Methods

| File | Topic | Key Concepts |
|------|-------|--------------|
| [ML_10_Transformers.md](ML_10_Transformers.md) | Transformers, RLHF & Ethics | Self-attention, positional encoding, RLHF, DPO, AI ethics |

---

## Recommended Reading Order

**Probabilistic ML (Part I)**
```
DL_01 → DL_02 → DL_04 → DL_05 → DL_06     Bayesian chain
DL_07 → DL_08 → DL_09                       Sequential models
DL_10 → DL_11 → DL_12 → DL_13 → DL_14     Generative models
DL_03 → DL_15 → DL_16                       Graphical models
```

**Classical ML (Part II)**
```
ML_01 → ML_02 → ML_03                       Supervised learning
ML_05 → ML_06 → ML_07                       Linear and kernel methods
ML_04 → ML_10                               Neural networks
ML_08                                        Theory
ML_09 + DL_16                               Clustering
```

**Cross-Part Connections**
```
DL_01 ↔ ML_07    MAP estimation = regularisation
DL_02 ↔ ML_06    Kernel trick / RKHS
DL_04 ↔ DL_12    EM → variational EM → VAE
DL_05 ↔ DL_12    SGVI = VAE training
DL_13 ↔ ML_04    Adversarial training
DL_16 ↔ ML_09    Spectral vs. compactness clustering
DL_15 ↔ DL_07    CRF generalises HMM (global normalisation)
```

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $p(x)$, $p_\theta(x)$ | Probability density / mass |
| $q(z)$, $q_\phi(z\|x)$ | Variational / approximate posterior |
| $\mathcal{L}(q)$ | ELBO (Evidence Lower Bound) |
| $D_{\mathrm{KL}}(P\|Q)$ | KL divergence |
| $\mathcal{N}(\mu,\Sigma)$ | Multivariate Gaussian |
| $\sigma(x)$ | Sigmoid $1/(1+e^{-x})$ |
| $R^{\mathrm{true}}(f)$ | True (test) risk |
| $R^{\mathrm{emp}}(f)$ | Empirical (training) risk |
| $h = \mathrm{VC}(\mathcal{F})$ | VC dimension |
| $L = D - W$ | Graph Laplacian |
| $\Lambda = \Sigma^{-1}$ | Precision matrix |
| $\alpha_t(i)$, $\beta_t(i)$ | HMM forward/backward variables |
| $K_t$ | Kalman gain |
| $\hat{w}_t^{(i)}$ | Normalised particle weight |
| $Q$, $K$, $V$ | Transformer Query/Key/Value matrices |

---

## Core References

- **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
- **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. [[Free](https://www.deeplearningbook.org)]
- **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [[Free PDF](https://hastie.su.domains/ElemStatLearn/)]
- **Rasmussen, C. E. & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press. [[Free PDF](http://www.gaussianprocess.org/gpml/)]
- **Shalev-Shwartz, S. & Ben-David, S.** (2014). *Understanding Machine Learning*. Cambridge. [[Free PDF](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)]
- **Boyd, S. & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge. [[Free PDF](https://stanford.edu/~boyd/cvxbook/)]

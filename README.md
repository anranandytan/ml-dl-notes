# Machine Learning & Deep Learning Notes

Personal course notes on machine learning and probabilistic deep learning. Files are numbered by knowledge dependency — each topic builds on those before it. All derivations are step-by-step with references to primary sources.

---

## Contents

| # | File | Topic | Key Concepts |
|---|------|-------|--------------|
| 01 | [01_Fundamentals.md](01_Fundamentals.md) | ML Fundamentals | Problem taxonomy, loss functions, bias-variance, ROC/PR curves, cross-validation |
| 02 | [02_LearningTheory.md](02_LearningTheory.md) | Statistical Learning Theory | Hoeffding, Ockham bound, PAC learning, VC dimension, growth function |
| 03 | [03_Regression.md](03_Regression.md) | Regression | OLS, Ridge, Lasso, elastic net, effective degrees of freedom |
| 04 | [04_BayesianRegression.md](04_BayesianRegression.md) | Bayesian Regression | Conjugate prior, posterior, marginal likelihood, kernel trick |
| 05 | [05_LinearMethods.md](05_LinearMethods.md) | Logistic Regression, Convex Opt., SVM | MLE, KKT, strong duality, support vectors, kernel SVM, SMO |
| 06 | [06_Kernels.md](06_Kernels.md) | Kernel Methods | RKHS, Mercer's theorem, representer theorem, kernel ridge regression |
| 07 | [07_GaussianProcess.md](07_GaussianProcess.md) | Gaussian Process | Weight-space vs. function-space, GP regression, marginal likelihood |
| 08 | [08_DecisionTrees.md](08_DecisionTrees.md) | Decision Trees | Entropy/Gini, C4.5, CART, GOSDT, branch-and-bound |
| 09 | [09_Ensembles.md](09_Ensembles.md) | Ensembles, Boosting & GAMs | Random forest, AdaBoost, gradient boosting, MR/CMR/SHAP/MCR |
| 10 | [10_NeuralNetworks.md](10_NeuralNetworks.md) | Neural Networks | Backprop, batch norm, Adam, ResNet, dropout, contrastive learning |
| 11 | [11_GraphicalModels.md](11_GraphicalModels.md) | Gaussian Graphical Models | Precision matrix, conditional independence, MRF, graphical Lasso |
| 12 | [12_EM.md](12_EM.md) | EM Algorithm | ELBO, E/M-step, monotone convergence, Fisher's missing information |
| 13 | [13_VariationalInference.md](13_VariationalInference.md) | Variational Inference | CAVI, score function estimator, control variates, reparameterisation |
| 14 | [14_MCMC.md](14_MCMC.md) | MCMC | Metropolis-Hastings, Gibbs, detailed balance, HMC, NUTS |
| 15 | [15_HMM.md](15_HMM.md) | Hidden Markov Model | Forward-backward, Baum-Welch, Viterbi, log-space scaling |
| 16 | [16_KalmanFilter.md](16_KalmanFilter.md) | Kalman Filter | Linear Gaussian SSM, predict-update, Kalman gain, RTS smoother |
| 17 | [17_ParticleFilter.md](17_ParticleFilter.md) | Particle Filter | SIS, bootstrap filter, ESS, resampling |
| 18 | [18_CRF.md](18_CRF.md) | Conditional Random Field | Log-linear model, forward-backward, Viterbi, vs. HMM |
| 19 | [19_GenerativeModels.md](19_GenerativeModels.md) | Generative Models (Overview) | Taxonomy: explicit/implicit, directed/undirected, reparameterisation |
| 20 | [20_RBM.md](20_RBM.md) | Boltzmann Machine & RBM | Energy function, conditional independence, contrastive divergence |
| 21 | [21_VAE.md](21_VAE.md) | Variational Autoencoder | ELBO, encoder/decoder, reparameterisation, posterior collapse |
| 22 | [22_GAN.md](22_GAN.md) | GAN | Minimax game, optimal discriminator, Jensen-Shannon divergence, WGAN |
| 23 | [23_Diffusion.md](23_Diffusion.md) | Diffusion Models | Score matching, VE/VP-SDE, DDPM, DDIM, classifier-free guidance |
| 24 | [24_Clustering.md](24_Clustering.md) | Clustering | K-means, K-means++, DBSCAN, hierarchical agglomerative |
| 25 | [25_SpectralClustering.md](25_SpectralClustering.md) | Spectral Clustering | Graph Laplacian, normalised cut, random-walk Laplacian, eigengap |
| 26 | [26_Transformers.md](26_Transformers.md) | Transformers, RLHF & Ethics | Self-attention, positional encoding, RLHF, DPO |

---

## Dependency Map

```
01 Fundamentals ──► 02 Learning Theory          core framework
         │
         ▼
03 Regression ──────────────────────────────────────────────┐
         │                                                   │
         ▼                                                   ▼
04 Bayesian Regression ──► 07 Gaussian Process    (Bayesian view of 03 and 06)
         │
         ▼
05 Linear Methods ──► 06 Kernels ──► 07 Gaussian Process

08 Decision Trees ──► 09 Ensembles
                                       └──► 10 Neural Networks ──► 26 Transformers

11 Graphical Models
         │
         ├──► 12 EM ──┬──► 13 Variational Inference ──► 21 VAE
         │            └──► 14 MCMC
         │
         └──► 15 HMM ──► 16 Kalman Filter ──► 17 Particle Filter
                    └──► 18 CRF

19 Generative Models (Overview)
         ├──► 20 RBM
         ├──► 21 VAE       (needs 12, 13)
         ├──► 22 GAN       (needs 10)
         └──► 23 Diffusion (needs 13)

24 Clustering ──► 25 Spectral Clustering
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
| $Q$, $K$, $V$ | Transformer Query/Key/Value |

---

## Core Textbooks

- **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
- **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. [[Free](https://www.deeplearningbook.org)]
- **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [[Free PDF](https://hastie.su.domains/ElemStatLearn/)]
- **Rasmussen, C. E. & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press. [[Free PDF](http://www.gaussianprocess.org/gpml/)]
- **Shalev-Shwartz, S. & Ben-David, S.** (2014). *Understanding Machine Learning*. Cambridge. [[Free PDF](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)]
- **Boyd, S. & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge. [[Free PDF](https://stanford.edu/~boyd/cvxbook/)]

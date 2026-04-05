# ML & DL Notes

Personal course notes on machine learning and probabilistic deep learning.
Files are ordered by knowledge dependency — each topic builds on those before it.
All derivations are step-by-step with references to primary sources.

---

## Contents

| # | Topic | Key Concepts |
|---|-------|--------------|
| [01](01_Fundamentals.md) | ML Fundamentals | Problem taxonomy, loss functions, bias-variance, ROC/PR curves, cross-validation |
| [02](02_LearningTheory.md) | Statistical Learning Theory | Hoeffding, Ockham bound, PAC learning, VC dimension |
| [03](03_Regression.md) | Regression | OLS, Ridge, Lasso, elastic net, effective degrees of freedom |
| [04](04_BayesianRegression.md) | Bayesian Regression | Conjugate prior, posterior, marginal likelihood, kernel trick |
| [05](05_LinearMethods.md) | Logistic Regression, Convex Opt., SVM | MLE, KKT, strong duality, support vectors, kernel SVM |
| [06](06_Kernels.md) | Kernel Methods | RKHS, Mercer's theorem, representer theorem, kernel ridge regression |
| [07](07_GaussianProcess.md) | Gaussian Process | Weight-space vs. function-space, GP regression, marginal likelihood |
| [08](08_DecisionTrees.md) | Decision Trees | Entropy/Gini, C4.5, CART, GOSDT, branch-and-bound |
| [09](09_Ensembles.md) | Ensembles, Boosting & GAMs | Random forest, AdaBoost, gradient boosting, MR/CMR/SHAP/MCR |
| [10](10_NeuralNetworks.md) | Neural Networks | Backprop, batch norm, Adam, ResNet, dropout, contrastive learning |
| [11](11_GraphicalModels.md) | Gaussian Graphical Models | Precision matrix, conditional independence, MRF, graphical Lasso |
| [12](12_EM.md) | EM Algorithm | ELBO, E/M-step, monotone convergence, Fisher's missing information |
| [13](13_VariationalInference.md) | Variational Inference | CAVI, score function estimator, control variates, reparameterisation |
| [14](14_MCMC.md) | Markov Chain Monte Carlo | Metropolis-Hastings, Gibbs, detailed balance, HMC, NUTS |
| [15](15_HMM.md) | Hidden Markov Model | Forward-backward, Baum-Welch, Viterbi, log-space scaling |
| [16](16_KalmanFilter.md) | Kalman Filter | Linear Gaussian SSM, predict-update, Kalman gain, RTS smoother |
| [17](17_ParticleFilter.md) | Particle Filter | SIS, bootstrap filter, ESS, resampling |
| [18](18_CRF.md) | Conditional Random Field | Log-linear model, forward-backward, Viterbi, vs. HMM |
| [19](19_GenerativeModels.md) | Generative Models | Taxonomy: explicit/implicit, directed/undirected, reparameterisation |
| [20](20_RBM.md) | Boltzmann Machine & RBM | Energy function, conditional independence, contrastive divergence |
| [21](21_VAE.md) | Variational Autoencoder | ELBO, encoder/decoder, reparameterisation, posterior collapse |
| [22](22_GAN.md) | GAN | Minimax game, optimal discriminator, Jensen-Shannon divergence, WGAN |
| [23](23_Diffusion.md) | Diffusion Models | Score matching, VE/VP-SDE, DDPM, DDIM, classifier-free guidance |
| [24](24_Clustering.md) | Clustering | K-means, K-means++, DBSCAN, hierarchical agglomerative |
| [25](25_SpectralClustering.md) | Spectral Clustering | Graph Laplacian, normalised cut, random-walk Laplacian, eigengap |
| [26](26_Transformers.md) | Transformers & RLHF | Self-attention, positional encoding, RLHF, DPO |

---

## Notation

| Symbol | Meaning |
|--------|---------|
| \(p(x)\), \(p_\theta(x)\) | Probability density / mass |
| \(q_\phi(z\mid x)\) | Variational / approximate posterior |
| \(\mathcal{L}(q)\) | ELBO (Evidence Lower Bound) |
| \(D_{\mathrm{KL}}(P\|Q)\) | KL divergence |
| \(R^{\mathrm{true}}(f)\) | True (test) risk |
| \(h = \mathrm{VC}(\mathcal{F})\) | VC dimension |
| \(L = D - W\) | Graph Laplacian |
| \(\Lambda = \Sigma^{-1}\) | Precision matrix |
| \(K_t\) | Kalman gain |
| \(Q, K, V\) | Transformer Query / Key / Value |

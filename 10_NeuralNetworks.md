# 10 — Neural Networks and CNNs

> **Keywords:** backpropagation, sigmoid, cross-entropy, softmax, CNN, ResNet, dropout, transfer learning, contrastive learning

---

## 1. Neural Networks and Backpropagation

### 1.1 Building Blocks

A feedforward neural network is a composition of parameterised layers. Each unit $b$ in a layer computes:

$$\text{net}_b = \sum_a w_{a,b}\,o_a, \qquad o_b = \phi(\text{net}_b)$$

where $o_a$ is the output of unit $a$, $w_{a,b}$ is the weight from $a$ to $b$, and $\phi$ is an **activation function**.

**Sigmoid activation:**

$$\phi(x) = \frac{1}{1+e^{-x}}, \qquad \phi'(x) = \phi(x)(1-\phi(x))$$

The sigmoid squashes any real number to $(0,1)$. Its derivative has a clean form in terms of itself — convenient for backpropagation.

### 1.2 Loss Functions

| Task | Loss |
|------|------|
| **Regression** | Mean squared error: $\frac{1}{n}\sum_i(y_i-\hat{y}_i)^2$ |
| **Binary classification** | Binary cross-entropy: $-\frac{1}{n}\sum_i[y_i\log\hat{y}_i+(1-y_i)\log(1-\hat{y}_i)]$ |
| **Multi-class classification** | Cross-entropy: $-\frac{1}{n}\sum_i\sum_c\mathbf{1}[y_i=c]\log q_c$ |

**Softmax** converts raw output scores $\mathbf{o}$ to a valid probability distribution:

$$q_c = \text{softmax}(\mathbf{o})_c = \frac{e^{o_c}}{\sum_k e^{o_k}}$$

**Cross-entropy for multi-class:**

$$E(\mathbf{y}_i) = -\sum_c \mathbf{1}[y_i=c]\log_2\,\text{softmax}(\mathbf{o})_c$$

**Cross-entropy for binary classification** (with $\hat{y}=\sigma(f(x))$):

$$H([y,1-y],[\hat{y},1-\hat{y}]) = -y\log\hat{y}-(1-y)\log(1-\hat{y})$$

$$= -y\log\frac{e^{f(x)}}{1+e^{f(x)}} - (1-y)\log\frac{1}{1+e^{f(x)}}$$

When $y=1$, this reduces to $\log(1+e^{-f(x)})$ — exactly the logistic loss.

### 1.3 Backpropagation

Backpropagation computes the gradient of the loss $E$ with respect to every weight by applying the chain rule layer by layer.

For a weight $w_{a,b}$ connecting unit $a$ to unit $b$, with $L$ the set of units in the next layer:

$$\frac{\partial E}{\partial w_{a,b}} = \frac{\partial E}{\partial o_b}\cdot\frac{\partial o_b}{\partial \text{net}_b}\cdot\frac{\partial\,\text{net}_b}{\partial w_{a,b}} = \left(\sum_{l\in L}\delta_l\,w_{b,l}\right)\cdot o_b(1-o_b)\cdot o_a$$

where $\delta_l = \frac{\partial E}{\partial\,\text{net}_l}$ is the error signal at unit $l$.

**Weight update (gradient descent):**

$$w_{a,b} \leftarrow w_{a,b} - \alpha\,\frac{\partial E}{\partial w_{a,b}}$$

where $\alpha$ is the learning rate.

**Stochastic / mini-batch gradient descent** computes the gradient on a small random subset at each step — the standard in deep learning.

---

## 2. Convolutional Neural Networks (CNNs)

### 2.1 Why Not a Fully Connected Network for Images?

1. **Spatial structure is destroyed** when an image is flattened into a vector.
2. **Computational intractability**: a $224\times224$ image has $\sim 50\text{K}$ pixels; a fully connected layer would require $\sim 50\text{K}^2$ weights.

CNNs exploit the **local structure** of images: nearby pixels are more related than distant ones. They share weights across spatial positions via convolution.

### 2.2 Convolution Operation

A filter (kernel) of size $K\times K$ slides over the input, computing a weighted sum at each position. Given:
- Input: $W\times H$
- Filter: $K\times K$
- Padding: $P$ (zeros added around the border)
- Stride: $S$ (step size between filter positions)

The output dimensions are:

$$\text{Output} = \left(\left\lfloor\frac{W+2P-K}{S}\right\rfloor+1\right)\times\left(\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1\right)$$

**Padding** preserves spatial dimensions. **Stride** > 1 downsamples.

### 2.3 Residual Networks (ResNets)

**Problem:** as networks get deeper, gradients vanish during backpropagation — early layers receive negligible gradient signal.

**ResNet solution:** add a **skip connection** that adds the input directly to the output of a block:

$$H(x) = F(x) + x$$

where $F(x)$ is what the block learns and $x$ is the identity shortcut.

**Why this works:** the gradient of $H(x)$ with respect to $x$ is $\frac{\partial F}{\partial x} + 1$. The $+1$ ensures the gradient is always at least 1, preventing vanishing. ResNets can be trained with hundreds of layers.

### 2.4 Dropout

**Idea:** during each forward pass, randomly set each neuron's output weights to 0 with probability $p$ (the **dropout rate**). During testing, use all neurons.

**Effect:**
- Forces the network to develop **redundant representations** — no single neuron can be relied upon.
- Acts like training exponentially many different subnetworks and averaging them (similar to bagging).
- Reduces co-adaptation between neurons.

### 2.5 Data Augmentation

Artificially expand the training set by applying label-preserving transformations:
- Horizontal/vertical flips, rotations, crops.
- Contrast and brightness adjustments.
- Elastic distortions.

This makes the model invariant to these transformations without requiring new labelled data.

### 2.6 Transfer Learning

Use the weights learned on a large dataset (e.g., ImageNet) as a starting point for a new task:

1. Take a pretrained model (e.g., ResNet-50 trained on ImageNet).
2. Remove the final classification layer.
3. Add a new classification head for the target task.
4. **Freeze** early layers (they extract general features like edges and textures).
5. **Fine-tune** only the last few layers on the target dataset.

Transfer learning is extremely effective when the target dataset is small.

### 2.7 Warnings

- **Confounders:** CNNs can latch onto spurious correlations (e.g., predicting "wolf" based on snow background, not the animal).
- **Brittleness / adversarial attacks:** a single-pixel change can flip a prediction with high confidence.
- **Deep fakes:** GANs trained on faces can produce realistic but entirely fake images/video.

---

## 3. Self-Supervised Learning

### 3.1 Motivation

Labelling data is expensive. Self-supervised learning trains on **naturally occurring signals** in raw data — no human annotations required.

The learned representations can then be fine-tuned with small amounts of labelled data (few-shot learning).

### 3.2 Contrastive Learning (SimCLR)

**Core idea:**
- **Similar** samples (two augmentations of the same image) should be **close** in the latent space.
- **Dissimilar** samples (different images) should be **far apart** in the latent space.

**Training procedure:**

1. For each image $x$, create two augmented views $x^+$ and $x^{++}$ (crop, colour jitter, blur, etc.).
2. Encode both with a shared encoder $f_\theta$: $z = f_\theta(x)$.
3. Maximise agreement between $z^+$ and $z^{++}$ while minimising agreement with representations from other images in the batch.

**NT-Xent loss (InfoNCE)** for a batch of $N$ images:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N\log\frac{\exp(\text{sim}(z_i^+, z_i^{++})/\tau)}{\sum_{k\neq i}\exp(\text{sim}(z_i^+, z_k)/\tau)}$$

where $\text{sim}(\cdot,\cdot)$ is cosine similarity and $\tau$ is a temperature parameter.

**Caveats:**
- Performance scales strongly with batch size (larger batches provide more negative examples).
- Batch sizes that are too large introduce **false negatives** (two augmentations of the same underlying scene treated as negatives).

### 3.3 Augmentation Design

Augmentation choices must be domain-specific:
- **Vision:** crops, flips, colour distortion.
- **Audio/signals:** temporal vicinity, noise injection.
- **Text:** back-translation, synonym replacement.

**Principles:**
1. Augmentation should preserve the label invariance (two views should represent the same concept).
2. Stronger augmentations yield more robust representations.
3. Augmentation choice has a larger impact on performance than architecture.

### 3.4 Prototype Models

Add a **prototype layer** to a black-box model, forcing the network to do case-based reasoning: "this prediction looks like prototype $p_k$, which is like this training example." This bridges the gap between black-box accuracy and interpretability.

---

## 4. GANs (Brief Note)

**Generative Adversarial Networks** are actor-critic models:
- **Generator $G$:** produces fake samples from noise $z\sim p_z$.
- **Discriminator $D$:** distinguishes real from fake samples.

**Objective:**

$$\min_G\max_D V(D,G) = \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

For a detailed derivation of the GAN objective, optimal discriminator, and Jensen-Shannon divergence connection, see **DL_13_GAN.md**.

---

## 5. References

1. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P.** (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. [Original LeNet/CNN paper.]
2. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). Deep residual learning for image recognition. *CVPR 2016*. [ResNet; skip connections for very deep networks.]
3. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.** (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929–1958.
4. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.** (2020). A simple framework for contrastive learning of visual representations (SimCLR). *ICML 2020*. [Contrastive self-supervised learning.]
5. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. Ch. 6 (Feedforward networks), Ch. 9 (CNNs).
6. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J.** (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536. [Original backpropagation paper.]

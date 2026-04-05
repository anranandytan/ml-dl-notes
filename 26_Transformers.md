# 26 — Transformers and RLHF

> **Keywords:** attention, self-attention, multi-head attention, positional encoding, transformer, RLHF, reward model, PPO, AI ethics, privacy

---

## 1. Historical Context

| Year | Event |
|------|-------|
| 1980s | Recurrent Neural Networks (RNNs): shared weights, backpropagation through time (BPTT); slow sequential training; vanishing gradients. |
| 1995 | Hochreiter & Schmidhuber introduce **LSTMs** — designed to overcome vanishing gradients; dominant for sequential tasks until 2017. |
| 2010s | Attention mechanisms emerge: learn how much each token should attend to others; improved long-range dependency modelling; still slow. |
| 2017 | **Vaswani et al.** — "Attention is All You Need": the Transformer eliminates recurrence entirely; attention is now the sole mechanism. |
| 2018–present | GPT, BERT, LLaMA, GPT-4, Claude: increasingly large Transformer-based language models. |

---

## 2. Building Blocks of the Transformer

### 2.1 Word Embedding

Each token (word or subword) is mapped to a dense vector in $\mathbb{R}^{d_{\text{model}}}$. These embedding vectors are **learned parameters**.

### 2.2 Positional Encoding

Attention has no inherent notion of order — the same attention weights apply regardless of position. **Positional encodings** inject position information:

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the position in the sequence and $i$ is the dimension index.

**Why sinusoids?**
- Nearby positions have similar encodings (the model can learn relative distance).
- The encoding generalises to unseen sequence lengths (extrapolation).
- Values at most dimensions are close to 0 or 1, keeping them numerically stable.

### 2.3 Linear Projection

A linear layer $W^\top\in\mathbb{R}^{d_{\text{small}}\times d_{\text{model}}}$ projects the $d_{\text{model}}$-dimensional embeddings down to a smaller $d_{\text{small}}$-dimensional space before computing attention.

### 2.4 Self-Attention

Self-attention allows each token to aggregate information from all other tokens, weighting by relevance.

**Ingredients:** Three learned weight matrices transform the input into:
- **Query matrix** $Q\in\mathbb{R}^{n\times d_{\text{small}}}$: what each token is "looking for."
- **Key matrix** $K\in\mathbb{R}^{n\times d_{\text{small}}}$: what each token "offers."
- **Value matrix** $V\in\mathbb{R}^{n\times d_{\text{small}}}$: the content each token contributes.

**Attention filter:** $QK^\top\in\mathbb{R}^{n\times n}$ — entry $(i,j)$ measures compatibility between token $i$'s query and token $j$'s key.

**Scaled dot-product attention:**

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{small}}}}\right)V\in\mathbb{R}^{n\times d_{\text{small}}}$$

The scaling by $\sqrt{d_{\text{small}}}$ prevents the dot products from becoming too large in magnitude (which would push the softmax into a saturated regime with near-zero gradients).

### 2.5 Multi-Head Attention

Rather than computing attention once, run $h$ parallel attention heads, each with its own projection matrices:

$$\text{head}_\tau = \text{Attention}(QW_\tau^Q,\, KW_\tau^K,\, VW_\tau^V)$$

Concatenate and project:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$$

Different heads can learn to attend to different types of relationships (syntactic, semantic, positional) simultaneously.

### 2.6 Masked Attention (Causal / Decoder)

For autoregressive generation, each token should attend only to **previous** tokens (not future ones). Add a mask $M$ before the softmax:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{small}}}} + M\right)V$$

where $M_{ij}=-\infty$ if $j>i$ (future tokens are masked out) and $0$ otherwise.

### 2.7 Other Components

**Add & Norm (Residual + Layer Normalization):**
- **Residual connection:** adds the input to the output of each sub-layer, preventing vanishing gradients and preserving information: $\text{output} = \text{LayerNorm}(x + \text{sublayer}(x))$.
- **Layer normalization:** normalises activations within each layer for stable training.

**Feed-Forward Network (FFN):** A position-wise two-layer MLP applied independently to each token:

$$\text{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$$

The FFN adds nonlinearity and mixes information across feature dimensions — attention alone is a linear operation over the value vectors.

---

## 3. Transformer Variants

| Variant | Architecture | Use case |
|---|---|---|
| **Encoder-only** (BERT) | Bidirectional attention (each token sees all others) | Classification, NER, sentence embeddings |
| **Decoder-only** (GPT, Claude, LLaMA) | Causal (masked) attention | Text generation, language modelling |
| **Encoder-decoder** (original Transformer, T5) | Encoder: bidirectional; Decoder: causal + cross-attention to encoder | Translation, summarisation |

Most modern LLMs (GPT-4, Claude, LLaMA) are **decoder-only**:
- Simpler: prompts and new generation separated by a unique token.
- Encoder-only models are better when the entire input needs global processing (e.g., sentiment analysis).

---

## 4. Positional Encoding Variants

| Method | Mechanism | Properties |
|---|---|---|
| **Absolute** (sinusoidal) | Fixed formula: $\sin/\cos$ of position | Simple; models memorise trained positions; poor extrapolation |
| **Relative** | Learn attention bias based on relative distance $i-j$ | Better local context; computationally expensive (more parameters) |
| **Rotary (RoPE)** | Rotate query and key vectors by position angle | Both absolute and relative info; better extrapolation; no learned parameters; used by most LLMs |
| **Linear Bias (ALiBi)** | Subtract a linear penalty $\lvert i-j\rvert$ from attention scores | Easy extrapolation; no learned parameters; trades performance for context length |
| **NoPE** | No positional encoding | Decoder-only transformers can implicitly learn position from masked attention patterns |

---

## 5. Reinforcement Learning from Human Feedback (RLHF)

### 5.1 Motivation

A pretrained language model generates fluent text but may not follow instructions or align with human values. RLHF fine-tunes the model to produce outputs humans prefer.

### 5.2 Components

1. **Agent:** the language model (generates text).
2. **Environment:** the human (or a reward model trained to simulate humans).
3. **Reward signal:** human preference between two generated outputs.

### 5.3 Training Pipeline

1. **Collect preferences:** Show a human two model outputs for the same prompt; they indicate which they prefer. Collect many such triples $(x, \tau^+, \tau^-)$ where $\tau^+$ is preferred over $\tau^-$.

2. **Train a reward model:** Starting from the pretrained model (which has learned language representations), fine-tune it to predict human preferences. Loss:

$$\ell(\hat{r}) = -\sum_{x,\tau^+,\tau^-}\log P(\tau^+\succ\tau^-|x)$$

where $P(\tau^+\succ\tau^-|x) = \sigma(\hat{r}(x,\tau^+)-\hat{r}(x,\tau^-))$ (Bradley-Terry model).

3. **Fine-tune the language model** using the reward model's signal, via **Proximal Policy Optimisation (PPO)** or similar. The language model is updated to generate outputs the reward model scores highly.

### 5.4 Remarks

- RLHF originally used PPO (Schulman et al., 2017); newer methods include DPO (direct preference optimisation), RLHF without PPO.
- The quality of the reward model is critical — a misspecified reward model leads to **reward hacking** (the model finds loopholes to score high without genuine quality).

---

## 6. Ethics and Impact

### 6.1 Privacy and Data Concerns

- LLMs are trained on vast amounts of web data, often including personal information.
- Models can **memorise** and reproduce training data, including sensitive personal details.
- Conversations with commercial LLMs are typically logged and may be used for future training.
- Minimal regulation currently governs what data can be used to train models.

### 6.2 Copyright Infringement

- Web content is not necessarily free to use commercially.
- AI music generators (e.g., Suno) have been sued for training on copyrighted songs without permission or payment.
- Similar issues arise with visual art and written works.
- The legal framework is still evolving.

### 6.3 Academic and Research Integrity

- Reviewers using LLMs to generate reviews violates conference policies.
- LLMs can hallucinate citations, facts, and results — uncritical use in research is dangerous.
- Transparency in AI-assisted writing is an active ethical debate.

### 6.4 Limitations of Black-Box Models

Black-box models (deep neural networks, large transformers) are:
- **Hard to troubleshoot** during development.
- **Hard to audit** for bias and fairness.
- **Hard to explain** to stakeholders or regulators.
- **Brittle:** adversarial examples, distributional shift, and dataset biases can cause dramatic failures.

Interpretable models (decision trees, GAMs, sparse linear models) are preferred in high-stakes domains (medicine, criminal justice, finance) where accountability matters.

---

## 7. Summary

```
Transformer core:
  Input → Embedding + Positional Encoding
        → Multi-Head Self-Attention (scaled dot-product)
        → Add & Norm (residual connection)
        → Feed-Forward Network (per-token MLP)
        → Add & Norm
        → (repeat L times)
        → Output head

Attention(Q,K,V) = softmax(QKᵀ / √d) · V

Multi-head: run h attention heads in parallel, concatenate, project

RLHF:
  1. Collect human preferences: (prompt, preferred, rejected)
  2. Train reward model on preferences
  3. Fine-tune LM with PPO to maximise reward
```

---

## 8. References

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). Attention is all you need. *NeurIPS 2017*. [Original Transformer paper.]
2. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.** (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*. [Encoder-only Transformer; bidirectional pretraining.]
3. **Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I.** (2019). Language models are unsupervised multitask learners (GPT-2). OpenAI Blog. [Decoder-only scaling.]
4. **Ouyang, L., Wu, J., Jiang, X., Almeida, D., et al.** (2022). Training language models to follow instructions with human feedback (InstructGPT). *NeurIPS 2022*. [RLHF for LLM alignment.]
5. **Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y.** (2021). RoFormer: Enhanced transformer with rotary position embedding. *arXiv:2104.09864*. [RoPE positional encoding.]
6. **Press, O., Smith, N. A., & Lewis, M.** (2022). Train short, test long: Attention with linear biases enables input length extrapolation. *ICLR 2022*. [ALiBi positional encoding.]
7. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*. [PPO algorithm used in RLHF.]

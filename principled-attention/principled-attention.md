# Principled Attention:<br>Gated Logits, Learned Ground States, and Scale-Invariant Normalization for Transformer Attention

**Richard Vermillion**
{: class="author" }

## Abstract

Standard scaled dot-product attention makes three implicit assumptions that limit the expressiveness of transformer architectures: (1) all attention mass must be distributed across keys, even when no key is relevant; (2) a single bilinear score conflates semantic relevance with selection; and (3) attention weights dilute as sequence length grows, degrading signal fidelity in long contexts. We introduce *principled attention*, a strict generalization of standard attention that addresses each limitation through a minimal, orthogonal extension. A learned *ground state* $v_0$ with per-query bias $\gamma_i$ allows the model to explicitly attend to "nothing." A *gating mechanism* decomposes query-key interaction into separate semantic and gate projections, enabling suppression without demotion. A $\log(K)$ scaling term restores scale invariance across sequence lengths. Standard attention is recovered as the special case $\beta_i = 0$, $\gamma_i = -\infty$. We present a systematic ablation across variants of each component and demonstrate that principled attention improves [TBD].

---

## 1. Introduction

The scaled dot-product attention mechanism is the computational core of the transformer architecture. For a query $q_i$, keys $k_j$, and values $v_j$, standard attention computes:

$$
w_{ij} = \frac{e^{q_i \cdot k_j / \sqrt{d}}}{\sum_{j'} e^{q_i \cdot k_{j'} / \sqrt{d}}}, \qquad o_i = \sum_j w_{ij} v_j
$$

This formulation is elegant and effective, but it embeds several implicit design choices that are rarely examined. The softmax is forced to allocate all probability mass across the available keys. The single dot product must simultaneously encode semantic relevance and the decision to attend. And the normalization carries no awareness of how many keys participate, causing attention to dilute as sequences grow.

These are not merely theoretical concerns — they manifest as concrete failure modes in practice: attention heads that learn to "park" on separator tokens as a makeshift null operation, an inability to suppress semantically matched but contextually irrelevant keys, and degraded performance on long sequences even when the relevant information is present.

In this work, we make each of these implicit choices explicit and learnable. The result is *principled attention* — a mechanism that strictly generalizes standard attention while remaining simple to implement and compatible with efficient attention algorithms such as FlashAttention-style tiling.

---

## 2. Problems with Standard Attention

### 2.1 Forced Allocation and the Absence of a Ground State

The softmax function maps any vector of real-valued logits to a probability distribution that sums to one. In the context of attention, this means every query *must* distribute its full attention mass across the available keys. There is no mechanism to express "none of these keys are relevant to this query."

In practice, models learn to work around this limitation. Attention heads frequently allocate disproportionate weight to structurally predictable tokens — BOS tokens, separators, punctuation — not because these tokens carry useful information, but because they serve as a sink for attention that has nowhere meaningful to go. This behavior has been widely observed and is sometimes called "attention sinking."

The problem is that this workaround is *emergent* rather than *explicit*. The model must sacrifice representational capacity to learn sink behavior, and the sink tokens themselves accumulate spurious attention signal that can propagate through residual connections. A principled mechanism would provide a dedicated, learnable "default output" — a ground state — that queries can attend to when no key is relevant, without hijacking unrelated tokens.

### 2.2 Bilinear Entanglement of Relevance and Selection

Standard attention computes a single score $s_{ij} = q_i \cdot k_j$ that serves dual duty: it measures how semantically related a key is to a query, and it determines whether the key should be attended to. These are fundamentally different operations.

Consider a scenario where a query is semantically related to a key but should suppress it — for example, in a language model where the correct next token is *not* the most obvious semantic associate, or in a retrieval setting where a distractor key is topically relevant but factually incorrect. Standard attention has no mechanism to express "I recognize this key but choose not to attend to it." The only option is to learn query and key projections that somehow encode both relevance and selection in a single bilinear form, which forces the model to entangle two distinct computational roles in one set of parameters.

A more principled decomposition would separate the *semantic score* (how related is this key?) from a *gate* (should I attend to it?), allowing the model to suppress keys without distorting the semantic space.

### 2.3 Sequence Length Dilution

As the number of keys $K$ grows, the softmax denominator $\sum_j e^{s_{ij}}$ grows roughly in proportion, causing the attention weight on any individual key to shrink. For a query that is primarily interested in a small number of keys, increasing the sequence length dilutes the signal from those keys even when the added keys are irrelevant.
 
A natural objection is that the model could learn to push irrelevant logits to large negative values, preventing them from contributing to the denominator. But the geometry of high-dimensional spaces makes this essentially impossible. In a high-dimensional embedding space, the dot product between any two randomly oriented vectors concentrates strongly around zero. The *modal* dot product between a query and an irrelevant key is not a large negative number — it is approximately zero. Since $e^0 = 1$, each irrelevant key contributes roughly unit mass to the softmax denominator, and the denominator grows approximately as $K$.
 
To prevent this, the model would need to learn query and key projections that push the majority of keys into a negative-dot-product regime relative to each query. This amounts to sacrificing representational capacity to solve a normalization problem — the model must use its limited projection dimensions to enforce geometric separation rather than to encode semantic distinctions. In practice, this is an unreasonable demand, and the denominator grows with $K$.
 
This is analogous to the well-understood problem that motivated $1/\sqrt{d}$ scaling: without compensating for the dimensionality of the dot product, the variance of the logits grows with $d$, pushing the softmax into saturation. The $1/\sqrt{d}$ factor restores variance invariance across embedding dimensions. But there is no analogous correction for the *number of keys*.
 
As transformers are applied to increasingly long contexts — documents, codebases, multi-turn conversations — this dilution becomes a practical bottleneck. A principled mechanism would scale the logits to maintain consistent attention sharpness regardless of sequence length.

---

## 3. Principled Attention

We now present a unified mechanism that addresses each of the problems identified in §2. The design is guided by a principle of *minimal extension*: each component solves exactly one problem, the components are orthogonal, and standard attention is recovered as a special case.

### 3.1 Formulation

Let $q_i$ denote query $i$ and $k_j$ denote key $j$, with $i \in [1..Q]$ and $j \in [1..K]$. We decompose each query and key into a *semantic* component and a *gate* component:

- $q^s_i, k^s_j$ — semantic projections (used for relevance scoring)
- $q^g_i, k^g_j$ — gate projections (used for selective suppression)

The full mechanism is:

$$
\begin{aligned}
s_{ij} &= q^s_i \cdot k^s_j && \text{(semantic score)} \\
g_{ij} &= q^g_i \cdot k^g_j && \text{(gate score)} \\
a_{ij} &= \lambda_i s_{ij} - \beta_i \, \mathrm{softplus}(-g_{ij}) && \text{(gated logit)} \\
z_i &= \sum_{j=1}^{K} e^{\max(\gamma_i,\, a_{ij})} && \text{(normalizer)} \\
w_{ij} &= \frac{e^{a_{ij}}}{z_i} && \text{(attention weight)} \\
w_{i0} &= 1 - \sum_{j=1}^{K} w_{ij} && \text{(ground weight)} \\
o_i &= w_{i0}\, v_0 + \sum_{j=1}^{K} w_{ij}\, v_j && \text{(output)}
\end{aligned}
$$

where $\beta_i$ is a learned gating strength, $\gamma_i$ is a learned ground bias, $\lambda_i$ is a scaling factor, and $v_0$ is a learned ground value.

### 3.2 How Each Component Addresses a Problem

#### 3.2.1 Ground state $v_0$ and ground bias $\gamma_i$
Directly addressing §2.1, the normalizer $z_i$ uses $\max(\gamma_i, a_{ij})$ rather than $a_{ij}$ alone. This means that any key whose gated logit falls below the ground bias $\gamma_i$ has its contribution to the denominator inflated to $e^{\gamma_i}$, effectively redirecting that attention mass to the ground state. The ground weight $w_{i0} = 1 - \sum_j w_{ij}$ captures all the mass that real keys fail to claim. The ground value $v_0$ is a learned vector that represents the "default output" — what the attention head produces when nothing in the context is worth attending to. This replaces emergent sink behavior with an explicit, learnable mechanism.

#### 3.2.2 Gating via $\beta_i$ and $\mathrm{softplus}(-g_{ij})$

Addressing §2.2, the gate score $g_{ij}$ is computed from dedicated gate projections $q^g_i$ and $k^g_j$, separate from the semantic projections. The $\mathrm{softplus}(-g_{ij})$ term is always non-negative, so the gate can only *subtract* from the semantic score — it suppresses, never promotes. The strength of suppression is controlled by $\beta_i$, which can be learned per-query, per-head, or per-layer.
 
The behavior of the gate is best understood through three regimes. When $g_{ij}$ is large and positive (gate is *aligned*), $\mathrm{softplus}(-g_{ij}) \approx 0$ and the semantic score passes through unmodified. When $g_{ij}$ is large and negative (gate is *misaligned*), $\mathrm{softplus}(-g_{ij}) \approx |g_{ij}|$ and the suppression is strong, subtracting a large value from the logit. The critical middle case is $g_{ij} \approx 0$ — *gate indifference* — which is also the modal outcome in high-dimensional space, since the dot product between unrelated gate vectors concentrates around zero. At indifference, $\mathrm{softplus}(0) = \log(2)$, so the suppression term becomes $\beta_i \log(2)$.
 
This makes $\beta_i$ interpretable: it controls how much the model penalizes gate indifference relative to gate alignment. For $\beta_i = 1$, an indifferent key has its logit reduced by $\log(2) \approx 0.69$ compared to an aligned key — a modest but meaningful suppression. Larger $\beta_i$ makes the model more demanding of explicit gate alignment before letting a key through. In this sense, $\beta_i$ is not merely a "gating strength" but a knob that sets the threshold between "I don't recognize this key" and "I recognize and accept this key," allowing the model to express "semantically relevant but contextually suppressed" without distorting the semantic projection space.

#### 3.2.3 Sequence length scaling via $\lambda_i = \frac{\log(K)}{\sqrt{d}}$
Addressing §2.3, we multiply the usual scaling factor by $\log(K)$. This factor, applied to the semantic score, counteracts the growth of the softmax denominator as $K$ increases. As the number of keys grows, $\sum_j e^{s_{ij}}$ grows roughly as $K$ (in expectation, for random keys), which dilutes individual weights by a factor of $\sim 1/K$. Multiplying the logits by $\lambda_i \log(K)$ effectively raises the exponents by a factor of $\lambda_i \log(K)$, counteracting the $K$-fold growth of the denominator. This is the minimal correction that preserves the relative ordering of attention weights while restoring approximate scale invariance across sequence lengths, while incorporating the $1/\sqrt{d}$ correction for embedding dimension.

### 3.3 Relationship to Standard Attention

Standard scaled dot-product attention is recovered as the special case:

- $\beta_i = 0$ — no gating; the gate projections are unused and $a_{ij} = s_{ij}$
- $\gamma_i = -\infty$ — no ground state; $\max(\gamma_i, a_{ij}) = a_{ij}$, so $z_i = \sum_j e^{a_{ij}}$ and $w_{i0} = 0$
- $\lambda_i = \frac{1}{\sqrt{d}}$ — the scaling factor is set to provide no scaling based on the sequence length, just based on the embedding dimension.

This means principled attention is a *strict generalization* — any model that uses standard attention can be initialized to equivalent behavior, and the additional components can be gradually learned during training.

---

## 4. Variants and Ablation Structure

Each of the four new components — $\beta_i$, $\gamma_i$, $\lambda_i$, and $v_0$ — can be parameterized at different levels of expressiveness:

| Component | Constant                                                                 | Per-Layer                | Per-Head | Per-Query |
|-----------|--------------------------------------------------------------------------|--------------------------|----------|-----------|
| $\beta$ (gating strength) | Hyperparameter                                                           | Learned scalar per layer | Learned scalar per head | Learned from $q_i$ |
| $\gamma$ (ground bias) | Hyperparameter                                                           | Learned scalar per layer | Learned scalar per head | Learned from $q_i$ |
| $\lambda$ (scaling factor) | Hyperparameter (e.g. $\frac{\log{K}}{\sqrt{d}}$) | Learned scalar per layer | Learned scalar per head | Learned from $q_i$ |
| $v_0$ (ground value) | Fixed (e.g., zero)                                                       | Learned vector per layer | Learned vector per head | — |

Standard attention occupies the corner $\beta = 0$, $\gamma = -\infty$, $\lambda = \frac{1}{\sqrt{d}}$, $v_0$ irrelevant. The per-query variants of $\beta_i$ and $\gamma_i$ are the most expressive, allowing the model to learn content-dependent selectivity: "this query should be highly selective" versus "this query should spread attention broadly and rely on the ground state." The per-head variants may capture the majority of the benefit at lower parameter cost.

The selection of $\lambda_i = \frac{\log(K)}{\sqrt{d}}$ is the most principled.

The ground value $v_0$ is unlikely to benefit from per-query parameterization, as it represents a head-level concept: "what to output when nothing is relevant."

[TBD: Experimental results across the ablation grid.]

---

## 5. Compatibility with Efficient Attention

Principled attention is compatible with FlashAttention-style tiling algorithms. The tiling loop maintains three accumulators — the running max, the sumexp, and the weighted value sum — using the standard online softmax rescaling trick. Principled attention requires only one additional accumulator: a *ground-aware sumexp* that tracks $\sum_j e^{\max(\gamma_i, a_{ij}) - m_i}$ alongside the standard $\sum_j e^{a_{ij} - m_i}$.

The ground-aware sumexp receives the same max-rescaling correction as the standard sumexp and value accumulator at each tile boundary. At the end of the tiling loop, the effective ground weight is computed as the difference between the two sumexp accumulators, normalized by the ground-aware sumexp:

$$
w_{i0} = \frac{\text{groundexp} - \text{sumexp}}{\text{groundexp}}
$$

The ground value is then blended into the output in a single final step. This means the inner tiling loop adds only one extra accumulator and one extra `max` operation per element — negligible overhead relative to the matmul-dominated cost of attention.

---

## 6. Conclusion

[TBD]

---

## References

[TBD]

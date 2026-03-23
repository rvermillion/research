# Principled Attention:<br>Gated Logits, Learned Ground States, and Scale-Invariant Normalization for Transformer Attention

**Richard Vermillion**
{: class="author" }

## Abstract

Standard scaled dot-product attention makes three implicit assumptions that limit the expressiveness of transformer architectures: (1) all attention mass must be distributed across keys, even when no key is relevant; (2) a single bilinear score conflates semantic relevance with selection; and (3) attention weights dilute as sequence length grows, degrading signal fidelity in long contexts. We introduce *principled attention*, a strict generalization of standard attention that addresses each limitation through a minimal, orthogonal extension. A learned *ground state* $v_0$ with per-query relevance threshold $\gamma_i$ allows the model to explicitly attend to "nothing." A *gating mechanism* governed by $\beta_i$ decomposes query-key interaction into separate semantic and gate projections, enabling suppression without re-entangling semantic relevance and selection. A $\log(K)$ scaling term governed by $\alpha_i$ restores scale invariance across sequence lengths. Standard attention is recovered in the limit as $\alpha_i, \beta_i, \gamma_i \to -\infty$. We present a systematic ablation across variants of each component and demonstrate that principled attention improves [TBD].

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
 
To prevent this, the model would often need to learn query and key projections that push the majority of keys into a negative-dot-product regime relative to each query. This amounts to sacrificing representational capacity to solve a normalization problem — the model must use its limited projection dimensions to enforce geometric separation rather than to encode semantic distinctions. In practice, this is an unreasonable demand, and the denominator grows with $K$.
 
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
b_{ij} &= \mathrm{softplus}(\beta_i) \, \mathrm{softplus}(-g_{ij}) &&  \text{(gate suppression)} \\
m_{ij} &= (1 + \mathrm{softplus}(\alpha_i) \log{K})(s_{ij} - \gamma_i) &&  \text{(amplified margin)} \\
a_{ij} &= \gamma_i + m_{ij} - b_{ij} && \text{(final logit)} \\
z_i &= \sum_{j=1}^{K} e^{\max(\gamma_i,\, a_{ij})} && \text{(normalizer)} \\
w_{ij} &= \frac{e^{a_{ij}}}{z_i} && \text{(attention weight)} \\
w_{i0} &= 1 - \sum_{j=1}^{K} w_{ij} && \text{(ground weight)} \\
o_i &= w_{i0}\, v_0 + \sum_{j=1}^{K} w_{ij}\, v_j && \text{(output)}
\end{aligned}
$$

where $\beta_i$ is a learned gating strength, $\gamma_i$ is a learned relevance threshold, $\alpha_i$ is a margin amplification factor, and $v_0$ is a learned ground value.

The final logit, $a_{ij}$, decomposes into three additive terms in log-space:
$$
a_{ij}
= \gamma_i +
\underbrace{(1+\mathrm{softplus}(\alpha_i) \log K)(s_{ij}-\gamma_i)}_{\text{amplified margin}}
\;-\;
\underbrace{\mathrm{softplus}(\beta_i)\,\mathrm{softplus}(-g_{ij})}_{\text{gate suppression}}
$$

This decomposition gives the mechanism a clean interpretation. The baseline term $\gamma_i$ sets the learned relevance threshold. The amplified-margin term increases a key’s claim on attention according to how far its semantic score exceeds that threshold, while counteracting the growing burden of proof in longer contexts. The gate-suppression term subtracts a nonnegative penalty based on the gate signal. Because all three terms combine additively in log-space, they operate in a common currency: semantic evidence can increase a key’s claim on attention, while gating can only reduce that claim, not create it.

### 3.2 How Each Component Addresses a Problem

#### 3.2.1 Ground state $v_0$ and relevance threshold $\gamma_i$
Directly addressing §2.1, the normalizer $z_i$ uses $\max(\gamma_i, a_{ij})$ rather than $a_{ij}$ alone. This means that any key whose logit falls below $\gamma_i$ contributes $e^{\gamma_i}$ to the denominator, even though its numerator contribution remains $e^{a_{ij}}$. The difference is not assigned to that key and therefore defaults to the ground state.

We interpret $\gamma_i$ as a learned relevance threshold. This threshold is one-sided: keys below it surrender part of their claim to the ground state, with the penalty increasing smoothly as they fall farther below, while keys at or above the threshold incur no penalty and retain their full claim on attention.

The ground weight $w_{i0} = 1 - \sum_j w_{ij}$ is not a fixed scalar, but the aggregate shortfall of all keys that fail to meet the relevance threshold.  Formally, the mass assigned to the ground state is:
$$
w_{i0} = \frac{\sum_{j=1}^{K}{\max(0, e^{\gamma_i} - e^{a_{ij}})}}{z_i}
$$
This ensures that the ground state only claims mass from keys to the extent that they are insufficiently relevant. In this way, the mechanism gives absolute meaning to scores relative to $\gamma_i$: being the best available key is no longer sufficient by itself; a key must also clear a learned baseline to retain its full share.

Critically, if all keys exceed the relevance threshold, the ground state receives no mass.

The ground value $v_0$ is a learned vector representing the "default output" — what the attention head produces when nothing in the context is sufficiently worth attending to. This replaces emergent sink behavior with an explicit, learnable mechanism.

#### 3.2.2 Gating via $\beta_i$ and $\mathrm{softplus}(-g_{ij})$

Addressing §2.2, the gate score $g_{ij}$ is computed from dedicated gate projections $q^g_i$ and $k^g_j$, separate from the semantic projections. The $\mathrm{softplus}(-g_{ij})$ term is always non-negative, so the gate can only *subtract* from the semantic score — it suppresses, never promotes. The strength of suppression is controlled by $\mathrm{softplus}(\beta_i)$, also non-negative, which can be learned per-query, per-head, or per-layer.
 
The behavior of the gate is best understood through three regimes. When $g_{ij}$ is large and positive (gate is *aligned*), $\mathrm{softplus}(-g_{ij}) \approx 0$ and the semantic score passes through unmodified. When $g_{ij}$ is large and negative (gate is *misaligned*), $\mathrm{softplus}(-g_{ij}) \approx |g_{ij}|$ and the suppression is strong, subtracting a large value from the logit. The critical middle case is $g_{ij} \approx 0$ — *gate indifference* — which is also the typical outcome in high-dimensional space, since the dot product between unrelated gate vectors concentrates around zero. At indifference, $\mathrm{softplus}(0) = \log(2)$, so the suppression term becomes $\mathrm{softplus}(\beta_i) \log(2)$.
 
This makes $\beta_i$ interpretable: it controls how much the model penalizes gate indifference relative to gate alignment. For $\beta_i = 0$, an indifferent key has its logit reduced by $\mathrm{softplus}(0) \log(2) = \log(2)^2 \approx 0.48$ compared to an aligned key — a modest but meaningful suppression. Larger $\beta_i$ makes the model more demanding of explicit gate alignment before allowing a key to retain its semantic claim. In this sense, $\beta_i$ is not merely a "gating strength" but a parameter that controls how much structural uncertainty or mismatch the model tolerates before suppressing an otherwise semantically relevant key.

This decomposition should be viewed as one principled point in a broader design space of grounded and decomposed attention mechanisms. We focus on the asymmetric suppress-only variant because it preserves a clean separation of roles: semantic projections propose relevance, gates modulate confidence in that proposal by suppressing unreliable matches, and the ground state handles fallback behavior when no key sufficiently earns attention. Other designs could allow gates to both suppress and promote attention, and we do not rule out such variants. We focus on suppress-only gating because promotion would re-entangle structural modulation with semantic relevance scoring, reducing interpretability and introducing redundant pathways for increasing attention logits.

#### 3.2.3 Sequence length margin amplification via $\alpha_i$
Addressing §2.3, we amplify the margin, $s_{ij}-\gamma_i$, by $1 + \mathrm{softplus}(\alpha_i)\log{K}$. Applied to the degree by which the semantic score exceeds the relevance threshold, this factor counteracts the increasing burden of proof imposed by longer contexts.

As the number of keys grows, the softmax denominator typically grows as well: for largely unrelated keys, many semantic scores cluster near a common baseline, often close to zero, so $\sum_j e^{s_{ij}}$ grows roughly linearly with the number of such keys. This dilutes the attention weight assigned to any fixed margin above baseline. In standard softmax, maintaining comparable allocation as $K$ grows therefore requires logit advantages that increase on the order of $\log{K}$.

The $\log{K}$ term provides a direct correction for this effect. By scaling the semantic margin relative to the relevance threshold, it increases a key’s effective claim on attention in proportion to the logarithm of the available context size. This leaves the ordering of semantic scores unchanged while reducing the extent to which longer sequences force relevant keys to become increasingly exceptional merely to retain comparable weight.

Because the amplification is a positive scalar applied uniformly to semantic margin, it preserves the ordering of semantic relevance while changing the strength with which those relevance differences are expressed in attention.

We view this as a principled first-order correction rather than an exact solution: it is motivated by the typical growth of denominator mass with key count, and aims to make attention allocation more stable across sequence lengths without otherwise changing the structure of the mechanism. Just as the $1/\sqrt{d}$ factor in standard attention corrects for the scale of dot products across embedding dimensions, the $\log{K}$ term here provides an analogous first-order correction for the growth of denominator mass across sequence length.

### 3.3 Relationship to Standard Attention

Standard softmax scaled dot-product attention is recovered in the limit as:

- $\alpha_i \to -\infty$ — no sequence length margin amplification 
- $\beta_i \to -\infty$ — no gating; the gate projections are unused
- $\gamma_i \to -\infty$ — no ground state bias to overcome
- $v_0$ — irrelevant as it no longer enters the equation

In that limit, $a_{ij} \to s_{ij}$, $z_i \to \sum_j{e^{s_{ij}}}$, and $w_{i0} \to 0$.  This means principled attention is a *generalization* — any model that uses standard attention can be initialized to equivalent behavior, and the additional components can be gradually learned during training.

---

## 4. Variants and Ablation Structure

Each of the four new components — $\beta_i$, $\gamma_i$, $\alpha_i$, and $v_0$ — can be parameterized at different levels of expressiveness:

| Component                              | Constant                                                               | Per-Layer                | Per-Head | Per-Query |
|----------------------------------------|------------------------------------------------------------------------|--------------------------|----------|-----------|
| $\beta$ (gating strength)              | Hyperparameter                                                         | Learned scalar per layer | Learned scalar per head | Learned from $q_i$ |
| $\gamma$ (relevance threshold)         | Hyperparameter                                                         | Learned scalar per layer | Learned scalar per head | Learned from $q_i$ |
| $\alpha$ (margin amplification) | Hyperparameter | Learned scalar per layer | Learned scalar per head | Learned from $q_i$ |
| $v_0$ (ground value)                   | Fixed (e.g., zero)                                                     | Learned vector per layer | Learned vector per head | — |

Standard softmax is recovered in the limit $\alpha, \beta, \gamma \to -\infty$, and $v_0$ irrelevant. The per-query variants of $\beta_i$ and $\gamma_i$ are the most expressive, allowing the model to learn content-dependent selectivity: "this query should be highly selective" versus "this query should spread attention broadly and rely on the ground state." The per-head variants may capture the majority of the benefit at lower parameter cost.

The ground value $v_0$ is unlikely to benefit from per-query parameterization, as it represents a head-level concept: "what to output when nothing is relevant."

[TBD: Experimental results across the ablation grid.]

---

## 5. Compatibility with Efficient Attention

Principled Attention remains compatible with FlashAttention-style tiled online accumulation. The sequence-length amplification term introduces a lightweight prepass requirement when effective key count depends on masking or dynamic key/value segment selection from a cache.

In such cases, the model must determine — either from the mask or from the key/value segment selection policy — how many keys are visible to each query *before* tile scoring begins. For causal masks and sliding-window masks, this is trivial. For segmented caches, it requires scanning the active segments to determine how many keys will be visible to each query.

Once the per-query sequence length $K$ has been computed, the scoring and accumulation are fully compatible with online tiled attention.  Compared to standard attention, Principled Attention requires only an additional denominator accumulator for grounded mass.

The ground-aware sumexp receives the same max-rescaling correction as the standard sumexp and value accumulator at each tile boundary. At the end of the tiling loop, the effective ground weight is computed as the difference between the two accumulators, normalized by the ground-aware sumexp:

$$
w_{i0} = \frac{\text{groundexp}_i - \text{sumexp}_i}{\text{groundexp}_i}
$$

The ground value is then blended into the output in a single final step:
$$
\text{out}_i = \frac{\text{values}_{i}}{\text{groundexp}_i} + w_{i0} v_0
$$
This means the inner tiling loop adds only one extra row-wise accumulator — negligible overhead relative to the matmul-dominated cost of attention.

---

## 6. Conclusion

[TBD]

---

## References

[TBD]

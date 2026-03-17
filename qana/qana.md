# Query-As-Network Attention: Generalizing Attention Through Query-Parameterized Nonlinear Scoring

**Author:** Richard Vermillion  
**Date:** 2025-10-22  

---

## Abstract {: class="abstract"}

We introduce **Query-As-Network Attention (QANA)**, a generalization of Scaled Dot-Product Attention (SDPA) in which
each query vector parameterizes a lightweight neural network that computes compatibility scores with keys. This extends
the “data-as-weights” interpretation of SDPA, replacing its strictly bilinear form with a query-conditioned nonlinear
scoring function while remaining compatible with existing Transformer architectures. QANA preserves initialization
equivalence with SDPA through a linear skip path, enabling drop-in replacement in pre-trained models without
changing their initial behavior. We show that QANA can express a richer family of attention kernels while maintaining
relative positional equivariance via a row-wise application of rotary positional encoding (RoPE). Finally, we
hypothesize that the added expressiveness could enable smaller key dimensions without performance degradation,
potentially reducing key/value cache memory requirements in autoregressive generation.

---

## 1 Introduction

Scaled Dot-Product Attention (SDPA) is the central mechanism in Transformer models. It computes attention logits as
a scaled inner product between *query* and *key* vectors:
$$
\text{score}(q_i, k_j) = \frac{q_i^\top k_j}{\sqrt{d_k}}.
$$
This bilinear form has proven remarkably effective, enabling Transformers to achieve state-of-the-art results across
domains. Despite its efficiency and elegance, SDPA imposes a **bilinear limitation**: the similarity function is
restricted to a single global metric shared across all tokens and contexts. This formulation implicitly treats the
**query as data**, not as parameters that could shape a more expressive compatibility function.

However, we can reframe this computation: rather than viewing the query as data to be compared against keys, we can
view it as parameters that define how to score keys. This shift in perspective opens new architectural possibilities.
We revisit SDPA through a **query-as-weights** lens, observing that
$$
q_i^\top k_j = \sum_d q_{i,d} k_{j,d}
$$
is equivalent to using the query as the *weights* of a one-layer network acting on the key. This observation motivates
a natural extension: rather than restricting the query to parameterize a simple linear function (the dot product), we
allow it to parameterize a small **neural network**. This Query-As-Network Attention (QANA) mechanism generalizes SDPA
while maintaining compatibility with existing architectures and positional encoding schemes.

---

## 2 From Dot-Product Attention to Query-As-Network Attention

### 2.1 Data-as-Weights Perspective

The standard formulation of attention treats queries and keys symmetrically -- both are vectors in the same space
whose similarity is measured. However, their roles are fundamentally asymmetric: the query determines *how to attend*,
while keys represent *what to attend to*. This asymmetry becomes explicit when we rewrite the dot product as a
weighted sum.

In SDPA, the query acts as a vector of linear coefficients over the key’s dimensions. Each element of $q$ functions
as a synaptic weight on $k$:
$$
\text{score}(q, k) = q^\top k = \sum_d w_d(k_d) \quad \text{with} \quad w_d = q_d.
$$
This “query-as-weights” view highlights the asymmetry: the key provides data, and the query supplies parameters. In
this view, each query dimension acts as a learned weight applied to the corresponding key dimension. The query
defines a simple linear scoring function over the key space. QANA extends this by allowing each query to define
a more expressive, nonlinear scoring function.

To increase expressiveness, we replace this single-layer linear map with a **query-parameterized MLP**, turning each
query into a micro-network that defines its own attention kernel.

The question then becomes: what is the minimal extension that adds expressive power while preserving the desirable
properties of SDPA—efficiency, rotational equivariance for positional encoding, and initialization compatibility?

---

## 3 Query-Parameterized Scoring Function

### 3.1 Definition

QANA augments the standard dot-product attention with a query-parameterized MLP that operates on each key. Crucially,
we retain the original SDPA computation as a skip connection, ensuring that at initialization, QANA behaves identically
to standard attention. For each query $q_i$ and key $k_j\in\mathbb{R}^{D_K}$, QANA defines the scoring function as:
$$
\text{score}(q_i, k_j)
= \frac{\widetilde{s}\_{i}^\top k\_j}{\sqrt{D\_K}}
+ V\_{i}\,\sigma(\widetilde{U}\_{i} k\_j + b\_{i}) + c\_{i},
$$

The scoring function consists of two components:

- Skip connection: $\frac{\widetilde{s}_{i}^\top k_j}{\sqrt{D_K}}$ preserves the standard SDPA computation
- Nonlinear term: $V\_{i}\,\sigma(\widetilde{U}\_{i} k\_j + b\_{i}) + c\_{i}$ adds query-specific nonlinear scoring
  capacity. The tildes indicate RoPE-encoded vectors, which we discuss in Section 3.3.

### 3.2 Query Slicing

To implement QANA, we expand the query dimension from $D_K$ to $D_Q > D_K$. Each query vector is partitioned into
components that parameterize both the skip connection and the MLP.

Each query vector $q_i\in\mathbb{R}^{D_Q}$ is divided into a linear query $s_i$ for the skip path and an input weight
$U_i$, an output weight $V_i$, an input bias $b_i$, and an output bias $c_i$ for the MLP:
$$
q_i = [s_{i}; \; U_{i}; \; V_{i}; \; b_{i}; \; c_{i}],
$$
with dimensions:
$$
\begin{aligned}
s_{i} &\in \mathbb{R}^{D_K},
U_{i} &\in \mathbb{R}^{h\times D_K}, &\quad &\text{(flattened size } h D_K)\\
V_{i} &\in \mathbb{R}^{1\times h}, \\
b_{i} &\in \mathbb{R}^{h}, \\
c_{i} &\in \mathbb{R}. \\
\end{aligned}
$$
Thus,
$$
D_Q = D_K + (h D_K + 2h + 1).
$$
For example, with $D_K = 64$ and $h = 4$ hidden units, we have $D_Q = 64 + (256 + 8 + 1) = 329$. While this increases
the per-query parameter count, it may enable compensating reductions in $D_K$ (see Section 5).

We efficiently extract these components through slicing and reshaping operations. In practice, this is a simple tensor
decomposition:
$$
\begin{aligned}
s_i &= q_i[\dots, :D_K], \\
U_i &= q_i[\dots, D_K:D_K + hD_K].\text{reshape}(\dots, h, D_K), \\
V_i &= q_i[\dots, D_K + hD_K:D_K + hD_K + h].\text{reshape}(\dots, 1, h), \\
b_i &= q_i[\dots, -h-1:-1], \\
c_i &= q_i[\dots, -1:]. \\
\end{aligned}
$$
The ellipses notation ($\dots$) preserves leading batch and head dimensions, which remain unchanged throughout
these operations.

Thus, each query encodes both a linear projection $s_i$ for the standard dot-product term and the full
parameterization of a one-hidden-layer MLP acting on each key.

At initialization, $W_Q$ will be set so that all nonlinear parameters $U,b,V,c$ are set to zero, making
QANA **function-preserving** with SDPA.

### 3.3 Rotary Positional Encoding

A key challenge in extending SDPA is preserving compatibility with rotary positional encoding (RoPE), which
encodes relative positions through paired rotations of queries and keys. While the skip connection naturally
supports RoPE, the MLP path requires careful treatment.

#### 3.3.1 Skip Rotary Positional Encoding

We apply standard **rotary positional encoding** (RoPE) to each linear skip query $s_i$:

$$
\widetilde{s}_i = \text{RoPE}\_i(s_i),
$$

This ensures the skip connection maintains the same relative positional dependence as standard SDPA.

#### 3.3.2 Row-Wise Rotary Positional Encoding

**Challenge**: Standard SDPA's dot product $q_i^\top k_j$ is rotationally invariant under RoPE's paired rotations,
naturally encoding relative position $(j - i)$. However, arbitrary MLPs lack this invariance. So how do we preserve
relative position information in query-parameterized scoring?

**Solution**: We observe that the first hidden layer computation $U\_{i} k\_j$ can be decomposed into $h$ individual
dot products — one per row of $U_i$:
$$
[U\_{i} k\_j]\_\ell = (U\_{i})\_\ell^\top k\_j.
$$

Since each dot product is independently subject to rotational invariance, we can encode relative position by applying
RoPE **row-wise** to $U\_{i}$ (using query position $i$), while encoding $k_j$ with its position $j$:
$$
\widetilde{U}\_{i} = \text{RoPE}\_i(U\_{i}), \quad \widetilde{k}\_j = \text{RoPE}\_j(k\_j).
$$

**Key Property**: Because RoPE applies orthogonal rotations, each row-wise dot product is preserved:
$$
(U_{i})\_\ell^\top k\_j = (\widetilde{U}\_{i})\_\ell^\top \widetilde{k}\_j,
$$

This means the entire hidden layer computation retains relative position dependence:
$$
U\_{i} k\_j = \widetilde{U}\_{i} \widetilde{k}\_j.
$$

The subsequent nonlinearity and output projection $V$ then operate on these position-aware hidden activations,
ensuring that the full scoring function $\text{score}(q_i, k_j)$ naturally encodes relative position $(j - i)$ through
a richer, query-conditioned function.

---

## 4 Function-Preserving Swap with SDPA

A practical advantage of QANA’s design is that it can be initialized to exactly replicate SDPA behavior. This
enables safe deployment in pre-trained models: we can swap in QANA and continue training, gradually learning
query-conditioned scoring without disrupting the model’s existing capabilities. The procedure is straightforward:

1.	**Expand the query projection.**  
Extend $W_Q$ so it outputs $D_Q = D_K + (hD_K + 2h + 1)$ instead of $D_K$.
    - Copy the first $D_K$ rows of the original $W_Q$ unchanged (these generate $s$).
    - Initialize the additional rows (which produce the flattened $U$, $b$, $V$, $c$) to zero.
This ensures the nonlinear portion of QANA initially contributes nothing.
2. **Preserve the baseline function.**  
Because the added rows of $W_Q$ are zero, the extra slices $U_{i}, b_{i}, V_{i}, c_{i}$ are all zero for every query.
The nonlinear term $V\_{i}\,\sigma(\widetilde{U}\_{i} k\_j + b\_{i}) + c\_{i}$ therefore vanishes, leaving the pure
SDPA term.
3. **Apply RoPE consistently.**  
    - Apply standard RoPE to both $s$ and $k$.
    - Apply RoPE row-wise (using the query position) to $U$ when computing the nonlinear term, ensuring relative
      rotational invariance.
4. **Fine-tune safely.**  
The model’s logits and outputs are identical to the pretrained SDPA at initialization. You can then fine-tune or
continue training, allowing the zero-initialized tail of $W_Q$ to learn query-conditioned nonlinear scoring. During
training, one can optionally use a learned gate parameter $\alpha$ to gradually blend in the nonlinear term:
$\alpha \cdot [V_i\sigma() + c_i]$, starting from $\alpha = 0$.

This initialization strategy provides a safety mechanism for deploying QANA in production systems. If the nonlinear
components fail to learn useful functions during fine-tuning, the model gracefully degrades to standard attention
rather than catastrophically forgetting. Note, however, that this approach maintains the original $D_K$, forgoing
potential cache size reductions until retraining from scratch with smaller keys (see Section 5).

---

## 5 Hypothesis: Smaller Keys, Same Power

A key question is whether QANA’s increased expressiveness merely adds overhead, or enables compensating efficiency
gains elsewhere in the architecture. We hypothesize that query-conditioned nonlinear scoring can compensate for
reduced key dimensionality.

**Rationale**: In standard SDPA, the key dimension $D_K$ must be large enough to capture sufficient information for
all possible query patterns across all contexts. This is a worst-case requirement—the keys must be expressive
enough for any query that might attend to them. With QANA, each query can apply a custom nonlinear transformation
to extract task-specific information from keys. This suggests that keys might need less inherent dimensionality if
queries can adaptively reshape and filter the key space.

**Practical implications**: The key/value cache in autoregressive generation scales as $O(D_K \times T)$ where $T$ is
sequence length. If QANA enables, say, a 2× reduction in $D_K$ while maintaining performance, the cache memory and
bandwidth requirements during generation would drop proportionally—even accounting for increased query computation.
Since KV cache access is often a primary bottleneck in long-context generation, this trade-off could be favorable.

**Open question**: The optimal balance between query expressiveness ($D_Q$, hidden width $h$) and key compression
($D_K$ reduction) is an empirical question we aim to explore through the experiments outlined in Section 6.

---

## 6 Experimental Plan

While this work presents QANA as a conceptual framework, validating its practical utility requires systematic
experimentation. We outline a proposed experimental methodology to test QANA’s expressiveness, efficiency trade-offs,
and architectural design choices.

### 6.1 Baseline Comparison
- Replace SDPA with QANA in LLaMA- or GPT-style models.
- Initialize as function-preserving swap.
- Fine-tune with gradually increasing gate parameter $\alpha$ on the nonlinear term.

### 6.2 Ablations
1. **Hidden width $h$:** {1, 2, 4, 8}  
2. **Key dimension reduction:** {1.0×, 0.75×, 0.5×, 0.25× baseline $D_K$}  
3. **With / without linear skip term**  
4. **RoPE variants**  
5. **Low-rank or shared-basis variants** (efficiency, reducing $D_Q$)  
6. **Effect of gating schedule** ($\alpha$ warmup vs. fixed)

### 6.3 Metrics
- Perplexity / accuracy on standard language modeling benchmarks  
- Inference-time latency and KV-cache memory usage  
- Logit-scale variance and attention entropy (to measure expressivity)

### 6.4 Visualization
- Attention heatmaps comparing SDPA vs. QANA under identical prompts  
- Analysis of the nonlinear contribution by comparing full QANA scores against skip-connection-only scores to
  identify contexts where query-conditioned scoring deviates most from standard attention patterns.

---

## 7 Discussion and Future Work

QANA generalizes dot-product attention by allowing queries to define adaptive scoring kernels—locally nonlinear
transformations of the key space. Rather than treating attention as a single globally-shared similarity metric,
QANA enables each query to specify how similarity should be computed for its particular context and task.

This design opens a new axis for attention mechanisms: **how much of the scoring function should be learned per-query
versus globally shared**? Standard SDPA sits at one extreme (fully shared bilinear metric), while hypothetically,
one could imagine the opposite extreme (fully independent scoring networks per query, with no shared structure).
QANA explores a middle ground that adds expressive capacity while maintaining efficiency and architectural
compatibility.

Future directions include:

- Low-rank or MoE parameterizations of $U$ to reduce per-query parameters
- Hybrid "query-as-network" and "expertized query" systems
- Scaling analysis to identify optimal $D_Q/D_K$ ratios
- Theoretical analysis of the function class expressible by QANA vs. SDPA
- Extension to cross-attention and encoder-decoder architectures

Ultimately, whether QANA provides practical advantages over standard attention is an empirical question. We hope
this framework stimulates exploration of alternative attention mechanisms that challenge the assumption that
dot-product similarity is the optimal foundation for all attention computations.

---

## References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.  
2. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, ACL 2021.  
3. Tay et al., *Efficient Transformers: A Survey*, ACM CSUR 2022.  
4. Choromanski et al., *Rethinking Attention with Performers*, ICLR 2021.  
5. Press et al., *Train Short, Test Long: Attention with Linear Biases*, arXiv:2108.12409.  
6. Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015.  
7. Jaegle et al., *Perceiver: General Perception with Iterative Attention*, ICML 2021.  
8. Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, JMLR 2022.  

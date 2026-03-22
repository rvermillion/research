# Grounded Attention
**Richard Vermillion**  
{: class="author" }

## Motivation

Standard softmax attention is shift-invariant: it operates on relative differences between
logits, not their absolute magnitudes. This means a query-key score of −2 receives high
attention if all other scores are −5, even though nothing in the context is truly relevant. The
model cannot express “attend to nothing.” In practice, models compensate by learning to
route garbage attention to designated sink tokens (typically BOS or separator tokens) and
by keeping value vectors defensively bland in case they receive spurious attention. Both are
emergent workarounds for an architectural gap.

## Method

We introduce a learned "ground" value vector $v_{\text{ground}} \in \mathbb{R}^{d_v}$ per
attention head (or per KV group under GQA) and a learned ground logit bias scalar $b_h$ per head.
The attention output for a given query becomes:

$$o = \frac{v_{\text{ground}} + \sum_{i} e^{a_i} v_i}{e^{b_h} + \sum_{k} e^{a_k}}$$

where $a_i$ are the standard query-key dot-product logits. The ground vector implicitly
participates in attention with a fixed logit of zero relative to $b_h$, requiring no additional
dot product. When real attention scores are large and positive, the $e^{b_h}$ term is
negligible and attention behaves normally. When all scores are low, the ground term dominates
and the output defaults to $v_{\text{ground}}$ — a learned, head-specific representation of
“nothing relevant was found.” The bias $b_h$ controls the threshold: heads with large
positive $b_h$ fall back to the null value more readily, while large negative $b_h$ recovers
near-standard softmax behavior.

## Parameters and Output-Preserving Initialization

The mechanism adds $n_{\text{kv_heads}} \times d_v + n_{\text{heads}}$ parameters per
layer (the ground value vectors plus the bias scalars) — negligible relative to total model size.
To apply this as a behavior-preserving patch to a pretrained model, initialize 
$v_{\text{ground}} = 0$ and $b_h$ to a large negative value (e.g., −10). At initialization, $e^{b_h} \approx 0$
and $v_{\text{ground}}$ contributes nothing, so the attention output is identical to standard
softmax. During fine-tuning, gradient descent discovers which heads benefit from the ground
mechanism and adjusts both $v_{\text{ground}}$ and $b_h$ accordingly.
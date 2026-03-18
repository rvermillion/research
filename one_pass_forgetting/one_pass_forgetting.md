# One-Pass Forgetting: Layer-Wise Self-Distillation for Learned Memory Compression in Transformers
**Richard Vermillion**  
{: class="author" }

**Abstract** {: class="abstract"}

Autoregressive transformers face a fundamental memory bottleneck as context grows. Transformers accumulate key–value
(KV) pairs during autoregressive inference, causing linear growth in memory and compute with sequence length.  
We introduce **One-Pass Forgetting**, a training framework that teaches transformers when and how to forget. A
lightweight **forget network** learns to predict which past tokens remain useful, assigning retention probabilities
to KV entries. A differentiable **gated teacher–student mechanism** provides supervision within a single forward pass.
Each layer computes full attention to measure every past token’s *future utility*, then trains a forget network to
approximate these scores using only causal information. Outputs from the fully-attentive “teacher” and the masked
“student” paths are blended layer-by-layer through a gate $\lambda$, which is annealed during training. This approach
enables stable, efficient training of adaptive KV-cache compression that directly mirrors inference-time behavior.  

---

## 1 Introduction

Transformers excel at long-context modeling but scale poorly: each new token expands the KV cache and quadratic attention cost. Real inference requires *forgetting*: discarding tokens whose contribution to future outputs is negligible.  
Existing heuristics (fixed windows, attention-based pruning) ignore downstream loss impact and are detached from training.  

We propose **One-Pass Forgetting**, a self-distilled learning scheme that trains the model to approximate its own contribution analysis *within* a single pass.  
Key ideas:

1. **Future-aware teacher:** During each forward, we compute per-token contribution scores $s_{t,j}$ using the unmasked full attention, measuring how much key–value pair $j$ affects future outputs $y_t$.
2. **Causal forget network:** A small network consumes only causal information (queries, keys, values, time indices, and local statistics) and predicts $r_{t,j}\in[0,1]$, an estimated retention probability.
3. **Differentiable mask and gate:** The predicted retentions generate an additive *forgetful bias* $B_{t,j}$ applied to attention logits; masked (“student”) outputs $y^f_t$ are combined with unmasked (“teacher”) outputs $y_t$ via $y^{mix}_t=(1-\lambda)y_t+\lambda y^f_t$.  
   The gate $\lambda$ is annealed from 0→1, smoothly transitioning from teacher supervision to fully on-policy forgetting.

This single-pass structure matches inference, where only one forward is available and tokens are evicted permanently.  

---

## 2 Methodology

### 2.1 Contribution scoring

For each layer $l$ and token pair $(t,j)$ with $j\le t$:

$$
a_{t,j} = \frac{q_t^\top k_j}{\sqrt{d_k}}, \quad
\alpha_{t,j} = \mathrm{softmax}_j(a_{t,j}),
\quad
u_{t,j} = \alpha_{t,j} v_j,
\quad
y_t = \sum_j u_{t,j}.
$$

Define normalized contribution:

$$
s_{t,j} = \Big\|\frac{u_{t,j}}{y_t}\Big\|^2_2,
$$
capturing both attention weight and value magnitude.

To estimate long-term importance, pool over future timesteps:
$$
S_{t,j} = \mathrm{softmaxpool}_{i\ge t}
  \big(\tau_f\,\gamma^{i-t}s_{i,j}\big),
$$
where $\tau_f$ anneals (sharpens) and $\gamma$ optionally discounts.  
This acts as a smooth, differentiable “future utility” target for retention.

### 2.2 Forget network and mask formation

Each layer ℓ hosts a small MLP or attention-lite module $f_\theta$ producing $r_{t,j}=f_\theta(\text{causal features})$.  
These features may include:
- relative position $t-j$,
- attention logits $a_{t,j}$,
- key/value norms, and
- EMA of past attention to $j$.

Retention probabilities are mapped to a forgetful bias:
$$
B_{t,j} = \log \sigma(\tau_f(r_{t,j}-\tau_0)),
$$
clipped to $[-B_{\max},0]$.
Biases are **column-monotone**:
$$
B_{t+1,j}\le B_{t,j},
$$
enforced via cumulative minimum along time to guarantee no “un-forgetting.”

The masked attention logits are
$$
\tilde a^f_{t,j} = a_{t,j} + M_{t,j} + B_{t,j},
$$
with $M$ the standard causal mask.

### 2.3 Layer-wise gated teacher–student blending

Each layer emits:
$$
y^{mix,(l)} = (1-\lambda_{l})\,y^{(l)} + \lambda_{l}\,y^{f,(l)},
$$
where $y^{(l)}$ is the full teacher output and $y^{f,(l)}$ is computed with the forgetful mask.  
As training progresses, $\lambda$ increases (per schedule or per-layer), gradually shifting reliance from teacher to student.

### 2.4 Training objectives

1. **Primary task loss:** $ \mathcal{L}_\text{LM}$ from final logits using mixed outputs.  
2. **Forget-net distillation:** KL or MSE between predicted retentions and normalized future utilities:
   $$
   \mathcal{L}_r = \mathrm{KL}(\mathrm{stopgrad}(S_{t,j})\;\|\;r_{t,j}).
   $$
3. **Budget regularizer:** keeps expected retention below capacity target $B_t$:
   $$
   \mathcal{L}_b = \lambda_b
     \big|\sum_{j\le t}\sigma(\tau_f(r_{t,j}-\tau_0))-B_t\big|.
   $$
Total loss:
$$
\mathcal{L}=\mathcal{L}_\text{LM}+\beta_r\mathcal{L}_r+\beta_b\mathcal{L}_b.
$$

Gradients propagate through the student path only (teacher side `stopgrad`), enabling stable credit assignment.

### 2.5 Curriculum

1. **Phase 1 (Distillation):**  
   $\lambda$ small (≈0.1); $\tau_{(f)}$ low (soft).  
   Forget net learns from teacher’s full-attention utilities.

2. **Phase 2 (On-policy):**  
   $\lambda \to 1$, $\tau_{(f)}$ annealed high (sharp).  
   Forget net directly modulates attention; supervision transitions to task loss only.

---

## 3 Inference Alignment

During training, contribution scores $s_{t,j}$ and their pooled future utilities $S_{t,j}$ quantify how much each key–value pair influences future outputs, providing the *teacher* signal for the forget net. The forgetful mask then down-weights unhelpful tokens via continuous, differentiable biases $B_{t,j}$ in attention logits.  

At inference, this mechanism is replaced by an **explicit cache eviction policy** derived from the same retention probabilities $r_{t,j}$:

- Each step, the forget net computes $r_{t,j}\in[0,1]$ for past KV pairs using only causal features (no lookahead or contribution estimates).
- Keys whose predicted retention falls below a threshold $\tau_{\mathrm{low}}$ are **evicted** from the cache through an O(1) swap-delete operation across all heads.  
  The corresponding key/value entries are replaced with the tail of the live prefix and the cache end pointer is decremented.
- A **remember window** of the most recent $w$ tokens is always protected from eviction, ensuring local coherence and short-term memory.
- Optional hysteresis prevents oscillatory behavior: once a token is evicted, it cannot re-enter the cache (enforcing column-wise monotonicity).

Thus, at inference:
- No attention masks or contribution computations are needed.
- The forget net’s scalar retentions act as *learned eviction probabilities*.
- The model’s live context automatically contracts and expands to fit relevance and hardware budgets.

This design eliminates quadratic scaling while preserving high-level semantics: during training, attention masking teaches *how much each token should matter*; during inference, eviction operationalizes *how long each token should persist.*

---

## 4 Experiments

**Datasets:** Long-context language benchmarks (PG19, Books3, Code, ArXiv).  
**Baselines:** Sliding-window attention, retrieval-based pruning, LongLoRA, DynamicKV, Streaming Transformers.  
**Metrics:** perplexity, retrieval F1 on long dependencies, compute and memory footprint.

**Ablations**
1. Forget net architecture (MLP vs. attention-lite).  
2. Pooling function (softmaxpool vs. max vs. discount).  
3. Bias annealing τ₍f₎ schedule.  
4. Layer-wise $\lambda$ gating schedules.  
5. Monotonicity enforcement vs. free retention.  
6. Fixed vs. learned budget targets.  
7. Phase-2 on-policy fine-tuning vs. none.

**Expected outcomes:** significant reduction in active KV cache (>50%) with minimal perplexity loss (<1%), smooth degradation when budget is tight, and graceful forgetting of function words and locally absorbed context.

---

## 5 Related work

- **Memory pruning & streaming:** Transformer-XL, Compressive Transformer, Longformer, Mamba, Dynamic Memory Transformer.
- **Gradient-free credit assignment:** Hebbian/eligibility traces, RTRL approximations.
- **Self-distillation:** Born-Again Networks, knowledge distillation for transformers.
- **Neural compression:** learned attention sparsification, key pooling.

Unlike prior two-pass schemes, One-Pass Forgetting achieves self-supervised memory compression *within a single forward pass*, maintaining gradient flow and strict causal constraints.

---

## 6 Discussion and Future Work

The distinction between **training-time contribution scores** and **inference-time retention** embodies the transition from *analysis* to *action*:  
- During training, $s_{t,j}$ and $S_{t,j}$ measure *how valuable each token truly was* for future predictions;  
- During inference, $r_{t,j}$ predicts *how valuable it is likely to be going forward*, using only causal state.

The model thus learns an internal estimate of *prospective utility*, not merely retrospective importance. This parallels biological learning: synapses strengthen or decay not by replaying all past activations but by maintaining a predictive estimate of future usefulness.

The continuous forgetful mask during training ensures smooth gradient flow and allows the model to explore different retention policies. The discrete eviction at inference time is simply the *hard projection* of that policy onto a constrained memory budget. In practice, this produces a near-seamless behavioral match between training and inference:
- Differentiable masking teaches *what forgetting means*;
- Retention-driven eviction enforces *when to forget*.

Future work may explore adaptive runtime policies—e.g., dynamically varying thresholds $\tau_{\mathrm{low}}$ based on context length, layer sensitivity, or compute budget—and coupling the forget net with summarizers that distill evicted information into compressed memory vectors. Together, these mechanisms bring transformers closer to systems that manage attention, memory, and forgetting as first-class, learned processes.

---

**Conclusion**

One-Pass Forgetting replaces post-hoc cache management with a differentiable, train-time mechanism for selective memory.  
By coupling a causal forget network with gated teacher supervision in a single forward pass, transformers can learn to balance remembering and forgetting dynamically—mirroring how biological and efficient artificial systems manage long-term context.

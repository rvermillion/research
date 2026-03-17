
# Adaptive Rank Modification via Hebbian Eligibility Traces
**Richard Vermillion**  
{: class="author" }

---

## Abstract {: class="abstract" }
We propose a biologically inspired mechanism for adaptive model capacity in low-rank neural architectures. Building on
low-rank factorization of weight matrices, we introduce **Hebbian eligibility traces** that track the long-term 
usefulness of each rank-1 component during training. These traces serve as distributed credit assignments that drive
both **rank growth** and **rank shrinkage**, allowing models to allocate representational capacity dynamically in
response to training pressure. Unlike post-hoc compression or static rank assignment, our approach maintains 
function-preserving updates during capacity changes, avoids catastrophic loss spikes, and uses local statistics derived
from activations and gradients. We outline theoretical motivations, implementation details, and experimental protocols
for evaluating this method in transformer architectures, particularly in the Q/K/V projections of attention layers.
Unlike post-hoc compression, which relies on data-agnostic matrix decompositions such as SVD, our method continuously
re-weights rank components by their utility under the observed data distribution and task gradients, allowing
neighboring parameters to co-adapt during training.

---

## 1 Introduction
Transformer networks and their derivatives are often over-parameterized, with many layers containing significant
redundancy. Recent research has demonstrated that low-rank factorization of linear projections can dramatically reduce
parameter count without large accuracy losses. However, most methods fix the rank *a priori* or apply compression 
*post hoc*, ignoring the dynamics of learning and the uneven distribution of representational pressure across layers
and time.

We propose a dynamic alternative: **Dynamic Hebbian Adaptive Rank Modification Algorithm (DHARMA)**. Each linear
projection \(W\in\mathbb{R}^{N\times N}\) is factorized as \(W=UV\), where \(U\in\mathbb{R}^{N\times K}\) and
\(V\in\mathbb{R}^{K\times N}\). During training, the model monitors each rank‑1 term \(u_i v_i^\top\) using local
eligibility traces derived from Hebbian interactions between pre‑activations and backpropagated gradients. These
traces accumulate evidence for the term’s long‑term utility. When global loss improvement plateaus, the traces guide
**targeted rank expansion or contraction**: growing new directions where gradient variance is high and pruning
under-utilized ones where eligibility decays toward zero.

This mechanism offers a simple, distributed, and biologically plausible route to *automatic capacity control* in deep
networks—analogous to synaptic consolidation and pruning in the brain.

---

## 2 Background

### 2.1 Low-Rank Factorization
Low‑rank parameterizations \(W = UV\) have been used for model compression, e.g., in LSTMs and Transformers. Techniques
such as LoRA add low‑rank adapters on top of pretrained dense matrices, while others retrain from scratch with reduced
rank. However, existing work either fixes rank globally or applies SVD‑based truncation after training, neither of
which exploits online feedback from learning dynamics.

### 2.2 Hebbian Learning and Eligibility Traces
In neurobiology, *eligibility traces* represent time‑decaying records of local synaptic activity that interact with
delayed reward or error signals to drive learning (Frémaux & Gerstner, 2016). The general three‑factor rule combines
pre‑synaptic activity, post‑synaptic activity, and a global modulatory factor. Analogous mechanisms have been explored
in reinforcement learning for temporal credit assignment. To date, such traces have rarely been applied to *structural*
adaptation in deep networks.

### 2.3 Adaptive Rank and Neural Plasticity
Some recent works introduce adaptive rank selection through parameter‑importance measures or learned binary gates, yet
these typically rely on magnitude or Hessian sensitivity criteria and operate only at layer granularity. Dynamic rank
expansion or shrinkage driven by online learning signals remains under‑explored.

---

## 3 Method

### 3.1 Low-Rank Parameterization
We represent each projection W in a low-rank factorized form \(W = U V\), where the inner dimension \(K\) controls
the representational capacity:

\[
W = UV = \sum_{i=1}^{K} u_i v_i^\top.
\]

Training proceeds as usual with gradients \(\partial \mathcal L / \partial U\) and \(\partial \mathcal L / \partial V\).

---

### 3.2 Hebbian Eligibility Traces
For input \(x\) and output gradient \(g_y = \partial \mathcal L / \partial y\), we define a local activity signal for
rank component \(i\):

\[
\phi_i = (u_i^\top x)\,(v_i^\top g_y).
\]

We maintain exponentially decaying traces:

\[
e_i \leftarrow \lambda e_i + \phi_i, \quad
q_i \leftarrow \lambda q_i + \phi_i^2,
\]

where \(e_i\) tracks signed usefulness and \(q_i\) tracks energy or volatility. Typical decay
\(\lambda \in [0.95, 0.995]\).

Optionally, a global performance delta \(\delta_t = L_{t-1} - L_t\) acts as a modulatory factor:

\[
c_i \leftarrow c_i + \alpha\, \delta_t\, e_i.
\]

Here \(c_i\) represents long-term credit (positive if the component’s activation correlates with improved loss).

---

### 3.3 Adaptive Rank Growth
When training loss or validation perplexity plateaus, components with high **gradient variance** or large \(|c_i|\)
indicate unresolved learning pressure. The algorithm duplicates the top-pressure components function-preservingly:

\[
U' = [U \;\; U_S], \quad
V' = \begin{bmatrix}V \\ 0\end{bmatrix},
\]

where \(S\) indexes selected components. New columns are initialized as slight perturbations of existing \(u_i,v_i\),
maintaining \(U'V' = UV\) before additional training.

---

### 3.4 Adaptive Rank Shrinkage
Conversely, when eligibility traces decay and \(|c_i|/\sqrt{q_i}\) falls below a threshold for prolonged periods,
the corresponding components are marked as *inactive*. To remove them without disrupting function:

1. **Rotation:** A small \(K\times K\) orthogonal rotation consolidates energy into leading components.  
2. **Zeroing phase:** Inactive columns/rows are zeroed while training continues briefly.  
3. **Pruning:** Columns/rows with persistently near-zero contribution are physically removed, reducing \(K\).  

This ensures shrinkage is smooth and data-driven rather than arbitrary.

A distribution-aware variant weights activity by activation and gradient covariances:

\[
I_i = \mathbb{E}\big[(u_i^\top x)^2\big]\,\mathbb{E}\big[(v_i^\top g_y)^2\big],
\]

or more generally uses diagonals of the Fisher information. Components with small \(I_i\) are considered expendable.

---

### 3.5 Optional Shared-U Parameterization

While the core DHARMA formulation treats each projection \(W_Q\), \(W_K\), \(W_V\) independently as
\(W_\bullet = U_\bullet V_\bullet\), an alternative implementation shares a common input-side projection \(U\) across
the three:
\[
W_Q = U V_Q,\quad W_K = U V_K,\quad W_V = U V_V.
\]
This reduces parameters from \(3N K + 3K N\) to roughly \(N K + 3K N\) (saving one factor of \(3\times\) on the input
side) and allows caching of \(xU\) during attention computation, since all three projections use the same intermediate
representation. The trade-off is a mild coupling of Q/K/V subspaces, which can limit expressivity but may act as a
regularizer in over-parameterized regimes. In preliminary analysis, this variant provides additional memory and
compute savings with negligible degradation for moderate K/N ratios and could be explored as a lightweight extension
of the DHARMA framework.

It is also possible to allow each projection to have its own rank, \(K_\bullet\), even with shared \(U\). In this case,
\(K_U = max(K_\bullet)\). When \(K_\bullet < K_U\), each projection \(W_\bullet\) consumes only the first \(K_\bullet\) latent components of
the shared basis, i.e., \(xW_\bullet = (xU){[:, :K_\bullet]}V_\bullet\). This allows differential rank growth while
preserving a single cached \(xU\) computation.

#### 3.5.1 Eligibility Traces for Shared \(U\)

When the same input-side projection U is shared across multiple attention projections (\(W_Q\), \(W_K\), \(W_V\)), each
column \(u_i\) contributes to three distinct outputs. Accordingly, its eligibility trace aggregates activity across
all associated pathways:
\[
e_i^{(U)} \leftarrow \lambda\, e_i^{(U)}
	•	(u_i^\top x)
\sum_{\bullet \in \{Q,K,V\}}
(v_{i\bullet}^\top g_\bullet),
\]
where \(g_\bullet = \partial \mathcal L / \partial y_\bullet\) denotes the gradient of the loss with respect to each
projection’s output.

This formulation effectively pools Hebbian updates across the attention subspaces, ensuring that the shared basis \(U\)
strengthens or weakens directions according to their aggregate utility across all Q/K/V projections.

The shared trace \(e_i^{(U)}\) governs growth and shrinkage of \(u_i\), while independent traces \(e_i^{(V_{\bullet})}\)
can guide reweighting within each projection’s output subspace.


### 3.6 Rank Saturation and Promotion to Full Rank

In rare cases, a layer may reach its maximum configured latent dimension \(K_U\) yet continue to accumulate high
eligibility or gradient variance, indicating persistent representational pressure. Rather than artificially capping
capacity, DHARMA permits a promotion step:
\[
W_{\text{dense}} \leftarrow U V,
\]
after which training proceeds with a standard dense parameterization. This transition effectively disables low-rank
constraints for that layer, acknowledging that the full ambient space \(N\) is required to model the data. The
promotion can be reversible: layers promoted to full rank may later be refactorized into low-rank form once eligibility
traces decay, providing a bidirectional pathway between efficiency and expressivity.


### 3.7 Algorithm Summary
1. Train with low-rank \(W=UV\).  
2. Maintain per-component traces \(e_i, q_i, c_i\).  
3. On plateau:  
    - **Grow:** duplicate top-pressure components (high gradient variance).  
    - **Shrink:** prune low-eligibility components.  
4. Continue training; repeat periodically.

This yields an online balance between model compression and capacity expansion.

---

## 4 Relation to Prior Work

| Theme                      | Prior Work                             | Difference                                                                                                |
|----------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Low‑rank&nbsp;adaptation   | LoRA,&nbsp;AdaLoRA,&nbsp;AROMA         | Those fix rank or adapt via gradients on rank masks; we adapt structure itself via eligibility traces     |
| Synaptic&nbsp;plasticity   | Three‑factor&nbsp;learning&nbsp;rules  | We use Hebbian traces not for parameter updates but to decide capacity allocation                         |
| Dynamic&nbsp;architecture  | Net2Net,&nbsp;grow‑and‑prune&nbsp;DNNs | Those operate at neuron/layer level; our adaptation is continuous and local within a matrix factorization |
| Compression                | SVD/Hessian&nbsp;pruning               | Offline and loss‑agnostic; ours is online, data‑ and gradient‑driven                                      |

To our knowledge, **no prior method combines Hebbian eligibility traces with adaptive low‑rank growth and shrinkage**
in modern transformer architectures.

---

## 5 Experimental Plan
We propose to evaluate DHARMA on transformer language models (e.g., GPT-style 125 M parameter
baseline) under the following conditions:

1. **Fixed low-rank baseline** — compare to static rank \(K\).  
2. **Post-hoc compression** — standard SVD truncation.  
3. **DHARMA** — with eligibility-trace driven growth/shrink.

**Metrics:**  

- Validation loss vs. compute cost (FLOPs).  
- Parameter efficiency (bits per perplexity).  
- Distribution of learned ranks per layer/head.  
- Stability under continued training and fine-tuning.  

**Ablations:**  

- Effect of trace decay \(\lambda\).  
- Hebbian term vs. gradient-variance heuristic.  
- Impact of eligibility-trace modulation (with/without \(\delta_t\)).  
- Separate vs. shared \(U\) among Q/K/V projections.

---

## 6 Discussion
The proposed mechanism provides a bridge between biological notions of synaptic efficacy and practical model
compression. Its entirely local statistics make it easily parallelizable and compatible with modern accelerators.
By letting rank evolve naturally, the model can exploit redundancy early in training and specialize later, improving
both efficiency and generalization.

Potential extensions include:

- Extending traces to nonlinear adapters (e.g., LoRA modules).  
- Combining with Fisher-weighted pruning for stronger theoretical grounding.  
- Applying to recurrent or neuromorphic architectures.

---

## 7 Conclusion
We introduced a conceptual framework for **adaptive rank selection via Hebbian eligibility traces**. The method offers
a biologically inspired path toward self-regulating model capacity, coupling long-term activity statistics with
structural adaptation. While preliminary, it suggests that ideas from neural plasticity—temporal credit assignment,
eligibility traces, and use-dependent consolidation—may have powerful analogues in optimizing artificial networks.

---

## References
- Frémaux, N., & Gerstner, W. (2016). *Neuromodulated spike‑timing‑dependent plasticity and theory of three‑factor learning rules.* Frontiers in Neural Circuits.  
- Hu, E. J., et al. (2022). *LoRA: Low‑Rank Adaptation of Large Language Models.* arXiv:2106.09685.  
- Aghajanyan, A., et al. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine‑Tuning.* ACL.  
- Chen, Y., et al. (2024). *Adaptive Rank Selections for Low‑Rank Approximation of Neural Networks.* NAACL.  
- Tay, Y., et al. (2022). *Efficient Transformers: A Survey.* ACM Computing Surveys.

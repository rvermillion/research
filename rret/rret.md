
# Read-Refreshed Eligibility Traces: Bridging BPTT and Long-Range Credit Assignment in Recurrent Transformers

**Author:** Richard Vermillion
**Date:** *2025-10-25*

---

## Abstract

Windowed back-propagation through time (BPTT) enables efficient training of recurrent architectures such as recurrent transformers but limits learning to short temporal horizons. Full BPTT captures long-range dependencies but scales linearly in both memory and compute with sequence length.  
We introduce **Read-Refreshed Eligibility Traces (RRET)** — a hybrid approach that maintains compact, usage-conditioned “credit traces” for past activations and updates them *only when those memories are retrieved* by attention.  
Inside a fixed-length window we compute exact gradients; outside it, we refresh Hebbian-style eligibility traces parameterized by the attention weights, routing sensitivities, and critic-derived postsynaptic drives.  
This yields a recurrent transformer that can learn from arbitrarily long sequences with fixed memory cost, propagating credit precisely to semantically relevant past states while ignoring unused ones.  
We derive the update rules, describe an efficient implementation, and propose experiments demonstrating that RRET matches full BPTT on long-range dependency benchmarks at a fraction of the memory cost.

---

## 1 Introduction

Standard recurrent transformers face a fundamental trade-off: **windowed BPTT** enables efficient training with bounded memory but prevents learning from gradients or rewards beyond the truncation horizon \(T\). **Full BPTT** captures those dependencies but becomes infeasible for long sequences due to quadratic attention cost and linear memory growth.

We propose **Read-Refreshed Eligibility Traces (RRET)** — a mechanism that blends exact short-range back-propagation with biologically inspired, long-range credit assignment.  
RRET maintains lightweight, synapse-like traces for past computations that are *refreshed only when the corresponding memories are read* from the key–value (KV) cache of a recurrent transformer.

When a query at time \(t\) attends to a cached key–value pair \((k_i,v_i)\) from step \(i<t\), the trace for that pair is updated according to:

- **Attention strength** \( \alpha_{t,i}\): how much the memory was used  
- **Routing sensitivity** \( \Delta_{t,i}\): how much changing its key would affect the output  
- **Postsynaptic drive** \( u_t\): a local gradient direction provided by either the true gradient (inside the window) or a learned critic

These read-time refreshes form *usage-conditioned eligibility traces* that accumulate credit exactly for the memories that matter.  
When a delayed loss or reward signal arrives, the accumulated traces serve as low-variance, temporally extended gradient estimators.

---

## 2 Method

### 2.1 Architecture Overview

RRET extends a standard recurrent transformer block with four components:

1. **Windowed unroll:** Exact BPTT is performed for the most recent \(T\) steps.  
2. **KV cache:** Each past step stores its compressed input \(\tilde x_i=P x_i\), keys \(k_i=W_k x_i\), values \(v_i=W_v x_i\), and timestamp \(\tau_i\).  
3. **Read-refreshed traces:** For cache entries older than the current window, per-head eligibility matrices \(e^{(k)}_{i,h}, e^{(v)}_{i,h}\in\mathbb{R}^{d_h\times r}\) are updated on every read.  
4. **Critic head:** A small network \( \hat r_t = f(o_t) \) predicts future returns and supplies the local Jacobian \( J_t=\partial \hat r_t/\partial o_t \), used as the postsynaptic drive \(u_t\) when true gradients are unavailable.

### 2.2 Eligibility Updates

For query \(q_t^{(h)}\) attending to item \(i\) with weight \(\alpha_{t,i}^{(h)}\):

\[
\begin{aligned}
e^{(v)}_{i,h} &\leftarrow \lambda\,e^{(v)}_{i,h}
    + \eta_v\,\alpha_{t,i}^{(h)}\, u_t^{(h)}\, \tilde x_i^\top,\\[3pt]
\Delta_{t,i}^{(h)} &= \frac{1}{\sqrt{d_h}}\,u_t^{(h)\top}(v_i^{(h)}-\tilde o_t^{(h)}),\\[3pt]
e^{(k)}_{i,h} &\leftarrow \lambda\,e^{(k)}_{i,h}
    + \eta_k\,\alpha_{t,i}^{(h)}\,\tilde\Delta_{t,i}^{(h)}\, q_t^{(h)}\,\tilde x_i^\top,
\end{aligned}
\]

where \(\tilde\Delta\) denotes the variance-reduced, normalized routing sensitivity.  
Traces decay with factor \(\lambda\in(0,1)\) and are updated only for items outside the current BPTT window.

When a scalar training signal \(\delta_t\) (e.g., TD error or per-step loss) arrives, parameter updates are:

\[
\Delta W_{v,k,q}
   = \Big(\sum_{i,h} e^{(v,k,q)}_{i,h}\Big) P^\top \,\delta_t.
\]

This structure matches the three-factor learning rule observed in biological synapses: *presynaptic activity* (\(\tilde x_i\)), *postsynaptic drive* (\(u_t\)), and a *modulatory signal* (\(\delta_t\)).

### 2.3 Compression and Memory Efficiency

To limit storage, inputs are projected to a low-rank subspace \( \tilde x_i = P x_i \), with \(P\in\mathbb{R}^{r\times d_{\text{model}}}\).  
If \(P\) is fixed (random or orthogonal), gradients flow only through \(W_{k,v,q}\); if learned, \(P\) is updated on a slower timescale.  
Each eligibility matrix is \(d_h\times r\); with \(r=d_\text{model}/4\) and top-k attention, trace memory remains \(O(n_\text{heads}\,r\,d_h\,k)\), independent of sequence length.

### 2.4 Variance-Reduction Techniques

- **Attention sharpening:** replace \(\alpha\) with \(\alpha^\beta/\sum \alpha^\beta\) (\(\beta=2\)–3).  
- **Normalization:** maintain per-head running mean/variance of \(\Delta_{t,i}\).  
- **Baselines:** subtract softmax-weighted or EMA baselines from \(\Delta\).  
- **Adaptive learning rates:** RMSprop statistics on trace increments.  
- **Overlap smoothing:** optional ramp transition for tokens crossing window boundaries.

---

## 3 Experimental Plan

### 3.1 Tasks

1. **Synthetic Long-Range Copy / Addition** – 10k-step sequences to test gradient reach.  
2. **Character-level Text Modeling** – enwik8 with long context; compare to truncated and full BPTT.  
3. **Reinforcement Learning** – delayed-reward maze or algorithmic tasks (e.g., DMLab MemoryMaze).  
4. **Long-horizon control** – recurrent world-model predicting rewards hundreds of steps ahead.

### 3.2 Baselines

- Full BPTT (upper bound)  
- Truncated BPTT (equal compute budget)  
- e-prop / Real-Time Recurrent Learning approximations  
- Feedback-alignment RNNs  
- Sparse Transformer with segment recurrence

### 3.3 Metrics

- **Loss / return** vs. sequence length  
- **Gradient reachability:** correlation of true vs. trace-based gradient estimates  
- **Memory footprint (GB)** vs. context length  
- **Wall-clock throughput**  
- **Variance of Δ and trace norms**  
- **Ablations:**  
  - No normalization / no baseline  
  - Different compression rank \(r\)  
  - Critic vs. random feedback  
  - Read-refresh disabled (time-decayed traces only)

### 3.4 Expected outcomes

RRET should:

- Match full BPTT within 1–3 % loss on long-range tasks while using \(<20 \%\) of the memory  
- Outperform truncated BPTT on horizons \(>\!T\)  
- Show stable training when critic Jacobian replaces true gradient

---

## 4 Related Work

- **Eligibility traces & e-prop** (Bellec et al., 2020; Lillicrap et al., 2020): local, three-factor rules approximating BPTT in spiking and recurrent nets.  
- **Feedback alignment / forward gradients** (Lillicrap et al., 2016; Akrout et al., 2019): random or learned local surrogates for backprop.  
- **Truncated BPTT variants** (Tallec & Ollivier 2018; Wojcik et al. 2021): bias-reduction schemes within finite windows.  
- **Recurrent transformers** (Dai et al., 2019; Hutchins et al., 2023): segment recurrence for long contexts, but still no true gradient propagation beyond the window.  
- **Neuroscience of credit assignment** (Gerstner et al., 2018): synaptic eligibility traces plus modulatory rewards as biological analogues to this mechanism.

No prior work combines read-conditioned eligibility updates with transformer-style KV caching and hybrid BPTT, making RRET a new member of this family.

---

## 5 Discussion

RRET reframes credit assignment as an *event-driven process*: memory tokens accumulate credit only when they influence future computation.  
By updating traces on retrieval rather than on every timestep, the model aligns temporal credit with semantic relevance — an efficiency principle absent from standard BPTT.

Limitations include sensitivity to critic quality and additional hyperparameters (decay λ, compression rank r).  
Future work includes multi-layer variants, stochastic-key training, and integration with differentiable memory systems or agents operating in continuous environments.

---

## 6 Conclusion

**Read-Refreshed Eligibility Traces** bridge the gap between truncated and full BPTT.  
They provide exact gradients where possible, approximate gradients where necessary, and biologically inspired long-range credit assignment everywhere else.  
This hybrid makes recurrent transformers scalable to arbitrarily long horizons while maintaining gradient flow aligned with their learned attention patterns.

---

## References

- Bellec, G. et al. (2020). *e-prop: A multi-factor learning rule for recurrent neural networks.* Nature Communications.  
- Lillicrap, T., Cownden, D., Tweed, D., Akerman, C. (2016). *Random synaptic feedback weights support error backpropagation for deep learning.* Nature Comms.  
- Akrout, M. et al. (2019). *Deep learning without weight transport.* NeurIPS.  
- Gerstner, W. et al. (2018). *Eligibility traces and plasticity on multiple timescales.* Biological Cybernetics.  
- Tallec, C., & Ollivier, Y. (2018). *Can recurrent neural networks warp time?* ICLR.  
- Dai, Z. et al. (2019). *Transformer-XL: Attentive language models beyond a fixed-length context.* ACL.  
- Hutchins, D. et al. (2023). *Recurrent memory transformers for long-horizon sequence modeling.* arXiv:2305.

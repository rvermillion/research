# Coordinate Attention:<br>Disentangling Position, Time, and Provenance from Semantic Content in Transformer Attention

**Richard Vermillion**
{: class="author" }

## Abstract

We introduce *coordinate attention*, an extension of principled attention (Vermillion, 2026) that replaces learned positional encodings such as RoPE with an explicit, structured *coordinate vector* supplied out-of-band to the attention gate. The coordinate vector encodes relative position, wall-clock time, and *token provenance* — metadata describing the origin of each token, including self/other distinction, system prompt membership, speaker identity, document identity, and sensor source. By routing coordinate information through the gate channel of principled attention, the model can suppress or promote context based on structural metadata without distorting the semantic key-query space. This disentanglement yields concrete benefits for long-context reasoning, memory systems, multi-speaker dialogue, multi-modal inputs, and adversarial robustness. Coordinate vectors for keys are provided by the training or inference harness and need not be recomputed at each layer, while query-side coordinate projections are learned as part of the standard query projection.

---

## 1. Introduction

Modern transformer architectures encode positional information by modifying the query and key vectors themselves — typically through rotary position embeddings (RoPE), which apply a position-dependent rotation to the query and key projections before computing the dot product. This is effective but entangles two fundamentally different kinds of information: *what* a token means and *where* it appears.

This entanglement has consequences. The semantic similarity between two tokens — whether they are about the same topic, refer to the same entity, carry related meaning — is computed in the same bilinear space as their positional relationship. A key that is semantically perfect but positionally distant is penalized in the same score that a key that is positionally close but semantically irrelevant is promoted. The model must learn query and key projections that simultaneously encode both roles, sacrificing capacity in each.

The problem deepens as we ask transformers to handle richer structural metadata beyond simple position. In a conversation, tokens carry speaker identity. In a retrieval-augmented system, tokens carry document provenance. In a multi-modal system, tokens carry sensor source. In an agentic system, tokens carry the distinction between self-generated content and external input. None of this metadata is naturally expressible as a rotation of the semantic embedding.

In this work, we propose *coordinate attention*, which builds on the gating mechanism of principled attention (Vermillion, 2026) to route all structural metadata — position, time, provenance — through a dedicated *coordinate vector* that feeds exclusively into the attention gate. The semantic score remains a pure function of content. The gate determines whether, given the structural relationship between query and key, the semantic match should be acted upon.

---

## 2. Background: Principled Attention

Principled attention (Vermillion, 2026) decomposes the attention logit into a semantic score and a gate:

$$
\begin{aligned}
s_{ij} &= q^s_i \cdot k^s_j && \text{(semantic score)} \\
g_{ij} &= q^g_i \cdot k^g_j && \text{(gate score)} \\
a_{ij} &= \lambda_i \, s_{ij} - \beta_i \, \mathrm{softplus}(-g_{ij}) && \text{(gated logit)}
\end{aligned}
$$

The gate can only suppress, never promote — $\mathrm{softplus}(-g_{ij})$ is always non-negative, so the gate subtracts from the semantic score. When the gate vectors are aligned ($g_{ij} \gg 0$), the suppression vanishes and the semantic score passes through. When they are misaligned ($g_{ij} \ll 0$), the logit is strongly suppressed. The parameter $\beta_i$ controls how much gate indifference (the modal case in high dimensions) penalizes the logit.

Principled attention also introduces a ground state $v_0$ with bias $\gamma_i$ and a sequence length scaling factor $\lambda_i$. For full details, see Vermillion (2026).

The key observation for this work is that the gate channel is *separate from the semantic channel*. This separation creates a natural site for injecting structural metadata: if the gate vectors encode coordinate information rather than (or in addition to) learned content features, the gate becomes a coordinate-aware filter on semantic attention.

---

## 3. The Coordinate Vector

### 3.1 Design

We define a *coordinate vector* $c_j$ for each key position $j$ and a *coordinate query projection* for each query position $i$. The coordinate vector is a structured, low-dimensional vector that encodes all non-semantic metadata about a token's identity and origin. It is composed of several sub-vectors concatenated together:

**Relative position** (12–16 dimensions). A sinusoidal encoding of the relative position between query and key, following the same design principles as classical sinusoidal position encodings but applied in the gate space rather than the semantic space. This replaces RoPE. Because the encoding lives in a dedicated subspace of the gate vector, it does not interfere with semantic similarity computation.

**Wall-clock time** (12–16 dimensions). A sinusoidal encoding of the wall-clock timestamp associated with each token, enabling the model to reason about temporal proximity independently of sequence position. Two tokens that are far apart in sequence position but close in time (e.g., interleaved streams) can have high temporal gate alignment, while tokens that are nearby in sequence but from different time periods (e.g., a remembered document inserted into context) can be temporally distinguished.

**Token provenance** (variable dimensions). A set of categorical and learned embeddings that describe the structural origin of each token:

- *Self/other* (1 dimension, binary). Whether the token was generated by the model itself or received from an external source. The inference harness tags autoregressively generated tokens with self-origin automatically.
- *System prompt flag* (1 dimension, binary). Whether the token belongs to the system prompt.
- *Thought/utterance flag* (1 dimension, binary). For self-generated tokens, whether the token is part of internal reasoning or external output — replacing structural delimiters like `<thinking>` tags with a continuous, gate-level signal.
- *Speaker identity* (learned embedding, ~4–8 dimensions). A per-speaker embedding for multi-turn dialogue, enabling the model to attend selectively by speaker without encoding speaker identity in the semantic space.
- *Document identity* (learned embedding, ~4–8 dimensions). A per-document embedding for retrieval-augmented or memory-augmented settings, allowing the model to gate attention by document provenance.
- *Sensor source* (learned embedding, ~4–8 dimensions). For multi-modal inputs, an embedding identifying the originating modality or sensor (text, vision, audio, structured data, etc.).

The total coordinate vector is approximately 40–60 dimensions — far smaller than the semantic embedding dimension, reflecting the fact that coordinate information is low-rank relative to content.

### 3.2 Key-Side Coordinates Are Out-of-Band

A critical design choice: the coordinate vector for each key is *not* computed by the model. It is provided by the training or inference harness as metadata, assembled from information that is known at tokenization time — sequence position, timestamp, speaker annotation, document boundary, system prompt demarcation, and so on.

This means coordinate vectors do not need to be recomputed at each layer. They are constructed once and injected into the gate channel at every layer, unlike RoPE which must be applied to every layer's query and key projections. This reduces computational cost and, more importantly, ensures that coordinate information is *consistent* across layers — the model cannot learn to distort positional or provenance signals at deeper layers, which preserves the interpretability and reliability of the gate.

### 3.3 Query-Side Coordinate Projections Are Learned

While key-side coordinates are fixed, the query-side gate projection $q^g_i$ is learned as part of the standard query projection. This asymmetry is deliberate: the query needs to learn *what coordinate patterns to attend to*, which varies by head and layer. An early layer might learn query projections that emphasize positional proximity (local attention), while a deeper layer might learn projections that emphasize document identity (cross-document reasoning) or speaker identity (dialogue tracking).

The query gate projection maps from the model's hidden state into the coordinate space, so it has shape $(d_\text{model}, d_\text{coord})$ — a small projection matrix per head.

---

## 4. Replacing RoPE

Rotary position embeddings encode relative position by applying a rotation to query and key vectors such that their dot product depends on the difference in their positions. This is elegant but has several drawbacks:

**Entanglement with semantics.** RoPE modifies the semantic query and key vectors directly. The rotation preserves the magnitude of the dot product but changes its direction dependence, meaning the model must learn semantic representations that are robust to position-dependent rotation. This uses capacity that could otherwise encode richer semantic distinctions.

**Fixed functional form.** RoPE encodes position as a specific family of sinusoidal rotations with geometrically spaced frequencies. The model cannot learn that position matters more for some heads than others, or that certain layers should be position-invariant — the rotation is applied uniformly.

**No natural extension to non-positional metadata.** RoPE encodes a single scalar (position) as a rotation. Extending it to encode time, speaker, document, or other metadata would require additional rotations that compound the entanglement problem.

Coordinate attention addresses all three issues. Position is encoded in a dedicated subspace of the coordinate vector using sinusoidal features, and the gate projection determines how much each head and layer cares about positional proximity. Heads that need local attention learn query projections with strong positional components; heads that need global or position-invariant attention learn projections that ignore the positional subspace. And additional metadata dimensions are added by concatenation, not by compounding rotations.

---

## 5. Provenance-Aware Attention

The provenance dimensions of the coordinate vector enable qualitatively new attention behaviors that are difficult or impossible with standard architectures.

### 5.1 Self/Other Distinction

The inference harness automatically tags each token with a binary self/other flag: tokens generated autoregressively by the model are marked as "self," while all externally provided tokens (user input, retrieved documents, system prompt, tool outputs) are marked as "other." This distinction is invisible to the semantic space but available to the gate.

This enables attention heads to specialize: some heads can learn to preferentially attend to self-generated content (useful for maintaining coherence in long generations), while others can learn to preferentially attend to external input (useful for grounding). The model does not need to infer the self/other distinction from content — it is provided as a first-class structural signal.

### 5.2 Thought vs. Utterance

For models that employ chain-of-thought or internal reasoning, the coordinate vector can distinguish "thought" tokens from "utterance" tokens without relying on structural delimiters like `<thinking>` and `</thinking>` tags. The inference harness sets the thought/utterance flag based on the generation mode, and the gate can learn to suppress or promote thought content depending on context.

This is more robust than delimiter-based approaches because the distinction is encoded in a continuous gate signal rather than in special tokens that occupy positions in the sequence and can be confounded by adversarial inputs. A model cannot be tricked into "leaking" its thoughts by manipulating token content, because the thought/utterance distinction is not *in* the token content.

### 5.3 System Prompt Separation

Tokens belonging to the system prompt are tagged with a dedicated flag in the coordinate vector. This provides a structural boundary between system instructions and user-provided content that does not depend on the model learning to distinguish them semantically.

This has direct implications for adversarial robustness. Prompt injection attacks work by embedding instruction-like content in user input that the model confuses with system-level instructions. With coordinate attention, the system prompt flag provides a gate-level signal that is set by the harness, not inferred from content. An attention head can learn to gate on this flag, attending to system instructions only when the coordinate vector confirms system-prompt provenance. Adversarial content in the user turn will have the wrong provenance tag regardless of its semantic content, making injection substantially harder.

### 5.4 Speaker Identity

In multi-turn dialogue, speaker confusion is a persistent failure mode — models sometimes attribute statements to the wrong speaker, especially in long conversations with multiple participants. Coordinate attention assigns a learned speaker embedding to each token based on speaker annotation provided by the harness.

This allows attention heads to filter by speaker: a head reasoning about "what did speaker A say about X?" can learn a query gate projection that selects for speaker A's embedding, suppressing all tokens from other speakers regardless of their semantic content. The semantic channel remains free to match on topic, while the gate handles speaker selection.

### 5.5 Document Identity and Memory

In retrieval-augmented generation and memory-augmented architectures, the context window contains tokens from multiple distinct documents — retrieved passages, cached memories, prior conversation turns. These documents have no inherent positional relationship to each other or to the current generation, yet they are concatenated into a single sequence and assigned sequential positions.

This creates a fundamental problem for position-based encodings: RoPE assigns positions based on sequence order, which imposes a false proximity structure on documents that were retrieved for content relevance, not positional adjacency. Two passages about the same topic but inserted at different points in the context will appear positionally distant, while adjacent passages about unrelated topics will appear positionally close.

Coordinate attention resolves this by assigning each document a learned embedding in the coordinate vector. Position encoding within a document reflects the token's position *within that document*, while document identity is encoded separately. This allows the model to attend within a document using positional gate features and across documents using document identity gate features, without the two interfering.

For memory systems specifically, this means that a "remembered" document inserted into context retains its internal positional structure regardless of where it appears in the sequence. The model can reason about the order of sentences within the remembered document without being confused by the absolute position at which the document was inserted.

### 5.6 Sensor Source and Multi-Modal Attention

In multi-modal architectures where vision, audio, and text tokens share a context window, the sensor source embedding allows attention heads to specialize by modality. A head performing visual grounding can learn to gate for vision-source tokens; a head performing speech understanding can gate for audio-source tokens. Cross-modal attention heads can learn gate projections that are indifferent to sensor source, allowing semantic matching across modalities.

---

## 6. Implementation Considerations

### 6.1 Compatibility with Efficient Attention

Coordinate attention is fully compatible with FlashAttention-style tiling. The coordinate vector modifies only the gate score $g_{ij}$, which enters the attention computation at the same point as in principled attention. The tiling algorithm is unchanged — the same accumulators (running max, sumexp, groundexp, weighted values) are maintained, and the only difference is how $g_{ij}$ is computed within each tile.

### 6.2 Harness Responsibilities

The training and inference harness is responsible for constructing the coordinate vector for each token. This requires:

- A tokenizer or data pipeline that annotates tokens with position, timestamp, speaker, document boundaries, and source modality.
- An inference loop that tags autoregressively generated tokens with self-origin and thought/utterance mode.
- A memory or retrieval system that provides document identity for inserted passages.

These are metadata that the harness already possesses or can trivially compute. The coordinate vector is assembled once per token and does not change across layers, so the overhead is negligible.

### 6.3 Parameter Overhead

The coordinate vector adds approximately 40–60 dimensions to the gate channel. Since the gate operates in a much lower-dimensional space than the semantic channel (which uses the full $d_\text{head}$ dimensions), the parameter overhead is modest: one additional projection matrix of shape $(d_\text{model}, d_\text{coord})$ per head for the query-side gate projection, plus learned embeddings for speaker, document, and sensor source. For a model with $d_\text{model} = 4096$ and $d_\text{coord} = 48$, this is roughly 200K parameters per head — a small fraction of the total model size.

---

## 7. Relationship to Prior Work

**RoPE and ALiBi.** Rotary position embeddings (Su et al., 2021) and ALiBi (Press et al., 2022) encode position in the attention score directly. Coordinate attention subsumes positional encoding as one component of a richer coordinate vector, routed through a separate gate channel rather than applied to the semantic score.

**Segment embeddings.** BERT-style segment embeddings (Devlin et al., 2019) add a learned embedding to the token representation to distinguish segments (e.g., sentence A vs. sentence B). This operates in the semantic space and is limited to a small number of segments. Coordinate attention provides richer provenance information in a dedicated gate channel that does not interfere with semantic representations.

**Instruction hierarchy.** Recent work on instruction hierarchy (Wallace et al., 2024) proposes training models to distinguish system-level from user-level instructions. Coordinate attention provides a structural mechanism for this distinction via the system prompt flag, making it available at the attention level rather than requiring the model to learn it from content.

**Prefix-based approaches.** Methods that prepend metadata tokens (e.g., `[SYSTEM]`, `[USER]`, `[DOC:3]`) to segments encode provenance as content in the sequence, consuming positions and requiring the model to learn to interpret these tokens. Coordinate attention encodes the same information out-of-band, preserving sequence positions for actual content.

---

## 8. Discussion

Coordinate attention represents a shift in how we think about the information available to the attention mechanism. Standard attention treats the context window as a flat sequence of embedded tokens, with position as the only structural metadata and that metadata entangled with content. Coordinate attention treats the context window as a *structured* collection of tokens, each carrying explicit metadata about its origin, timing, and role, with that metadata routed through a dedicated computational channel.

This shift has implications beyond the specific benefits discussed above. As language models are deployed in increasingly complex settings — multi-agent systems, long-running conversations with persistent memory, multi-modal perception, tool use — the amount of structural metadata available about each token grows. Coordinate attention provides a principled, extensible mechanism for making all of this metadata available to the attention mechanism without sacrificing semantic capacity.

The coordinate vector is also *interpretable* by construction. Because each dimension of the coordinate vector has a known meaning — position, time, speaker, document, source — the gate scores can be inspected to understand what structural features each attention head has learned to select for. This is a significant advantage over entangled representations where positional and semantic contributions to the attention score cannot be separated.

---

## 9. Conclusion

[TBD]

---

## References

- Vermillion, R. (2026). Principled Attention: Gated Logits, Learned Ground States, and Scale-Invariant Normalization for Transformer Attention.
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- Press, O., Smith, N. A., & Lewis, M. (2022). Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Wallace, E., et al. (2024). The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.

[TBD: Additional references]
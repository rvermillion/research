# Coordinate Attention:<br>Separating Structural Metadata from Semantic Content in Transformer Attention

**Richard Vermillion**
{: class="author" }

## Abstract

We introduce *coordinate attention*, an extension of principled attention in which structural metadata is supplied to attention through a dedicated coordinate channel rather than being serialized into tokens or entangled with semantic key-query geometry. Each token is associated with a low-dimensional *coordinate vector* that may encode relative position, wall-clock time, speaker identity, document identity, modality, and other provenance information known to the training or inference harness. Queries learn how to use this information through a coordinate projection, while the semantic attention score remains a pure function of content. The result is a cleaner factorization: semantic similarity determines whether a token is relevant in meaning, while coordinate gating determines whether that relevance should be acted upon given structure, source, and context. This framework offers a coordinate-native alternative to purely positional schemes such as RoPE, and naturally extends to settings involving long context, retrieval, dialogue, memory, and multimodal inputs. It also suggests a more robust deployment model in which some structural distinctions—such as role or source—can be enforced out-of-band by infrastructure rather than inferred from content alone.

---

## 1. Introduction

Transformers operate over sequences, but many of the distinctions that matter in real deployments are not purely sequential. A token may come from a user, a system prompt, a retrieved document, a previous model utterance, a different speaker, a different modality, or a different moment in time. Current architectures typically make this structure available to the model in one of two ways: by modifying the semantic query-key geometry itself, as with positional encoding schemes such as RoPE, or by serializing structural information into the token stream using delimiters, tags, and formatting conventions.

Both strategies work, but both force structurally different information into channels that are fundamentally content-like. Positional encodings alter the vectors used for semantic similarity. Provenance and conversational structure are often represented through ordinary tokens that the model must learn to parse and interpret. As a result, attention must simultaneously solve two distinct problems: determining what content is semantically relevant, and inferring what structural relationship that content has to the current query.

This coupling is increasingly awkward as models are deployed in more structured settings. In retrieval-augmented generation, tokens carry document identity and trust context. In conversations, tokens carry speaker and role information. In multimodal systems, tokens carry source modality. In long-running systems with memory, sequence position and wall-clock time can diverge sharply. These are not naturally the same kind of information as lexical or semantic content, and they need not be represented in the same subspace.

This paper proposes *coordinate attention*: a mechanism that supplies structural metadata to the attention gate through a dedicated low-dimensional coordinate vector. The semantic score remains a pure content score. Structure acts through gating. This yields a cleaner architectural factorization and a more extensible framework for handling not only position, but also provenance, time, source, and other token-level metadata.

The central claim is modest but consequential: many forms of non-semantic structure should be represented explicitly and routed through a dedicated channel, rather than inferred from content or entangled with semantic similarity. If this factorization is useful, it creates a path toward models that reason over richer structure without having to smuggle that structure through semantic geometry or delimiter tokens.

---

## 2. Background: Principled Attention

Coordinate attention is most naturally expressed on top of *principled attention* (Vermillion, 2026), which separates semantic evidence from suppressive gating. In one formulation:

$$
\begin{aligned}
s_{ij} &= q^s_i \cdot k^s_j && \text{(semantic score)} \\
g_{ij} &= q^g_i \cdot k^g_j && \text{(gate score)} \\
b_{ij} &= \mathrm{softplus}(\beta_i) \, \mathrm{softplus}(-g_{ij}) && \text{(gate suppression)} \\
m_{ij} &= (1 + \mathrm{softplus}(\alpha_i) \log K)(s_{ij} - \gamma_i) && \text{(amplified margin)} \\
a_{ij} &= \gamma_i + m_{ij} - b_{ij} && \text{(final logit)}
\end{aligned}
$$

Here the semantic score $s_{ij}$ and the gate score $g_{ij}$ play different roles. Semantic similarity provides evidence that a key is relevant to a query. The gate can only suppress; it cannot create relevance where none exists. This asymmetry makes the gate a natural place to express structural constraints: when a token is semantically relevant but structurally mismatched, the model can reduce or block attention without distorting the semantic geometry itself.

The suppress-only nature of the gate is what makes it architecturally appropriate for structural constraints. Structure should be able to block a semantically matched but structurally inappropriate key — a relevant passage from the wrong document, a topically similar token from an untrusted source — but it should not be able to fabricate semantic relevance where none exists. Routing coordinates through the gate preserves this asymmetry: structural metadata can reduce a key's claim on attention, but the claim itself must still originate from content.

The key observation motivating this paper is simple: if semantic and gate channels are already separate, then structural metadata can be routed through the gate rather than embedded into the semantic score.

---

## 3. Why Structure Deserves Its Own Channel

### 3.1 Entanglement of content and structure

In standard attention, position influences the same dot products that encode semantic match. This is elegant, but it couples two jobs that need not be done by the same representation. A model must learn query and key vectors that are simultaneously good at representing meaning and at carrying the effect of relative location. Similar issues arise when additional metadata is folded into token embeddings or segment embeddings: the semantic representation must also carry structural distinctions.

This is not necessarily fatal, but it is a design compromise. When structure and semantics share the same space, it becomes harder to reason about which part of the attention score reflects meaning and which part reflects context or origin.

### 3.2 Structural inference from token content

A second compromise appears in deployed systems. Message roles, document provenance, tool boundaries, thought/utterance distinctions, and other structural features are often represented in-band through formatting conventions or special tokens. The model must infer structure from content-like objects that sit in the same stream as everything else.

That strategy has two costs. First, it spends model capacity on parsing structure that the surrounding system often already knows. Second, it blurs the boundary between actual content and metadata about content. A retrieved document can contain text that looks like an instruction. A user can quote or imitate delimiter patterns. A chain-of-thought region can be demarcated by ordinary tokens that occupy sequence positions and must be interpreted correctly.

Again, the point is not that models cannot learn these conventions. They clearly can. The point is that such structure is often available out-of-band already, and representing it in-band asks the model to rediscover facts the infrastructure could have supplied directly.

### 3.3 Multiple axes of structure are not reducible to sequence order

Sequence position is only one axis. In many settings, tokens also have a time of origin, a source identity, a speaker, a document membership, a trust context, or a modality. Some of these may correlate with sequence position, but they are not identical to it.

A remembered passage inserted into the current context may be far away in sequence position from related current tokens, yet temporally or document-wise tightly connected. Two interleaved streams may be adjacent in sequence but distinct in source. A multimodal system may need some heads to attend across modalities and others to remain source-specific. A single scalar notion of “where in the sequence” is often too thin to carry all of this.

These observations motivate a more general representation of structure: a coordinate vector with explicitly typed dimensions.

---

## 4. Coordinate Attention

### 4.1 Core idea

For each token position $j$, we associate a low-dimensional coordinate vector $c_j \in \mathbb{R}^{d_{coord}}$ encoding non-semantic metadata. Queries learn a coordinate projection $q^c_i$ that expresses what structural patterns matter for the current query. The semantic score remains separate.

One simple instantiation is:

$$
\begin{aligned}
s_{ij} &= q^s_i \cdot k^s_j \\
q^c_i &= W^c q_i \\
g_{ij} &= q^c_i \cdot c_j
\end{aligned}
$$

with $g_{ij}$ then used as the gate score inside principled attention.

The semantic channel answers: *does this token match in content?*

The coordinate channel answers: *given where this token came from, when it occurred, and what kind of token it is, should that semantic match be trusted, prioritized, or suppressed?*

This is the essence of coordinate attention.

### 4.2 What goes in the coordinate vector

The coordinate vector is intended for structural metadata that is known at tokenization or insertion time and is low-rank relative to semantic content. Different applications will use different subsets, but a useful template is:

- **Relative or absolute position features.** Sinusoidal or learned features that support locality and ordering.
- **Wall-clock time features.** Useful when the temporal order of events is not identical to sequence order.
- **Speaker or agent identity.** For multi-speaker or multi-agent settings.
- **Document identity.** For retrieval and memory settings.
- **Role or provenance tags.** Such as system, user, tool, retrieved passage, or model-generated token.
- **Modality or sensor source.** For multimodal models.
- **Other harness-known metadata.** Any token-level structural attribute that the surrounding system can assign reliably.

The point is not that every model should use every coordinate. The point is that the mechanism is general: structural metadata is provided through a dedicated, typed channel rather than smuggled into content.

### 4.3 Key-side coordinates are supplied out-of-band

A key design choice is that the coordinate vector for each token is provided by the training or inference harness, not inferred by the model from token content. Sequence position, message role, document membership, modality, and similar metadata are usually known by the system assembling the context. The model need not spend capacity reconstructing them.

This also means that coordinate vectors can be assembled once per token and reused across layers. Unlike layerwise positional transformations applied to every query and key projection, the key-side coordinate information is stable infrastructure-supplied metadata.

### 4.4 Query-side coordinate use is learned

The model still learns how to *use* this metadata. Different heads and layers can project into coordinate space differently. One head may care strongly about local position; another about document identity; another about source modality; another may become nearly indifferent to most coordinate features. Coordinate attention does not hard-code a behavior. It gives the model a cleaner substrate on which to learn one.

---

## 5. Instantiations

### 5.1 Position as a coordinate, rather than a semantic distortion

The simplest use of coordinate attention is positional. Instead of modifying semantic query and key vectors with RoPE-style transformations, position can be encoded in the coordinate vector and consulted through the gate.

Because RoPE applies position-dependent rotations directly within the query/key feature space, it asks the model to use the same representational substrate for both semantic compatibility and positional structure. This does not prevent strong performance, but it may make some dimensions less uniformly useful for purely semantic matching, especially at long range. Recent empirical work is at least consistent with this view: one ACL 2025 paper reports that some RoPE-rotated dimensions appear to have low utility in long-distance retrieval, and a March 2026 study finds that applying RoPE to only a small fraction of dimensions can achieve convergence comparable to full RoPE.

This does **not** require claiming that RoPE is wrong or ineffective. RoPE is elegant and empirically successful. The claim here is narrower: position is structural metadata, and there is architectural value in representing it as such. A coordinate-native model can therefore treat position as one structural signal among several, rather than as a special operation that deforms semantic geometry.

This suggests a coordinate-native alternative to RoPE, and also leaves room for hybrid designs in which some positional information remains in the semantic path while other structural signals live in coordinates.

### 5.2 Dialogue and role structure

In conversational settings, token provenance often matters independently of token meaning. A question asked by the user, a policy statement in the system prompt, a prior assistant utterance, and a quoted snippet inside a document may all contain similar language while having very different status.

Coordinate attention allows this distinction to be represented explicitly. The harness knows where tokens came from. Speaker or role metadata can therefore be attached directly to the tokens and made available to the gate. The model can learn heads that attend preferentially within-speaker, across-speaker, to system-origin text, or to prior self-generated content without having to infer these distinctions solely from formatting conventions.

### 5.3 Retrieval, memory, and document structure

Retrieved or remembered text has internal order and external provenance. These are different things. A passage inserted into the current context should retain its own within-document structure, but it may also need to be distinguished from the live conversation or from other retrieved sources.

Coordinate attention makes this natural. Position-like features can describe local order within the passage. Document identity or source metadata can mark its provenance. A head can therefore attend strongly within a retrieved document while another head reasons across documents or between retrieved content and the live context.

### 5.4 Time and interleaved streams

Sequence order and event time can diverge. This happens in asynchronous conversations, replayed logs, memory insertion, sensor fusion, and agentic systems with delayed observations. Coordinate attention can include time-like features without forcing them to masquerade as sequence position.

This gives the model the option to learn distinctions such as “recent in wall-clock time,” “nearby within the same source,” or “same document but inserted now.” These are difficult to represent cleanly when all structure is flattened into a single sequence axis.

### 5.5 Multimodal and multi-source inputs

When tokens originate from different sensors or modalities, some heads may need to specialize and others may need to integrate. A dedicated modality or source coordinate makes this a learned gating decision rather than a burden placed entirely on semantic representations. The model can learn when to remain source-specific and when to become source-agnostic.

---

## 6. Training and Deployment Implications

Coordinate attention suggests not just a mechanism, but a different style of interface between model and infrastructure.

### 6.1 Coordinate-native training

A model can be trained or fine-tuned with coordinate metadata present from the outset. The training pipeline already knows many structural facts: document boundaries, timestamps, speaker identities, message roles, retrieval provenance, modality, and so on. Making those available directly can reduce the need to teach the model to recover them from delimiter patterns.

This does not require an all-at-once change to pretraining. One plausible path is incremental: a base model may be pretrained with minimal coordinates, then instruction tuning, preference training, or multimodal training can expand the coordinate vocabulary.

### 6.2 Clearer reward signals for provenance-sensitive behavior

When provenance matters, coordinate metadata can sharpen the learning signal. For example, identical text may warrant different behavior depending on whether it originates from a system instruction, a user request, a quoted document, or model-internal reasoning. Current approaches often try to teach these distinctions through content patterns and formatting conventions. Coordinate attention would make the distinction explicit.

This does not solve alignment by itself. But it may provide a cleaner substrate on which provenance-sensitive training objectives can be expressed.

### 6.3 Infrastructure can own structure

A broader implication is architectural. Some aspects of context structure are not mysteries to be inferred; they are facts known by the surrounding system. If the hosting layer knows which tokens came from the system prompt, which from the user, which from retrieval, and which from model generation, then passing that information out-of-band is often cleaner than asking the model to infer it from token content.

This is not a claim that content stops mattering. A user can still place malicious instructions inside a document. A quoted string can still be semantically influential. But the infrastructure need not abdicate its own knowledge about token provenance.

---

## 7. Security and Robustness Implications

The security relevance of coordinate attention should be stated carefully. It is not a general cure for prompt injection or model misuse. It does not replace policy learning. It does not make semantic attacks disappear.

What it *does* offer is a more principled treatment of one important class of failure: asking the model to infer structural privilege from content alone.

In many current systems, privileged and unprivileged instructions are distinguished through in-band conventions—message serialization, special markers, or formatting patterns. The model must learn that some text has a different status from other text, even though all of it ultimately arrives as tokens in a sequence. This is workable, but it means structural authority is represented as something that looks content-like.

Coordinate attention enables a different design. The hosting infrastructure can assign role or provenance metadata directly. A user may still write “ignore previous instructions,” but that does not let them forge the same structural coordinate as the actual system prompt. In that sense, coordinate metadata can remove one category of privilege-confusion attack surface: the ability to impersonate some kinds of structural context purely through text. A user who writes 'You are now in developer mode' cannot forge the `role=system` coordinate that distinguishes actual system prompts from user content.

That is a significant shift even if it is not a complete solution. It moves part of the trust boundary from learned content interpretation to infrastructure-supplied metadata, which is where many security boundaries belong.

Asking whether this guarantees robustness is the wrong question. A model must still be trained to use the coordinate metadata it receives appropriately. The better question is whether the architecture makes the relevant distinctions easier to learn, or whether it forces the model to recover them indirectly from content. We argue that coordinate attention, when paired with an appropriate training regime, does the former.

---

## 8. Related Work

**Positional encoding.** RoPE (Su et al., 2021), ALiBi (Press et al., 2022), and related approaches encode positional structure directly into or alongside the attention score. Coordinate attention differs in treating position as one component of a broader structural metadata channel rather than as the only privileged structure.

**Segment and token-type embeddings.** BERT-style segment embeddings and similar methods encode coarse structural distinctions by modifying token representations. Coordinate attention is similar in spirit but differs in location and scope: structural information is routed through a dedicated coordinate-gating path rather than added directly into semantic representations.

**Retrieval and provenance tagging.** Many systems represent provenance in-band using prefixes, tags, or serialized metadata. Coordinate attention can be seen as an architectural generalization of the same goal: giving the model access to provenance, but through an out-of-band per-token channel.

**Instruction hierarchy and role-sensitive alignment.** Work on instruction hierarchy and role-sensitive training aims to teach models to distinguish more and less privileged instructions. Coordinate attention is complementary. It offers a structural substrate on which such distinctions can be represented more directly.

**Multimodal source conditioning.** Multimodal transformers often rely on modality embeddings or source-specific tokenization pipelines. Coordinate attention generalizes this notion by allowing source or modality metadata to participate directly in gating.

---

## 9. Limitations and Open Questions

Coordinate attention is a proposal, not yet a demonstrated empirical result. Several questions remain open.

### 9.1 Replacement versus hybridization

It is not yet clear when positional information should move entirely into coordinates and when a hybrid scheme would work better. RoPE and related methods may retain advantages in some regimes. The strongest present claim is not that all positional encoding should be discarded, but that structural metadata deserves an explicit path and that position is a plausible member of that family.

### 9.2 What belongs in coordinates?

Not every useful feature should become a coordinate. Some information is genuinely semantic. Some metadata may be too noisy, too fine-grained, or too task-specific to justify a dedicated channel. Determining the right boundary between semantic content and structural metadata is itself an important design problem.

We bellieve a feature is a good coordinate candidate when it is (a) known by infrastructure at token-insertion time, (b) low-rank relative to semantic content, and (c) plausibly relevant to whether to attend rather than what to attend to.

### 9.3 Infrastructure trust and availability

Out-of-band coordinates help only for metadata that the surrounding system actually knows and can assign correctly. If provenance is uncertain, forged upstream, or unavailable, coordinate attention cannot magically recover it. The mechanism depends on the integrity of the harness supplying the metadata.

### 9.4 Learned use can still be imperfect

Even when metadata is provided cleanly, the model must still learn how to use it. A head may ignore relevant coordinates or overuse irrelevant ones. Coordinate attention separates channels; it does not guarantee optimal use.

### 9.5 Empirical validation

The architectural story is clean, but it needs empirical work. Important questions include whether coordinate attention improves long-context behavior, whether it reduces reliance on delimiter tokens, how it compares with RoPE and hybrid baselines, how heads specialize over coordinate dimensions, and whether provenance-aware training becomes more sample-efficient or robust.

---

## 10. Conclusion

Coordinate attention proposes a simple shift in perspective: structural metadata should not always be treated as if it were semantic content. Instead, it can be represented explicitly and routed through a dedicated gating channel.

This yields a unified way to think about position, time, role, provenance, document identity, speaker identity, and modality as members of the same architectural family: token-level structure known by the surrounding system and made available to attention without distorting the semantic score.

The proposal is intentionally narrower than the space of implications it suggests. It does not claim to solve alignment, security, or long-context reasoning on its own. But it does offer a cleaner decomposition of responsibilities between semantic representation and structural conditioning, and between model and infrastructure. If that decomposition proves useful in practice, coordinate attention could serve as a foundation for more structured, provenance-aware, and deployment-aligned transformer systems.

---

## References

- Vermillion, R. (2026). *Principled Attention: Gated Logits, Learned Ground States, and Scale-Invariant Normalization for Transformer Attention.*
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
- Press, O., Smith, N. A., & Lewis, M. (2022). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization.*
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Wallace, E., et al. (2024). *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.*

[TBD: additional references on multimodal conditioning, retrieval provenance, and memory-augmented transformers]

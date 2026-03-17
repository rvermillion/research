# Rotational Attention Layers: Geometric-Algebraic Transformations within Transformer Architectures

**Author:** Richard Vermillion  
**Date:** 2025-10-22  

---

## Abstract {: class="abstract"}

We propose a modification to standard Transformer blocks wherein the value vector output of the attention mechanism
is interpreted not as an additive residual update, but as a geometric rotation applied to the input representation.
By framing attention as a rotor in geometric algebra (GA), we leverage the “sandwich” form \( R\,x\,\tilde R \) (or
equivalently \( a\,b\,x\,b\,a \)) to perform rotation in high-dimensional embedding space without computing full
\(N\times N\) rotation matrices. In practice we fix (or optionally learn) one reference vector \(a\) and let the
attention output serve as \(b\), normalize appropriately, reflect/rotate the input \(x\), then pass through the
subsequent MLP + residual path. We hypothesize that many semantic and numeric transformations—especially analogical
and arithmetic relations—are naturally expressed as rotations rather than translations, and that providing an explicit
rotational capability can reduce depth/parameter needs, improve interpretability, and better support tasks such as
analogies and arithmetic reasoning (e.g., where numbers are embedded on helices). We detail the block design,
initialization strategy (so as not to degrade pretrained models), and propose a set of ablation and evaluation
experiments on analogy benchmarks and arithmetic tasks. We believe this rotational-layer hybrid (additive + rotational)
is a promising direction for richer inductive biases in large language models (LLMs).

## 1. Introduction

The standard Transformer architecture (Vaswani et al., 2017) uses self-attention and additive residual updates via
value vectors \(V\) combined with queries/keys. The residual update is of the form:

\[
x_{\rm out} = x_{\rm in} + \mathrm{MLP}(\mathrm{LayerNorm}(x_{\rm in} + \mathrm{SelfAttn}(x_{\rm in})))
\]

Although extremely powerful, this additive update paradigm treats attention output purely as a vector shift. We argue there is value in decoupling *directional re-orientation* from *magnitude shifts*. In particular:

- Many semantic relations (e.g., *king → queen*, *man → woman*) may behave more like rotations in embedding space: preserving norm, re-orienting direction rather than purely translating.
- Recent work on numeric reasoning in LLMs shows numbers are represented as helical embeddings and arithmetic corresponds to phase shifts (i.e., rotations) on that helix (Kantamneni & Tegmark, 2025).
- By giving the model explicit rotational capacity, we might enable more efficient representation of structured transformations, reduce reliance on additive layers alone, and improve interpretability of learned latent geometry.

We propose inserting **rotational transformer blocks** periodically within a standard architecture, so that the model has both rotational and additive operations available. Our implementation uses geometric algebra: we define a reference direction \(a\) (e.g., a fixed basis vector) and attention output \(b\) to form a rotor that rotates input \(x\). This avoids computing a full \(N\times N\) rotation matrix, maintaining computational efficiency.

## 2. Rotational Block Design

### 2.1 Geometric Algebra Primer (very brief)

In geometric algebra (GA), a *rotor* \(R\) can be used to rotate a vector \(x\) by embedding \(x\) into a sandwich product:

\[
x \mapsto R\,x\,\widetilde{R}
\]

where \(\widetilde{R}\) is the reverse (conjugate) of \(R\). For a simple rotation in a plane spanned by orthonormal
unit vectors \(a,b\), one can write:

\[
R = a\,b \Rightarrow R\,x\,\widetilde{R} = a\,b\,x\,b\,a.
\]

This performs a reflection of \(x\) in the plane perpendicular to \(b\), then a second reflection in the plane
perpendicular to \(a\), which altogether yields a rotation in the plane spanned by \(a,b\). Importantly, this uses
only vector-bivector algebra rather than constructing a full rotation matrix of size \(N\times N\).

### 2.2 Block Implementation

Let \(x \in \mathbb{R}^{d}\) (embedding dimension).

1. Compute standard self-attention output \(b = \mathrm{SelfAttn}(\mathrm{LayerNorm}(x))\).
2. Add a fixed reference direction: \(b_{\rm ref} = b + a\), where \(a\) is a fixed (or optionally learned) unit vector (e.g., basis vector \(e_0\)).
3. Normalize \(b_{\rm ref}\): \(b \leftarrow b_{\rm ref} / \|b_{\rm ref}\|\).
4. Reflect input \(x\) in direction \(b\): \( h = x - 2 (b \cdot x) b \).
5. Flip sign of the reference component \(h[a\text{-dim}] \leftarrow -h[a\text{-dim}]\).
6. Pass through MLP and residual: \( h_{\rm norm} = \mathrm{LayerNorm}(h) \), \( r = \mathrm{MLP}(h_{\rm norm}) \), \( \text{output} = h + r \).

### 2.3 Initialization for Compatibility with Pretrained Models

To avoid disrupting pretrained models:

- Initialize attention projection weights small so \(b \approx 0\).
- Then \(b + a \approx a\), and after normalization, \(b \approx a\), which makes the rotor a near-identity transformation.
- Thus, the block starts close to \(x \mapsto x\) and can be fine-tuned without destabilizing the model.

## 3. Rationale & Hypotheses

### 3.1 Semantic Analogue Transformations

Analogies often involve reorienting a subspace while preserving other properties. Rotational blocks may represent such analogies (e.g. gender, plurality) more naturally than additive residuals.

### 3.2 Numerical Reasoning and Helices

Work like Kantamneni & Tegmark (2025) shows LLMs encode numbers as helices and perform addition as rotations. Our layers directly support such transformations.

### 3.3 Additional Degree of Freedom

In additive residuals, the same update is applied to all inputs. Rotational transformations depend on \(b \cdot x\), introducing a conditional transformation capacity.

### 3.4 Integration into Standard Architectures

Our blocks can be inserted periodically (e.g. every 7 layers) into standard transformer architectures, maintaining compatibility with existing models and training regimes.

## 4. Experimental Plan

### 4.1 Model Setup

- Baseline: transformer without rotational blocks.
- Experimental: insert 4 rotational layers into a 28-layer model (total 32). Only these layers are trained initially.

### 4.2 Tasks & Datasets

**Analogy Tasks:**

- Google Analogy Dataset
- BATS (Bigger Analogy Test Set)
- SAT Analogy Questions

**Arithmetic Tasks:**

- Multi-digit addition/subtraction (2, 4, 8 digits)
- Extrapolation to 10+ and 20+ digit arithmetic (self-generated)
- RL fine-tuning (e.g., DeepSeek R1-style curriculum/self-improvement)

### 4.3 Ablations

- Fixed vs learnable reference vector \(a\)
- With vs without additive residuals
- Vary frequency/number of rotational blocks

### 4.4 Metrics

- Accuracy on analogy and arithmetic datasets
- Extrapolation generalization gap
- Training efficiency and convergence speed
- Activation norm stability
- Interpretability of rotation planes

### 4.5 Hypotheses

- Rotational layers improve analogy and arithmetic accuracy
- Generalization improves on longer-digit inputs
- Learnable \(a\) improves over fixed \(a\)
- Additive path is still useful (removal hurts)
- Rotational blocks yield more stable activation norms

## 5. Related Work

- Vaswani et al. (2017) - Transformers
- Kantamneni & Tegmark (2025) - Helical number representations
- Kobayashi et al. (2020) - Norms in attention
- Assaad et al. (2022) - Rotation-equivariant attention (VN-Transformers)
- Bounsi et al. (2024) - Neural algorithmic reasoning in transformers
- Lu & Guo (2023) - Helix encodings in transformers

## 6. Risks & Limitations

- Limited rotation planes if \(a\) is fixed
- Instability if \(b\) is poorly conditioned (mitigated via normalization)
- Interpretation in high dimensions is difficult
- Additional compute cost (though minor)

## 7. Timeline

1. Implementation and verification (Weeks 0–4)
2. Compatibility tests with frozen base (Weeks 4–6)
3. Analogy task tuning and ablations (Weeks 6–12)
4. Arithmetic extrapolation experiments (Weeks 12–20)
5. Probing and interpretability (Weeks 20–24)
6. Paper drafting and submission (Weeks 24–28)

## 8. Conclusion

We believe that enriching transformer architectures with explicit rotational capacity—via geometric-algebraic rotors built from attention output and a reference direction—offers a promising inductive bias for structured transformations, analogical reasoning, and numeric reasoning. The proposed block is efficient, integrates with existing models, and is easy to initialize near identity to avoid degradation. Through ablations and benchmark experiments we aim to test whether this approach yields tangible improvements.

## References

- Vaswani, A., et al. (2017). *Attention is All You Need*.
- Kobayashi, G., et al. (2020). *Attention is Not Only a Weight: Analyzing Transformers with Vector Norms*. EMNLP.
- Kantamneni, S., Tegmark, M. (2025). *Language Models Use Trigonometry to Do Addition*. arXiv preprint.
- Assaad, S., et al. (2022). *VN‑Transformer: Rotation‑Equivariant Attention for Vector Neurons*. NeurIPS Workshop.
- Bounsi, W., et al. (2024). *Transformers meet Neural Algorithmic Reasoners*. arXiv.
- Lu, JHJ., Guo, Q. (2023). *The Double Helix inside the NLP Transformer*.
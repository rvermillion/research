# ML Systems & Representation Research

**Richard Vermillion**
Research Engineer · ML Systems Builder
\
[richard@vermillion.nyc](mailto:richard@vermillion.nyc) · [GitHub](https://github.com/rvermillion) · [X @rivermillion](https://x.com/rivermillion) · [LinkedIn](https://linkedin.com/rvermillion)

---

## Overview

I build tools and architectures that challenge standard Transformer assumptions. My work sits at the intersection of low-level systems engineering and high-level architectural theory — implementing ideas end-to-end, from custom training frameworks down to gradient flow, across multiple hardware backends.

My research intuitions are shaped by cross-disciplinary work in geometric algebra, philosophy of mind, and dynamical systems theory. The proposals below reflect a consistent thesis: that the next generation of efficient, adaptive models will come from rethinking the primitive operations — attention scoring, memory retention, credit assignment — not from scaling the existing ones.

---

## Core Tooling

Research moves at the speed of infrastructure. These libraries are the scaffolding for everything below.

### [tensile](https://github.com/rvermillion/tensile)

A dual-backend ML framework abstracting PyTorch and MLX. Write once, train on CUDA or Apple Silicon — tensile hides the semantic gap between eager and lazy execution models, unifies optimizer abstractions across backends, and provides a composable instrumentation architecture for training diagnostics.

*Status: Published on GitHub. Training experiments validated on WikiText-103 (nanollama), confirming trajectory consistency across backends.*

### [patchlm](https://github.com/rvermillion/patchlm)

A library for applying composable, behavior-preserving architectural patches to pretrained LLMs. Every patch is initialized to identity, so you can fine-tune novel architectural ideas on top of existing weights without catastrophic divergence. Think of it as a surgical toolkit for modifying the internals of a running model.

*Status: In active development.*

---

## Research Directions

### 1. Efficient Context & Learned Forgetting

Standard Transformers pay a linear memory tax as context grows. These proposals make forgetting a first-class, learned capability — compressing context without discarding signal.

**One-Pass Forgetting** — A self-distillation framework where a lightweight "forget network" learns to predict token utility within a single forward pass, enabling dynamic KV-cache eviction. The goal: fixed-memory context processing at marginal quality cost.

**Attention Dilution Fix** — Long-context retrieval degrades because irrelevant keys accumulate probability mass under softmax. A geometric correction — centering scores and scaling by √n — preserves the signal-to-noise ratio of relevant keys as context length grows.

### 2. Non-Bilinear & Rotational Architectures

The bilinear dot-product is elegant but geometrically impoverished. These proposals explore richer similarity kernels and output transformations.

**QANA (Query-As-Network Attention)** — Reinterprets the query vector as the weights of a micro-neural network that computes nonlinear compatibility scores with keys. This replaces the fixed bilinear form with a learned, input-dependent similarity kernel — allowing attention to express relationships that dot-products cannot.

**Rotational Attention Layers** — Uses Geometric Algebra rotors to reinterpret attention output as a rotation in representation space rather than an additive residual shift. The sandwich product RxR̃ performs high-dimensional reorientation, preserving norm while enabling richer representational transformations.

### 3. Adaptive Operators & Credit Assignment

Models need better mechanisms for learning over long horizons and adapting their own internal operators online.

**RRET (Read-Refreshed Eligibility Traces)** — A hybrid BPTT/eligibility-trace approach where credit traces are updated only when memories are retrieved, aligning temporal credit assignment with semantic relevance rather than recency.

**Purpose-Driven Low-Rank Operators (PLoRO)** — A framework for learning online, adaptive low-rank subspaces structured as linear autoencoders, explicitly trained to support downstream objectives like gradient preconditioning. Early experiments show faster convergence and improved training stability compared to standard approaches.

**Head Preconditioning** — A two-phase LM head update scheme inspired by recent work on gradient bottlenecks, implemented as an instrumentation plugin (`HeadPreconditionInstrument` in `tensile.extra`). The head is preconditioned in a fast initial phase before full-model training begins, reducing early gradient pathology.

---

## Background

I've been building AI systems since 1991, when I wrote tree-based neural networks for space shuttle telemetry anomaly detection at NASA Johnson Space Center. My formal training is in aerospace engineering (Princeton BSE), and my ML/AI expertise is self-directed — built through two decades of applied work followed by a return to architecture-level research.

I believe the most productive work in AI right now happens when you hold the theory and the implementation in your head at the same time. That's what this repository is for.

---

*This is a living document. Research proposals are at varying stages of development — from formal writeups to working implementations in tensile. If any of this resonates, I'd welcome a conversation: [richard@vermillion.nyc](mailto:richard@vermillion.nyc)*
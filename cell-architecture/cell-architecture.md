# The Cell Architecture: A Predictive Coding Framework for Modular, Asynchronous Intelligence

**Richard Vermillion — Draft v0.1, March 2026**

*A living reference document. This captures the current state of thinking on an architecture for intelligent agents grounded in predictive coding, modular computation, and active perception.*

---

## 1. Overview and Motivation

Current dominant architectures for sequence modeling and generation — transformers in particular — are monolithic, synchronous, and passive. A transformer processes a fixed context window in a single forward pass, produces an output, and has no internal state between calls. It cannot think without input. It cannot act without being prompted. It cannot choose what to attend to in the world. Its computation is yoked to its input: one token in, one token out.

The Cell Architecture is a fundamentally different computational paradigm. It is:

- **Modular**: computation is distributed across self-contained units ("cells") that communicate via typed messages.
- **Recurrent**: each cell maintains persistent state across time steps ("ticks"), with working memory that gives it temporal depth.
- **Parallel**: all cells update simultaneously, reading messages from the previous tick and publishing new ones. There is no sequential cascade.
- **Asynchronous** (in the general case): cells can operate at different clock rates, with messages accumulating or decaying in slots between producer and consumer.
- **Active**: perception is not passive reception but the consequence of motor actions. The system chooses what to sense.
- **Event-based**: silence can be the message. If no error arrives, the prediction was good. If no prediction arrives, the last one still holds. Computation only happens where and when the world model breaks down. Computational cost is proportional to surprise.

The architecture is grounded in **predictive coding**, the theory (with deep roots in neuroscience, information theory, and variational inference) that intelligent systems maintain a generative model of their environment and learn by minimizing prediction error at every level of abstraction. Higher-level cells predict the states of lower-level cells; lower-level cells report back the discrepancy. This is the only currency of communication: predictions flowing down, errors flowing up.

The motivation is not to build a better language model (though the architecture can be evaluated in that regime). The motivation is to define a substrate for agents that model the world, act in it, perceive the consequences, and learn — continuously, modularly, and with the kind of flexible, multi-timescale cognition that current architectures lack.

---

## 2. Core Abstractions

The architecture consists of **cells** connected by **edges** in a directed acyclic graph (DAG). While complex topologies are contemplated (multi-modal, branching, agentic), the simplest realization is a **linear stack**.

### 2.1 Cells

A cell $C_i$ is a self-contained computational unit. It maintains:

- A **state vector** $s^t_i \in \mathbb{R}^{d_i}$ at each tick $t$.
- A **working memory** $m^t_i$ (a fixed-size key-value buffer; see Section 6).
- Learned parameters for its update function, prediction heads, and memory operations.

Each cell has a dimensionality $d_i$ that may differ from other cells. The state vector, all incoming predictions, and all outgoing errors for cell $i$ share this dimensionality.

### 2.2 Edges

Cells are connected by three types of edges:

- **Prediction edges** $p_{i,j}$: cell $i$ predicts the state of cell $j$. Predictions flow "downward" (from more abstract to more concrete).
- **Error edges** $e_{i,j}$: cell $i$ reports the prediction error to cell $j$. Errors flow "upward" (from more concrete to more abstract). Every prediction edge $p_{j,i}$ has a corresponding error edge $e_{i,j}$ flowing in the opposite direction.
- **State (self) edges** $s_{i,i}$: the recurrent connection from a cell to itself. This is simply the state vector $s^t_i$ persisting (with update) from tick to tick.

### 2.3 Edge Values

Each edge carries a vector value at each tick. We use superscripts for time:

- $p^t_{i,j} \in \mathbb{R}^{d_j}$: the prediction that cell $i$ makes about cell $j$'s state, published at tick $t$.
- $e^t_{i,j} \in \mathbb{R}^{d_i}$: the error that cell $i$ computes and sends to cell $j$, published at tick $t$.
- $s^t_i \in \mathbb{R}^{d_i}$: cell $i$'s state at tick $t$.

Note the dimensionality convention: predictions $p_{i,j}$ live in the target cell's space ($\mathbb{R}^{d_j}$), while errors $e_{i,j}$ live in the source cell's space ($\mathbb{R}^{d_i}$).

---

## 3. Notation and Conventions

| Symbol | Meaning |
|--------|---------|
| $C_i$ | Cell $i$ |
| $d_i$ | Dimensionality of cell $i$'s state |
| $s^t_i$ | State of cell $i$ at tick $t$ |
| $p^t_{i,j}$ | Prediction from cell $i$ about cell $j$, at tick $t$ |
| $e^t_{i,j}$ | Error from cell $i$ sent to cell $j$, at tick $t$ |
| $m^t_i$ | Working memory of cell $i$ at tick $t$ |
| $p^t_{*,i}$ | All incoming predictions to cell $i$ at tick $t$ |
| $e^t_{*,i}$ | All incoming errors to cell $i$ at tick $t$ |

**Time convention**: a cell's state at tick $t+1$ is a function of tick-$t$ quantities. Predictions are computed from the current state at the same tick. Errors compare the *previous tick's* prediction against the *current tick's* state. All cells update in parallel; there are no within-tick dependencies between cells.

---

## 4. Update Rules

The dynamics of the system are governed by four equations, evaluated in parallel across all cells at each tick.

### 4.1 State Update

$$s^{t+1}_i = s^t_i + g_i(s^t_i,\; p^t_{*,i},\; e^t_{*,i},\; m^t_i)$$

The state update is a **residual**: the cell computes a delta from its current state, incoming predictions, incoming errors, and working memory (see Section 6 for the internal structure of $g_i$). The residual form ensures that the default behavior is persistence — a cell's state is stable unless there is reason to change it.

### 4.2 Prediction

$$p^t_{i,j} = h_j^{(i)}(s^t_i)$$

Each cell $i$ has a **separate prediction head** $h_j^{(i)}$ for each cell $j$ that it predicts. The prediction is a function of the current state, published at the same tick. Prediction heads are MLPs (with normalization; details TBD).

### 4.3 Error

$$e^t_{i,j} = f(p^{t-1}_{j,i},\; s^t_i)$$

The error compares what cell $j$ predicted about cell $i$ *last tick* against cell $i$'s *current* state. In the simplest form, $f$ is the difference $p^{t-1}_{j,i} - s^t_i$. More complex error functions (sparse, precision-weighted; see Section 7) are contemplated.

The error computation is **uniform across the stack**. The only asymmetry is in how $s^t_i$ is determined (exogenous for sensory cells, learned for interior cells), not in how the error is calculated.

### 4.4 Memory Update

$$m^{t+1}_i = f_{\text{mem}}^{(i)}(m^t_i,\; s^t_i,\; p^t_{*,i},\; e^t_{*,i})$$

Memory updates in parallel with state, consuming the same tick-$t$ inputs. See Section 6 for the structure of $f_{\text{mem}}$.

### 4.5 Execution Model

All four equations are evaluated **in parallel across all cells**. At tick $t$, every cell reads only tick-$(t-1)$ state values from other cells, mediated through predictions and errors computed as intermediate quantities. There is no within-tick cascade from top to bottom and back. Each cell operates as a self-contained unit with an **inbox** (incoming predictions, incoming errors, prior state, memory) and an **outbox** (new predictions, new errors, updated state, updated memory).

This parallel execution model is a deliberate design choice. It ensures:

1. **No sequential bottleneck**: computation scales with the number of cells, not the depth of the graph.
2. **Compositionality**: cells can be added, removed, or rewired without changing the execution model.
3. **Compatibility with asynchronous operation**: because cells only read from slots, those slots can be updated at different rates (see Section 11).

---

## 5. Cell Types

### 5.1 Interior Cells

An interior cell $C_i$ has both incoming and outgoing edges in both directions. It receives predictions from above, errors from below, and sends predictions downward and errors upward. Its state is a learned recurrent function of its inputs (Equation 4.1).

In a linear stack, an interior cell has exactly two incoming edges (one prediction from above, one error from below) and two outgoing edges (one prediction downward, one error upward), plus its self-edge. In a branching graph, a cell may have multiple prediction and error edges in either direction.

### 5.2 Sensory Cells (S-Cells)

An S-cell is a bottom cell whose state is **clamped to external input**. Its state $s^t_0$ is an embedding of sensory data (a token embedding, patch embedding, audio frame, joint position, etc.). The S-cell performs minimal computation: it compares the incoming prediction against its perceived state and sends the error upward.

$$e^{t}_{0,j} = f(p^{t-1}_{j,0},\; s^t_0)$$

S-cells have no outgoing prediction edges (there is nothing below them to predict) and no meaningful state update function (their state is exogenous).

**Crucially, the input that an S-cell perceives is generally conditioned on motor actions** (see Section 10). The "next token" is only the next token because a motor action chose to read it. A visual input is only that particular view because a motor action directed gaze there.

### 5.3 Motor Cells (M-Cells)

An M-cell is a bottom cell whose state **drives an action** in the environment. Where an S-cell's state is determined by perception (the world writes to it), an M-cell's state is determined by the prediction from the cell above it (the model writes to the world through it).

$$s^t_{\text{motor}} = \text{constrain}(p^{t}_{j, \text{motor}})$$

The constraint function may include clipping, rate-limiting, safety bounds, or discretization (e.g., sampling from a token distribution). The M-cell's state *is* the motor command.

M-cells are **causal dead ends** internally. Their state exits the system into the environment and does not return along the same edge. There is no meaningful error edge from an M-cell in the usual sense — the "error" for a motor prediction is tied to the proprioceptive consequence (see below).

### 5.4 The Sensorimotor Loop

An M-cell and a proprioceptive S-cell are typically connected to a common parent (or ancestor) cell, but **not directly to each other**. The causal link between them passes through the environment:

$$C_{\text{parent}} \xrightarrow{p} M\text{-cell} \rightarrow \text{environment} \rightarrow S\text{-cell} \xrightarrow{e} C_{\text{parent}}$$

The parent cell generates a **joint prediction**: "I will issue motor command $X$ and I will perceive consequence $Y$." The proprioceptive S-cell's error tells the parent whether the consequence matched the prediction. Over time, the parent learns a forward model of the environment — the relationship between actions and their sensory consequences — without any explicit model of the environment's dynamics.

This means:

- The system must **learn** the action-consequence relationship through experience.
- Motor learning is driven by **proprioceptive prediction error**, not by direct gradient from the motor output.
- The architecture does not need to know anything about the environment's physics; it only needs S-cells and M-cells wired to the same parent structure.

### 5.5 Top Cells

A top cell has no incoming prediction and sends no outgoing error. It is driven purely by error signals propagating up from below, plus its own recurrence and memory. It is the most abstract layer in the stack — it generates top-down expectations but never has its own expectations violated by a higher cell.

$$s^{t+1}_{\text{top}} = s^t_{\text{top}} + g_{\text{top}}(s^t_{\text{top}},\; e^t_{*,\text{top}},\; m^t_{\text{top}})$$

### 5.6 Boundary Conditions (Linear Stack Example)

For a three-cell linear stack $C_0$ (sensory), $C_1$ (interior), $C_2$ (top):

| Cell | Incoming | Outgoing | State determined by |
|------|----------|----------|---------------------|
| $C_0$ (S-cell) | $p_{1,0}$ | $e_{0,1}$ | Exogenous (sensory embedding) |
| $C_1$ (interior) | $p_{2,1}$, $e_{0,1}$ | $p_{1,0}$, $e_{1,2}$ | Learned: $g_1(s^t_1, p^t_{2,1}, e^t_{0,1}, m^t_1)$ |
| $C_2$ (top) | $e_{1,2}$ | $p_{2,1}$ | Learned: $g_2(s^t_2, e^t_{1,2}, m^t_2)$ |

---

## 6. Working Memory and Attention

### 6.1 The State Update as Attention

The state update function $g_i$ is implemented as an **attention mechanism over a working memory buffer**. Each cell maintains a fixed-size key-value buffer $m^t_i$ containing entries derived from recent ticks of its own computational history.

At each tick:

1. The current inputs — $s^t_i$, $p^t_{*,i}$, $e^t_{*,i}$ — are projected into **keys** and **values** that are appended to the working memory buffer (with eviction of old entries as needed).
2. The same inputs are projected into **queries** that attend over the full buffer via multi-head attention.
3. The attention output is passed through a projection and MLP to produce the residual state update.

This means the cell's state transition is **not Markov in the state alone** — it has access to a window of its own recent history. The cell can recognize temporal patterns: repeated errors, converging predictions, oscillating states.

### 6.2 Projection Structure

Whether the projections from state, predictions, and errors into query/key/value space are shared (a single projection over the concatenation) or structured (separate projections per input type) is an open design choice. The structured variant enforces that the model can easily distinguish "this key came from a prediction" versus "this key came from an error" without having to learn that from content alone.

### 6.3 Coordinate Metadata

Entries in the working memory buffer are tagged with **coordinate metadata**: at minimum, the tick number and the provenance (state, prediction from cell $j$, error to cell $k$). This metadata is encoded into the keys and queries, allowing the attention mechanism to condition on *when* and *where* information came from. This follows the design philosophy of **making emergent behavior easy**: the model must learn how to *use* the metadata, but it doesn't have to *infer* it from patterns in the content stream.

Standard positional encodings (e.g., RoPE on tick indices) are likely insufficient, especially once long-term memory and multi-rate clocks are introduced. Richer coordinate attention schemes — encoding wall-clock time, modality provenance, cell identity — are anticipated.

### 6.4 Eviction Policy

The working memory buffer is **fixed-size**. When the buffer is full, entries must be evicted. Candidate policies include:

- Oldest-first (simple FIFO).
- Lowest-attention-weight (learned importance).
- Compression: summarize old entries into fewer, denser key-value pairs.

The eviction policy is an open research question. A learned policy is preferred, as it lets the cell decide what to remember.

---

## 7. Rich Messages and Precision

### 7.1 Beyond Point Predictions

In the base architecture, predictions and errors are point vectors in $\mathbb{R}^{d_i}$. However, the edges may optionally carry **precision metadata**: additional structure encoding confidence, variance, or uncertainty.

A prediction $p^t_{i,j}$ might carry:

- A **point estimate** $\mu^t_{i,j} \in \mathbb{R}^{d_j}$ (the predicted state).
- A **variance structure** $\sigma^t_{i,j} \in \mathbb{R}^{d_j}$ (a diagonal encoding of precision/confidence).

An error $e^t_{i,j}$ might carry:

- The **residual** $\delta^t_{i,j} = p^{t-1}_{j,i} - s^t_i$.
- An estimate of **error reliability**: whether the error reflects a genuinely bad prediction or noisy state (inferred from recent error variance).

### 7.2 Precision-Weighted Errors

If precision metadata is available, the error computation can be precision-weighted:

$$e^t_{i,j} = f(p^{t-1}_{j,i},\; s^t_i,\; \sigma^{t-1}_{j,i})$$

A large residual on a low-confidence prediction is less surprising than a small residual on a high-confidence one. The error signal becomes "how surprised should you actually be" rather than "how far off were you."

This connects directly to the predictive coding literature where precision weighting arbitrates between top-down priors and bottom-up evidence.

### 7.3 Design Philosophy

The approach to precision follows the architecture's general principle: **provide scaffolding that makes the desired behavior easy to learn, but don't over-constrain it.** An explicit variance diagonal gives the model a leg up. But the attention mechanism over working memory (Section 6) can learn richer notions of confidence, reliability, and noise from temporal patterns in the explicit metadata — and from patterns that go beyond what any hand-designed structure encodes.

---

## 8. Multi-Tier Memory

### 8.1 The Memory Hierarchy

Working memory (Section 6) is a fast, fixed-size buffer. For longer-term storage and retrieval, a **multi-tier memory hierarchy** is contemplated:

- **Tier 1 (Working Memory)**: the KV buffer described above. Fast, small, always available. Updated every tick.
- **Tier 2+ (Long-Term Memory)**: larger, slower stores of precomputed key-value pairs. Queried asynchronously based on recent working-memory queries. Results are loaded into working memory when relevant.

### 8.2 Representation-Specific Storage

Long-term memory is **per-cell and representation-specific**. Stored key-value pairs live in cell $i$'s representation space ($\mathbb{R}^{d_i}$), tagged with coordinate metadata (timestamps, provenance, modality). You cannot dump raw sensory tokens into a higher cell's long-term memory; the stored representations must be in that cell's learned abstract space.

### 8.3 Asynchronous Loading

Long-term memory queries run **asynchronously**, decoupled from the main tick loop. An auxiliary process monitors recent queries from working memory, searches the long-term store for relevant entries, and loads them into the working memory buffer. This is analogous to an L1/L2 cache hierarchy: working memory is L1 (fast, small, always checked), long-term memory is L2 (slower, larger, loaded on demand).

The details of the long-term memory system — storage format, indexing, retrieval algorithm, the interface between tiers — are open research questions.

---

## 9. Loss and Training

### 9.1 The Loss Function

The loss at each tick is the sum of per-edge error terms:

$$\mathcal{L}^t = \sum_{(i,j) \in \mathcal{E}} \text{loss}(e^t_{i,j})$$

where $\mathcal{E}$ is the set of all error edges in the graph. The simplest choice for $\text{loss}$ is the squared norm $\|e^t_{i,j}\|^2$, but other choices (L1, Huber, precision-weighted) are possible.

This means:

- The system's sole objective is to **predict well at every level**.
- The loss decomposes into **per-edge, per-tick terms** — there is no separate reconstruction loss, contrastive objective, or auxiliary task.
- Each interior cell has access to (at least) two local loss signals: the error it sends upward (how well it was predicted) and the error it receives from below (how well it predicted).

### 9.2 The Training Problem

The true gradients of $\mathcal{L}^t$ with respect to any cell's parameters are deeply entangled across the full graph and across time (through recurrence and memory). Full backpropagation through time (BPTT) would give exact gradients but is expensive, memory-intensive, and biologically implausible.

The architecture is designed to support a **spectrum of training approaches**, from cheap-and-approximate to expensive-and-exact:

**Local backprop with detached messages.** Each cell treats incoming predictions and errors as fixed inputs (no gradient flows across cell boundaries). The cell backpropagates only through its own parameters: $g_i$, $h_j^{(i)}$, $f_{\text{mem}}^{(i)}$. This is cheap, parallelizable, and gives each cell a clear local learning signal. But it is **myopic**: a cell cannot learn to request better inputs by shaping the errors it sends, and a cell cannot learn how its predictions are used downstream.

**Eligibility traces with global error modulation.** Each cell maintains local traces of "what parameter changes would have changed my output." The global scalar $\mathcal{L}^t$ (or its temporal difference) modulates these traces — a REINFORCE-like signal saying "whatever you just did, do more or less of it." This is much cheaper than BPTT but provides some credit assignment across cell boundaries. This approach is particularly relevant for M-cell motor prediction heads, which have no direct local loss and *require* a non-local signal.

**Low-rank gradient approximations.** Cells could pass compressed or low-rank approximations of gradient information along edges, providing richer inter-cell credit assignment than a scalar signal without the full cost of BPTT.

**Full BPTT (upper bound).** Unrolling the computation graph across ticks and cells and backpropagating through everything. Useful as a benchmark to measure how much the approximate methods leave on the table, but not a target for the production system.

### 9.3 The M-Cell Gradient Problem

M-cells present a specific training challenge. The motor prediction head projects into an environmental sink — there is no error edge returning from the M-cell. The only signal about whether a motor command was *correct* arrives indirectly, through the proprioceptive S-cell's error, which reaches the parent cell but not the motor head directly.

Under local backprop, the motor head receives **no gradient at all**. This makes eligibility traces with global (or at least parent-local) error modulation a likely *necessary* component of the training regime for motor learning, not merely a nice-to-have.

### 9.4 Open Questions

The choice of training method is the architecture's **central open research question**. The design philosophy is to characterize the failure modes of simple methods first, then let those failures guide the introduction of more sophisticated signals — analogous to how biological evolution may have developed neuromodulatory systems in response to optimization pressures.

Candidate mechanisms to explore include:

- Hebbian-like eligibility traces modulated by global or semi-local error signals.
- Neuromodulatory "broadcast" signals (analogous to dopamine) that modulate learning rates or trace decay across cells.
- Inter-cell salience signals that allow cells to flag important state changes.
- EMA-smoothed gradient estimates passed along edges.
- Hybrid approaches that use BPTT over short windows within a cell but approximate signals between cells.

---

## 10. Active Perception

### 10.1 Perception as Action

In this architecture, **all perception is action-conditioned**. Every sensory input is the consequence of a motor decision, even if that decision is the default "continue doing what you were doing."

A token sensor does not passively receive the next token in a stream. Something — an M-cell — chose to *read* that token rather than skip backward, pause, or switch to a different input source. A visual sensor does not passively receive an image. Something chose to look in that direction, at that distance, with that focus.

This means the parent cell that predicts both the motor action and its sensory consequence is always making a **joint prediction**: "given that I do $X$, I will perceive $Y$." The sensory error tells it whether the joint prediction was correct.

### 10.2 Examples

**Language**: An M-cell outputs a "read action" (read-next-token, skip-back-$n$-tokens, pause). The S-cell perceives whatever token is at the resulting position. The parent cell must learn that the token it predicts depends on the action it chose.

**Vision**: An M-cell controls gaze direction (saccade, head turn, camera selection). The S-cell perceives the resulting visual field. The parent cell must learn the relationship between gaze commands and visual input.

**Embodiment**: M-cells control actuators (joint torques, gripper force). Proprioceptive S-cells sense the resulting body state (joint positions, contact forces). The parent cell must learn forward dynamics.

### 10.3 Minimizing Long-Term Surprisal

A system that minimizes only *immediate* prediction error can learn to avoid surprise trivially — by never looking, never acting, never gathering information. But a system that minimizes surprisal over a **longer horizon** (enabled by working memory and multi-tier memory giving temporal depth) must be *curious*: it must seek out information now that reduces future surprisal, even if looking introduces short-term prediction errors.

This is **active inference**: the agent acts to gather information that improves its world model, not merely to confirm what it already believes. The behavior emerges from the architecture's structure — cells with temporal depth optimizing over longer horizons will naturally learn to "look left before crossing the street" because the short-term surprise of seeing (or not seeing) a bus is far less costly than the long-term surprise of being hit by one.

### 10.4 The "Quiet" Mechanism

M-cells need a **quiet option**: the ability to produce no output. For a language M-cell, this could be a special "silence" token in the output distribution, or a binary gate with a learned probability of speaking versus not speaking.

This is important because it means the system must learn **when to act**, not just how. During quiet ticks, the rest of the cell stack continues to run — higher cells are predicting, receiving errors, updating state and memory. The system is *thinking*. When it finally does act, the action is informed by however many ticks of internal processing occurred since the last action. The depth of "thought" is **dynamic**, not fixed.

This also opens the door to **internal simulation**: during quiet ticks, the system may run predictions about what it *would* perceive under various actions, using its learned world model to evaluate options before committing. That is planning.

---

## 11. Extensions and Open Questions

### 11.1 Multi-Rate Clocks

In the general case, cells may operate at **different clock rates**. A low-level sensory cell might tick at high frequency (processing raw audio frames), while a high-level abstract cell might tick much more slowly (updating a world model). Communication is handled via message **slots**:

- A faster cell's errors accumulate in the slower cell's inbox (e.g., as an exponential moving average) until the slower cell is ready to process them.
- A slower cell's predictions persist (possibly decaying) in the faster cell's inbox between updates.

The cell interface — typed vectors in slots — is invariant to clock rate. A cell does not need to know whether its neighbor is running at the same rate or ten times slower; it simply reads whatever is in the slot.

### 11.2 Multi-Modal Merge

For multi-modal processing, separate linear stacks handle each modality (vision, language, audio, proprioception). At some level, these stacks **merge** into a cell that predicts the top of each modality stack through separate prediction heads $h_j^{(i)}$, one per child.

The merge cell is forced to build a representation rich enough to predict *all* child modalities through its single state vector. This is a **bottleneck argument**: the state must capture whatever structure is common across modalities. The result is an emergent multi-modal representation — not one that was explicitly designed, but one that arose because it was necessary for prediction.

### 11.3 Asynchronous and Distributed Operation

Because cells communicate only via message slots and update in parallel, the architecture is naturally compatible with **distributed execution**. Cells can run on different devices, with message slots implemented as network buffers. Latency in message delivery is equivalent to a clock-rate mismatch — the receiving cell simply works with the most recent available message.

### 11.4 Neuromodulatory Signals

Beyond prediction and error, the architecture may benefit from additional message types:

- **Salience signals**: a cell flags that its state has changed significantly, prompting neighbors to attend.
- **Reward/dopamine analogs**: a global or semi-global scalar broadcast that modulates learning rates or eligibility trace decay.
- **Arousal/gain signals**: modulating the overall sensitivity or update magnitude of a cell.

These are speculative extensions, to be explored if and when the failure modes of simpler training methods demand them.

### 11.5 Lagged Predictions

In the base architecture, $p^t_{i,j}$ predicts cell $j$'s state at the next tick. A generalization allows a cell to predict a lower cell's state **$T$ ticks in the future**, where $T$ may vary by level. Higher cells, operating more abstractly, might predict further ahead. This introduces additional timing complexity but could improve credit assignment and enable hierarchical temporal abstraction.

---

## 12. Readout for Practical Experiments

### 12.1 The Short-Term Expedient

For initial experiments (e.g., language modeling benchmarks to verify that the architecture can learn at all), a pragmatic readout can be implemented:

- Use a standard S-cell with token embeddings as input.
- Attach a conventional **LM head** (a linear projection to vocabulary logits) to the S-cell's incoming prediction $p^t_{1,0}$, or to the interior cell's state $s^t_1$.
- Treat the system as a next-token predictor and train with cross-entropy loss on the LM head output, in addition to (or instead of) the per-edge prediction error loss.

This is **not** the intended long-term architecture. It is a scaffolding for validation: can the cell stack learn useful representations? Do the predictions converge? Does working memory help?

### 12.2 The Long-Term Vision

In the full architecture, generation is driven by M-cells (Section 5.3). A language M-cell outputs a token distribution; a proprioceptive S-cell reads back the generated token. The system learns to speak by learning the joint prediction of action and consequence.

The transition from the short-term LM-head readout to the full M-cell generation architecture is itself a research milestone.

---

## Appendix A: Summary of Open Research Questions

1. **Training regime**: local backprop, eligibility traces, gradient approximations, or hybrids?
2. **Working memory eviction**: learned vs. heuristic, compression strategies.
3. **Long-term memory**: storage format, indexing, retrieval, tier interfaces.
4. **Precision structure**: how much to make explicit vs. let emerge?
5. **Coordinate attention**: encoding schemes for time, provenance, modality.
6. **Error function**: simple difference, sparse, precision-weighted, or learned?
7. **Prediction head architecture**: MLP depth, normalization, shared vs. per-target.
8. **M-cell training**: how to get gradient signal to motor prediction heads.
9. **Multi-rate scheduling**: EMA, decay, accumulation strategies.
10. **Quiet mechanism**: gate vs. special token, learned timing.
11. **Active perception actions**: discrete vs. continuous, action space design.
12. **Loss weighting**: uniform across edges, level-dependent, precision-modulated?
13. **Initialization and warmup**: how does the system bootstrap from random states?
14. **Lagged predictions**: which cells predict how far ahead?

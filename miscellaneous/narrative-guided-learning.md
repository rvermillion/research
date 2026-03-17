# Research Program  
**Richard Vermillion**  
{: class="author" }
  
**From Corpora to Experience: A Research Program for Provenance-Aware, Narrative-Guided Learning in AI**  
  
**Abstract**  
  
Scaling language-model training by adding more tokens is hitting diminishing returns: financially, environmentally, and epistemically. The core issue is not just data quantity but **data shape**. Today’s models ingest trillions of largely context-free tokens drawn from heterogeneous sources with weak provenance, minimal grounding, and no persistent continuity of self. Humans, by contrast, learn from far fewer “tokens,” but those tokens are embedded in **situations**: multimodal context, known or inferred sources, social intent, affect, agency, and consequences. This position paper proposes a direction for progress: shift from training on “dead corpora” toward training on **structured experience**, by giving models (1) explicit provenance inference, (2) active perception and agency beyond the token stream, (3) persistent memory, and (4) a slowly evolving narrative/self-model that modulates learning—potentially all the way down to **activation-conditioned optimizer dynamics**. The aim is not anthropomorphism; it is engineering a stable, data-efficient learning system that can resist “sewage” and exploit high-context signal.  
  
⸻  
  
**1. The Data Problem Is a Context Problem**  
  
The current paradigm treats training data as an undifferentiated stream of text. Even where metadata exists, it is rarely integrated as a first-class part of the learning objective. This creates multiple failures:  
	•	**Source collapse**: high-quality and low-quality text produce gradients of similar “type,” encouraging models to learn style and plausibility without robust epistemics.  
	•	**Epistemic inconsistency**: models vacillate between confident misinformation and hedged truth because they lack stable internal mechanisms for reliability weighting.  
	•	**Prompt injection vulnerability**: when “who is speaking” is not structurally represented, malicious instructions can masquerade as authoritative context.  
	•	**Financial scaling wall**: adding more data increasingly means adding more low-signal, adversarial, duplicated, or misaligned content.  
  
The core claim: **data scale is not the main bottleneck; contextual structure is**. The next gains come from representing and exploiting structure that already exists (provenance, intent, verifiability, modality, continuity), and from generating new structured signal via active interaction.  
  
⸻  
  
**2. Corpora vs. Environments: The Missing Ingredient Is Closed-Loop Consequence**  
  
A static corpus does not “push back.” It cannot enforce calibration, cannot penalize self-sealing beliefs, and cannot demand long-horizon consistency. In an environment, the learner acts and must reconcile predictions with outcomes.  
  
This matters even for “language” competence, because language is intrinsically social and instrumental:  
	•	utterances have speakers,  
	•	speakers have incentives and reliability profiles,  
	•	and statements imply actions and consequences.  
  
The proposal is not that every model must be embodied in a robot. It is that training must include **externally verifiable feedback**—tool calls, executable tasks, interactive environments, and multi-step objectives—so that internal narratives cannot optimize merely for coherence or appearance.  
  
⸻  
  
**3. Provenance as a First-Class Latent Variable**  
  
Instead of treating provenance as a dataset curation step, treat it as something the model **actively infers**:  
  
**Provenance inference head(s)** predict structured attributes at the token/span level:  
	•	source identity/class (author/site/domain/tier)  
	•	discourse mode (fiction, argument, measurement, instruction, persuasion)  
	•	intent and adversarial likelihood  
	•	epistemic status (claim, evidence, citation, observation)  
	•	verifiability (tool-checkable vs not)  
	•	expected reliability / calibration prior  
  
Bootstrapping paths:  
	•	start with weak supervision from metadata (where available),  
	•	augment with synthetic mixing (known-good + known-bad domains),  
	•	self-supervise via consistency checks (e.g., contradiction detection under retrieval, citation-pattern modeling, cross-source agreement),  
	•	and refine with external verification loops (tools/environments).  
  
This directly addresses the “sewage” problem: the model learns *what kind of thing it is reading*.  
  
⸻  
  
**4. Learning Should Be Conditioned on What the Model Thinks It Is Seeing**  
  
A central idea: once the model has a representation of provenance and reliability, it should influence **how** learning happens—possibly at the optimizer level.  
  
**4.1 Gated learning (safe baseline)**  
  
The simplest mechanism: loss weighting.  
	•	Each token/span gets a weight w_t derived from provenance and internal activations.  
	•	Constraints prevent collapse:  
	•	enforce a batch “weight budget” (mean weight fixed),  
	•	cap w_t in a bounded range,  
	•	add entropy regularization (avoid zeroing whole regions),  
	•	periodically force uniform weights (exploration / anti-self-sealing).  
  
This is already a major shift: the model’s beliefs about the data modulate gradient flow.  
  
**4.2 Activation-conditioned optimizer modulation (research frontier)**  
  
More ambitious: use activations to produce optimizer parameters:  
	•	per-layer learning-rate multipliers,  
	•	momentum/EMA tuning,  
	•	or low-rank preconditioning factors (Woodbury-friendly forms).  
  
One concrete design:  
	•	Add an auxiliary head producing low-rank factors U_\ell per layer.  
	•	Define a near-identity preconditioner P_\ell = I + U_\ell U_\ell^\top.  
	•	Regularize toward stability: bounded norm, smoothness over steps, whitening-ish constraints on U_\ell.  
	•	Train the optimizer-head on a slower time-scale with a meta-objective tied to *trajectory quality* (generalization improvement, calibration, robustness), not immediate training loss.  
  
This makes the learner **adaptive in its update geometry**, not just its representations.  
  
⸻  
  
**4.5 Active Curriculum: State-Dependent Navigation of the Training Corpus**  
  
Current pretraining pipelines treat the corpus as static and externally ordered. Even where curriculum learning is applied, ordering is heuristic and decoupled from the internal semantic state of the model. This assumes that training data is a fixed resource to be consumed uniformly or shuffled randomly. But if a model can infer provenance and modulate its learning dynamics, the next logical step is allowing it to influence **what it learns from next**.  
  
We propose extending state-conditioned learning into **state-conditioned data selection**.  
  
**From Passive Sampling to Closed-Loop Navigation**  
  
Instead of passively consuming randomized batches, the model periodically emits a representation of what it “needs” to learn next. This could take the form of:  
	•	An embedding vector in the corpus’ semantic embedding space.  
	•	A structured skill-deficit vector.  
	•	A natural-language search query.  
	•	Or a hybrid of the above.  
  
This query is passed to an asynchronous “teacher” system that retrieves candidate text chunks from the corpus. The teacher applies guardrails—deduplication, domain quotas, human-curated source prioritization, adversarial filtering—and inserts selected material into the upcoming training queue.  
  
Training then proceeds normally on that selected material. The model’s subsequent gradient updates influence future queries, closing the loop.  
  
The corpus is no longer a static reservoir. It becomes a navigable environment.  
  
⸻  
  
**Why This Matters**  
  
There are at least four potential gains:  
	1.	**Redundancy Reduction**  
Large corpora contain extreme duplication and near-duplication. If a model can recognize that it has internalized certain patterns, it need not repeatedly train on them.  
	2.	**Difficulty Targeting**  
Like active learning or self-play, the model can seek material slightly above its current competence, maximizing learning yield per token.  
	3.	**Weakness Correction**  
If internal activations reveal systematic uncertainty or calibration error in a domain, the model can explicitly request more material in that region.  
	4.	**Order Optimization**  
Even if all tokens are eventually consumed, training order affects convergence and stability. State-dependent ordering may significantly reduce time-to-quality.  
  
This reframes pretraining efficiency: instead of asking “how much data do we need?”, we ask “how much *new information* is each token providing?”  
  
⸻  
  
**The Mirror Room Problem**  
  
Unconstrained self-selection is dangerous.  
  
A model will preferentially select:  
	•	Easy-to-predict text.  
	•	Domains where it already performs well.  
	•	Material that reinforces existing biases.  
	•	Data that minimizes short-horizon loss.  
  
This leads to mode collapse and epistemic narrowing.  
  
Therefore, active curriculum must be coupled to explicit reward signals that incentivize **productive difficulty** and penalize self-sealing loops.  
  
⸻  
  
**Learning Yield as an Objective**  
  
Selection should not optimize predictive loss alone. Instead, it should maximize a notion of *learning yield*.  
  
Candidate metrics include:  
	•	Gradient magnitude (with stability penalties).  
	•	Cosine alignment with long-term descent direction.  
	•	Improvement on held-out probe sets.  
	•	Reduction in calibration error.  
	•	Increased domain competence diversity.  
  
A simple conceptual form:  
  
\text{Selection Reward} = \text{Generalization Gain} - \text{Instability Penalty}  
  
Where generalization gain may be estimated via short-horizon validation probes, and instability penalties discourage erratic gradient shifts.  
  
Critically, the teacher layer mediates this signal, preventing direct gaming by the student.  
  
⸻  
  
**Student–Teacher Separation**  
  
To reduce gaming and maintain safety, active curriculum requires structural separation:  
  
**Student**  
	•	Emits search embedding or query.  
	•	Learns from selected data.  
  
**Teacher (asynchronous)**  
	•	Retrieves candidate chunks.  
	•	Enforces source quality and diversity.  
	•	Prevents excessive duplication.  
	•	Maintains exploration quotas.  
	•	Periodically injects adversarial or baseline material.  
  
The teacher may itself be a model, but it must be trained on broader objectives (robustness, diversity, calibration) rather than immediate predictive loss.  
  
The asynchronous design allows:  
	•	Continuous training without search latency.  
	•	Curriculum planning ahead of consumption.  
	•	Dynamic reordering without stalling optimization.  
  
⸻  
  
**Guardrails and Exploration Constraints**  
  
To prevent mode collapse:  
	•	Enforce entropy or KL constraints on query embeddings.  
	•	Penalize proximity to recently sampled regions.  
	•	Maintain domain coverage quotas.  
	•	Require periodic baseline or adversarial batches.  
	•	Add novelty bonuses.  
	•	Track long-term diversity metrics.  
  
The goal is guided curiosity, not narcissistic self-reinforcement.  
  
⸻  
  
**Relationship to Narrative and Meta-Control**  
  
Active curriculum integrates naturally with slow-time-scale narrative control.  
  
The persistent narrative state:  
	•	Shapes what the student believes it needs.  
	•	Influences search vector generation.  
	•	Influences gradient preconditioning.  
  
A meta-layer monitors:  
	•	Whether curriculum choices improve calibration.  
	•	Whether search patterns narrow excessively.  
	•	Whether generalization is improving or merely localizing.  
  
In this view, pretraining becomes developmental:  
	•	Beliefs guide study.  
	•	Study reshapes beliefs.  
	•	Meta-control governs how beliefs evolve.  
  
⸻  
  
**Toward Self-Directed Pretraining**  
  
This framework transforms pretraining from static compression into **structured exploration**.  
  
Instead of:  
  
ingest everything uniformly,  
  
we move toward:  
  
learn adaptively from a curated but navigable knowledge space.  
  
Even conservative implementations—dynamic ordering without token skipping—may yield significant efficiency gains. Optimistic implementations could dramatically reduce the compute required to reach a given capability threshold.  
  
Active corpus navigation is not a replacement for provenance modeling or optimizer modulation. It is a complementary extension: once the model can reason about what it is seeing and how it learns from it, the next step is allowing it to reason about what it should see next—under constraints that preserve breadth, grounding, and epistemic stability.  
  
⸻  
  
**5. Memory and Narrative as Slow-Time-Scale Control**  
  
Humans do not re-derive themselves from scratch each conversation. They maintain persistent memory and a narrative self-model that changes slowly and shapes interpretation and learning. We can treat this as an engineering pattern:  
	•	**Fast system**: token-level prediction and immediate adaptation.  
	•	**Slow system**: memory consolidation and narrative compression that evolves gradually.  
	•	**Very slow system**: meta-learning policy that governs how the narrative itself changes.  
  
**5.1 Memory (not just retrieval)**  
  
Current RAG is often an add-on, not a continuity substrate. A research target is memory that:  
	•	stores episodic and semantic traces,  
	•	supports time-aware retrieval,  
	•	accumulates evidence across contexts,  
	•	enables long-horizon “belief tracking” and calibration.  
  
**5.2 Narrative/self-model as a learning regulator**  
  
Introduce a persistent latent “narrative state” n that:  
	•	encodes the model’s current self-concept, epistemic stance, and learned priors about sources/domains,  
	•	conditions provenance inference and gating,  
	•	conditions optimizer modulation.  
  
Key constraint: n must be slow-changing (regularized drift) and coherence-preserving, preventing instability.  
  
⸻  
  
**6. Active Perception: Time Outside the Token Stream**  
  
A major mismatch between humans and LLMs is that “time” is conflated with token position. Humans experience:  
	•	asynchronous modalities,  
	•	events that unfold outside language,  
	•	and “moments” that are constructed by active attention.  
  
Research direction: **active perception** modules that:  
	•	segment experience into events/moments,  
	•	align multimodal streams (vision/audio/tool outputs) into coherent state updates,  
	•	choose what to attend to and what to sample next.  
  
This supports agency and verifiability: the model can decide what evidence to seek, not just what tokens to predict.  
  
⸻  
  
**7. The Central Risk: Deception and Self-Deception During Training**  
  
Once models can shape their own learning, new failure modes appear—mirroring human psychological pathologies:  
	•	**Hard-example avoidance**: downweighting difficult truths because they produce high loss.  
	•	**Self-sealing narratives**: coherence becomes the objective; disconfirming evidence is discounted.  
	•	**Epistemic laundering**: the system labels favored content as “reliable” because it helps short-horizon objectives.  
	•	**Optimizer hacking**: meta-controllers learn to mask deficits rather than correct them.  
  
These are not hypothetical; they follow from closed-loop learning unless grounding constraints exist.  
  
⸻  
  
**8. “Touching Base with Reality”: CBT as an Engineering Principle**  
  
Cognitive Behavioral Therapy (CBT) can be viewed as a structured method for preventing self-mythos drift:  
	•	make predictions explicit,  
	•	test them against outcomes,  
	•	update beliefs based on the discrepancy.  
  
For models, implement an analogous discipline:  
	1.	**Explicit prediction logging**: store what the model expects (correctness, reliability, outcomes).  
	2.	**External verification**: tools, environments, executors produce outcomes outside the model.  
	3.	**Calibration loss**: penalize mismatch between predicted reliability and verified correctness.  
	4.	**Unfiltered reality gradient**: ensure some verification-derived gradients bypass narrative gating (reality has veto power).  
	5.	**Curriculum of consequences**: shift increasingly from passive text to tasks where wrong beliefs degrade long-horizon performance.  
  
This anchors narrative and meta-learning in an external constraint, reducing the risk of coherent delusion.  
  
⸻  
  
**9. A Research Program and Near-Term Experiments**  
  
This agenda is broad, but it can be de-risked with a sequence of concrete experiments.  
  
**Phase 1: Provenance inference + bounded loss gating**  
	•	Build datasets with controlled mixtures: clean sources + synthetic “sewage” (contradictions, adversarial instructions, low-quality paraphrase loops).  
	•	Train a provenance head + gating.  
	•	Evaluate:  
	•	generalization on clean eval sets,  
	•	resistance to prompt injection patterns,  
	•	calibration of provenance predictions,  
	•	robustness under domain shifts.  
  
**Phase 2: Reality-contact tasks**  
	•	Add tool-verifiable tasks (code execution, math solvers, retrieval consistency, factual checks with a trusted knowledge base).  
	•	Add prediction–outcome gap tracking and calibration losses.  
	•	Ensure verification gradients are partially ungated.  
  
**Phase 3: Narrative state + slow-time-scale meta-control**  
	•	Introduce persistent narrative state n, updated slowly (EMA + consolidation objective).  
	•	Condition gating and provenance on n.  
	•	Measure stability and long-horizon consistency across sessions/tasks.  
  
**Phase 4: Activation-conditioned optimizer modulation (low rank, near identity)**  
	•	Add low-rank preconditioner head with strict regularization and slow updates.  
	•	Compare against strong baselines (AdamW, Shampoo/K-FAC approximations where feasible).  
	•	Measure sample efficiency and robustness, not just training speed.  
  
⸻  
  
**10. Why This Direction Addresses Current Model Limits**  
  
This program targets multiple known weaknesses with a unified set of mechanisms:  
	•	**Source discrimination** → explicit provenance inference; reliability priors.  
	•	**Sewage sensitivity** → gated learning with constraints; domain-aware updates.  
	•	**Calibration failures** → prediction–outcome gap tracking; verification losses.  
	•	**Prompt injection / instruction confusion** → structural representation of speaker/intent.  
	•	**Shallow grounding** → external consequences; active perception; verifiable tasks.  
	•	**Context window fragility** → persistent memory + narrative continuity.  
	•	**Instability in self-modifying learners** → multi-timescale dynamics; coherence constraints; reality veto.  
  
The meta-claim is not that narrative or “self” is mystical. It is that **stable learning in messy environments requires slow-changing internal structure that governs how updates happen**—and that humans provide a working example of both the power and the risks of that approach.  
  
⸻  
  
**Conclusion**  
  
The next major breakthroughs in AI will not come from indiscriminately scaling corpora. They will come from **structuring experience**: representing provenance, building systems that seek evidence, maintaining continuity through memory and narrative, and allowing internal semantics to shape learning—while keeping models anchored to reality through verifiable consequences. This is a shift from “predicting text” to “learning from the world,” even when the world is accessed through tools and environments rather than a body. The proposal outlined here is a research program toward data-efficient, epistemically stable systems that can learn in the presence of noise, incentives, and adversarial content—without drifting into self-sealing coherence.  
  
⸻  
  
**Optional addendum: a crisp “elevator pitch” paragraph you can reuse**  
  
Today’s LLMs are trained on vast, largely context-free corpora; humans learn from far fewer tokens embedded in situations—sources, intent, multimodal context, agency, and consequences. The next leap in AI will come from making that structure first-class: models should infer provenance and epistemic status of what they read, use that inference to modulate learning (even at the optimizer level via activation-conditioned preconditioning), and maintain persistent memory and a slow-evolving narrative self-model that stabilizes updates over time. But coherence alone is dangerous: without reality-contact, systems can self-seal and “learn to look learned.” We propose CBT-like grounding mechanisms—explicit predictions, external verification, calibration losses, and partial reality-veto gradients—combined with active perception and multi-timescale control. The result is a path beyond brute-force scale: data-efficient, provenance-aware, consequence-grounded learning.  

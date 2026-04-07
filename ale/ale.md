# ALE: Adaptive Local Experimentation

## A framework note

This is not a formal paper draft. It is a structured write-up of the core idea behind **ALE** — *Adaptive Local Experimentation* — and why it may be a useful broader frame for thinking about learning and optimization. It is written to preserve the conceptual center of the idea and the main reasons for excitement around it, while leaving room for a later paper grounded in actual experimental results, likely beginning with PBDCA.

---

## 1. The motivating shift

Most mainstream optimization in machine learning is framed around a fairly fixed story:

1. compute a gradient or gradient-like signal,
2. smooth, scale, or precondition it,
3. take a step.

Even when the optimizer is sophisticated — momentum, Adam, Shampoo, K-FAC, whitening, clipping, trust regions, and so on — the basic epistemic posture is usually the same. The system does not explicitly represent what it *knows* and *does not know* about the local loss landscape. It transforms an update signal, but it does not ordinarily maintain an explicit local model of the terrain or allocate extra compute to reduce uncertainty when the signal is ambiguous.

ALE starts from a different premise:

> **Optimization can be viewed as adaptive local experimentation in an intervention space under a compute budget.**

Under this view, the learner is not merely handed an update signal. It chooses interventions, observes outcomes, fits a local model, tracks uncertainty, decides whether more evidence is worth its cost, and then acts.

That is a genuine change in perspective. It makes the optimizer's **epistemic state** a first-class object in the system.

---

## 2. Core idea

ALE treats learning as a repeated cycle of four things:

1. **Select an intervention space**
   - a local set of actions the learner is allowed to try,
   - often low-dimensional or otherwise constrained,
   - e.g. probe directions in weight space, module-level perturbations, routing changes, memory-write policies.

2. **Run local experiments**
   - evaluate selected interventions,
   - typically in parallel if possible,
   - subject to a fixed or adaptive budget.

3. **Maintain an epistemic state**
   - a local surrogate or metamodel of how interventions affect downstream outcomes,
   - plus uncertainty estimates and any relevant priors or structural assumptions.

4. **Use that epistemic state to control both motion and measurement**
   - take a step when the evidence is clear enough,
   - or allocate more experiments when the evidence is weak, noisy, or structurally ambiguous.

This is not merely "optimization with a better estimator." It is a broader view in which the learner may spend compute not just on acting, but on **learning about the current update problem itself**.

---

## 3. Why this is different from standard optimizer state

Standard optimizers do have state:

- momentum buffers,
- running second moments,
- preconditioners,
- trust-region radii,
- exponential moving averages,
- sometimes curvature approximations.

But these are usually summaries of past gradients or update statistics. They are not typically explicit models of:

- what local structures are plausible,
- what remains uncertain,
- which measurements would be most informative,
- whether the current evidence justifies a step,
- or whether the intervention basis itself is poorly chosen.

ALE, by contrast, treats the optimizer's epistemic state as a real object:

- a metamodel of local response,
- uncertainty around that metamodel,
- and a policy for selecting new measurements in light of that uncertainty.

In that sense, ALE is closer to **Bayesian experimental design applied to optimization** than to ordinary gradient transformation.

This need not mean that every implementation is fully Bayesian in a strict sense. The important point is structural:

1. maintain an uncertainty-bearing local model,
2. choose experiments based on expected informational value or decision relevance,
3. update the model from outcomes,
4. act using the updated model.

That is the key shift.

---

## 4. ALE as meta-learning framework, not optimizer replacement

An earlier version of this thinking positioned ALE primarily as an alternative to SGD — a different way to compute updates that might outperform gradient-based methods. That framing was both too ambitious and too narrow.

Too ambitious because SGD with modern preconditioners (Adam, Shampoo) is an extraordinarily effective optimizer when its assumptions hold: smooth loss landscape, informative gradients, well-conditioned curvature. Trying to beat it on its home turf is fighting uphill, and early experimental results bear this out. In one set of tests, an ALE-style method that temporarily decoupled the LM head from the backbone showed promising improvement in the first several hundred steps, then consistently degraded and finished worse than standard joint training. The likely explanation is instructive: decoupling temporarily disrupts the co-adaptation between components that SGD handles naturally. The early gains came from a regime where active experimentation had an opening; the later degradation came from forcing experimentation in a regime where coupled gradient flow was already doing the right thing.

Too narrow because the deeper insight is not that ALE replaces any particular optimizer, but that it provides a **regime-aware meta-framework** that governs *when and where* to deploy different training strategies — including standard SGD as the default.

Under this reframing:

- **SGD (with preconditioners) is the default experimenter.** It runs when conditions are right, and its conditions are right most of the time. When the loss landscape is smooth, gradients are informative, and curvature is well-conditioned, the optimal compute allocation is: use the analytical gradient, take the step, move on.

- **ALE activates selectively** when there is evidence that the default strategy is underperforming — that gradients are noisy, that steps are not producing expected improvement, that the optimization is in a regime where first-order methods are wasting compute.

- **The meta-framework maintains a lightweight epistemic state** about the training process itself — not a full surrogate model of the loss landscape, but a set of diagnostics and priors that govern transitions between training strategies.

This resolves the core tension in the original framing. ALE does not need to be globally better than SGD. It needs to be *locally better in specific regimes*, and it needs to know when those regimes are active. That is a much easier bar to clear, and it is falsifiable at each transition point.

---

## 5. Backprop as one point in a larger design space

ALE is useful in part because it reframes backpropagation.

Backprop is often treated as *the* canonical learning rule, with alternatives judged by how closely they approximate its signal. ALE suggests a different picture.

Backprop can be understood as a special case in which:

- the local model is effectively supplied analytically by the computation graph,
- the experiment is fixed in form,
- the inference rule is exact reverse-mode accumulation,
- and the budget is largely non-negotiable: one backward pass.

Under this framing, backprop is not invalidated. It is simply one corner of a broader design space:

- **Analytic local inference**: backpropagation.
- **Fixed experimental local inference**: finite differences, SPSA, some evolutionary/perturbation-style methods.
- **Adaptive experimental local inference**: ALE-style methods.

This is important conceptually. It stops treating alternatives as ersatz gradients and instead asks:

> What intervention space is available, what local model is maintained, and how is compute budget allocated between learning about the next step and taking it?

That is a healthier frame for comparing methods, especially in regimes where clean analytic gradients lose their privileged status.

And under the meta-framework view, the question becomes even more pointed: when does the analytic corner of this design space stop being the best use of compute, and how do you detect that?

---

## 6. The three-tier diagnostic hierarchy

If ALE is a meta-framework that governs regime transitions, it needs a principled way to detect when the current training strategy is inadequate. This leads naturally to a three-tier diagnostic hierarchy, where each tier has a distinct epistemic role.

### Tier 1: Cheap continuous indicators

These run always (or on cheap schedules) and watch for instability or diminishing usefulness of the current optimizer's signal. They are the perception layer.

Examples:

- **Gradient variance and directional consistency.** Cosine similarity between successive gradient vectors (or per-parameter-group versions) is essentially free to track. A sustained drop means the optimizer is receiving contradictory instructions step to step. Adam's second moment already provides a per-parameter noise estimate.

- **Sign-flip frequency.** Track the sign of parameter updates over a sliding window. High oscillation indicates the effective learning rate is too high for the local geometry — a curvature mismatch the optimizer isn't correcting for.

- **Progress per unit compute.** Loss delta per step, smoothed. The interesting signal is when this ratio drops faster than expected from smooth convergence.

- **Microbatch or accumulation-window disagreement.** If gradients across microbatches within a single step point in substantially different directions, the step is being dominated by noise rather than signal.

These are raw local symptoms. Individually, none is decisive. Their value comes from persistent patterns and from interaction with the other tiers.

### Tier 2: Moderate-cost validation probes

These activate when Tier 1 raises flags. They test whether the optimizer's implicit local model — the assumption that the gradient direction leads to proportional improvement — still holds. They are the investigation layer.

Examples:

- **Predicted vs. actual loss drop.** Rerun a forward pass on a batch after an update to compare the realized loss change with the linearized prediction. If the prediction says loss should drop by X and it actually drops by 0.3X, the optimizer is in a highly nonlinear region where first-order methods are wasting compute. Cost: one extra forward pass, affordable on a schedule rather than every step.

- **Short-horizon local linearity checks.** Take a half-step, measure loss, compare with the full step. Persistent sublinearity means curvature is significant at the current step size.

- **Symmetric perturbation tests.** Tiny perturbations around the proposed step to estimate local curvature or asymmetry in the loss surface.

- **Blockwise scaling tests.** Would a scaled-down step in certain parameter groups have produced better total improvement? This probes whether the global learning rate is masking per-component pathology.

These are more expensive than Tier 1 but still far cheaper than full active experimentation. They convert vague symptoms into actionable diagnoses.

### Tier 3: Structural priors

These are not measured per-step. They are known in advance from the architecture and task structure. They represent prior knowledge about *where* standard gradient-based optimization is likely to be inadequate — not because anything has gone wrong empirically, but because the computational structure involves decisions whose consequences are temporally or causally distant from the point where the gradient is computed.

Examples:

- **Discrete routing or gating.** The gradient tells you how to adjust the router's soft scores, but it cannot tell you whether a completely different routing decision would have been better — that counterfactual was never evaluated.

- **Memory writes with delayed consequences.** The gradient through a write operation tells you how to adjust *what* was written, but not *whether* writing was the right thing to do.

- **Retrieval decisions.** What to retrieve, when, and from where — decisions with downstream consequences the gradient may not capture well.

- **Tool use and stateful environment interaction.** Consequences that are delayed, discrete, or dependent on external state.

- **Any setting where even a stable gradient may be the wrong object to trust fully** — not because the gradient is noisy, but because the structure of the credit-assignment problem is mismatched to what gradients can express.

### How the tiers interact

The crucial refinement is that Tier 3 is not a separate trigger class running in parallel with Tiers 1 and 2. Instead, it functions as a **prior over how to interpret Tier 1 and Tier 2 evidence**.

High gradient variance globally might be normal during early training. But high gradient variance *specifically in the router parameters* of a mixture-of-experts model is a much stronger signal that something structurally problematic is happening. The meta-learner doesn't just check "is variance high?" — it checks "is variance high in a component where I already have structural reasons to distrust the gradient?"

This gives the hierarchy a properly Bayesian flavor:

- **Tier 3** provides a prior over where problems are likely to arise and how they will manifest.
- **Tier 1** provides a continuous likelihood stream — cheap observations about the current state of training.
- **Tier 2** is an active inference step taken to reduce uncertainty when the posterior (prior × likelihood) crosses some threshold.

The meta-learner spends epistemic compute in proportion to its uncertainty about whether the current optimizer is adequate, weighted by its prior belief about where inadequacy is likely.

---

## 7. Triple locality as design principle

A natural concern with any meta-learning framework is overhead. If monitoring and regime detection cost more than the improvement they produce, the system is worse than doing nothing.

ALE addresses this through what might be called **triple locality** — the principle that meta-reasoning is constrained to be local along three dimensions simultaneously:

1. **Local in parameter/component space.** ALE does not monitor everything everywhere. Tier 3 priors focus Tier 1 diagnostics on the components where trouble is structurally expected — router parameters, memory-write heads, gating mechanisms. Components where SGD's assumptions are architecturally sound get minimal monitoring.

2. **Local in time.** ALE does not run expensive diagnostics every step. Tier 1 indicators run cheaply and continuously (possibly piggybacking on optimizer state that already exists, like Adam's moment estimates). Tier 2 probes fire only when Tier 1 flags persist. Active experimentation activates only when both tiers indicate the current strategy is failing in a flagged component.

3. **Local in epistemic sophistication.** ALE does not deploy full surrogate models when a running average would suffice. The level of meta-reasoning scales with the evidence that more reasoning is warranted — from a few running statistics, to targeted probes, to structured local experimentation, and only then to full ALE-style adaptive intervention.

The total overhead is the product of three small fractions rather than the sum of three large ones. This is what keeps the framework computationally viable.

Triple locality also mirrors good engineering practice: you don't instrument every variable in a production system. You put monitors on the components you have reason to worry about, set alerts that trigger deeper diagnostics, and intervene locally when you find a problem. ALE applies that pattern to the optimization process itself.

---

## 8. The intervention space

The intervention space is central to ALE.

An intervention space is the set of local controllable actions the learner can explore. It need not be the full parameter space. In fact, one of the strengths of ALE is that it naturally encourages constrained or structured local action spaces.

Examples include:

- low-dimensional subspaces of weight updates,
- learned or hand-chosen probe directions,
- module-specific perturbations,
- routing or gating changes,
- memory-write perturbations,
- curriculum or data-selection actions,
- architectural or control actions in systems with discrete components.

This matters because the choice of intervention space determines what local geometry can even be represented. A poor intervention basis does not just make learning inefficient; it can distort the learner's inferred model of the landscape.

That means ALE naturally raises questions about:

- how intervention spaces are chosen,
- how they evolve,
- when the current basis is failing,
- and when new directions should be created, combined, rotated, or discarded.

---

## 9. Metamodels and local epistemic state

A natural way to instantiate ALE is through a local surrogate or metamodel fit to intervention-outcome pairs.

For example, the learner might collect tuples like:

- intervention configuration,
- resulting loss or return,
- perhaps delayed consequences,
- perhaps additional diagnostics.

Then it fits a local model to these observations. That local model might be:

- linear,
- quadratic,
- low-rank quadratic,
- kernel-based,
- Bayesian linear regression,
- recursive least squares,
- or some other structured surrogate.

A quadratic model is especially appealing because it provides, in one object:

- a slope estimate,
- interaction terms between intervention directions,
- a notion of curvature,
- and therefore a natural approximate preconditioner.

This is powerful because it pools information across all local samples rather than relying only on direct finite-difference style comparisons. It allows the learner to infer a local landscape rather than merely ask whether one perturbation happened to help.

The danger, of course, is sample complexity. Full quadratics scale poorly with dimension. That pressure naturally pushes ALE toward:

- very low-dimensional local subspaces,
- structured metamodels such as diagonal or low-rank-plus-diagonal curvature,
- online/recursive fitting rather than repeated from-scratch regression,
- or priors that impose strong inductive bias on the local geometry.

This is not a flaw in the framework. It is one of the real design constraints it forces into the open.

---

## 10. Uncertainty is not a nuisance — it is part of the control signal

In ordinary training, ambiguity often collapses into:

- small gradients,
- noisy gradients,
- or plateau behavior.

ALE allows ambiguity to be interpreted more richly.

The system can distinguish among different kinds of uncertainty, for example:

- uncertainty about the best direction within the current basis,
- uncertainty about whether the basis itself is missing important directions,
- uncertainty caused by high curvature or local nonlinearity,
- uncertainty caused primarily by stochastic noise,
- uncertainty due to unresolved interactions among intervention dimensions,
- disagreement among plausible local models.

These different epistemic states need not trigger the same response.

Possible responses include:

- take the step,
- shrink step size and probe more densely nearby,
- expand the probe radius,
- replicate measurements to reduce noise,
- introduce or rotate basis directions,
- test combinations of interventions,
- allocate more branches around the current point,
- or refuse to step until the local picture is better identified.

This is one of the most important consequences of ALE:

> unclear signal is not merely something to average away; it can be turned into a reason to adapt the experiment itself.

That gives the learner knobs to turn when ordinary optimization would mostly just grind forward.

---

## 11. Active local experimentation under a budget

Once a metamodel exists, the learner can use it not just to choose a step, but to choose the **next measurements**.

This turns local probing into an active design problem.

Given a branch budget or evaluation budget, the learner can ask:

- where is uncertainty highest *and relevant* to the decision,
- where do plausible local models disagree most,
- which measurement would most change the recommended step,
- whether more evidence is worth the compute,
- whether to exploit the current best direction or explore to better identify the local structure.

This is why ALE is better described as **adaptive local experimentation** rather than merely "perturbation-based learning." The learner is not just moving through the space; it is deciding how to spend limited experimental effort before committing to motion.

In that sense, ALE is closer to **sequential decision-making under uncertainty** than to ordinary update rules.

---

## 12. Why this may matter in practice

The conceptual case for ALE is strong on its own, but the practical case is strongest in regimes where standard backpropagation is no longer obviously the right local object or where gradients are blocked, misleading, too local, or too expensive.

Examples include:

- delayed credit assignment,
- memory writes with long downstream effects,
- discrete routing or gating decisions,
- non-differentiable environment transitions,
- stateful tool use,
- recurrent systems with sparse or delayed consequence structure,
- settings where local derivatives exist but do not line up well with useful intervention choices.

These are the settings where "what happens if I intervene here?" may be a more natural learning question than "what is the instantaneous derivative of the scalar objective with respect to this parameter?"

Memory is a particularly strong example. Small changes at write time can have delayed, discontinuous, or highly state-dependent effects later. Even where gradients are technically available, they may not be the most useful object for deciding what kinds of write-side interventions are worthwhile. ALE provides a language for explicitly modeling and experimenting with such interventions.

This does not imply that ALE will outperform backprop on ordinary dense supervised training. That would be a much stronger claim and is unnecessary. The point is narrower:

> there are important regimes where explicit local experimentation over a constrained intervention space may be better matched to the structure of the credit-assignment problem than a fixed backward pass.

And the meta-framework view adds a crucial corollary:

> the same training run may pass through regimes where SGD is optimal and regimes where it is not, and a well-designed system should be able to detect the transition and respond.

---

## 13. Why PBDCA is a good first instantiation

PBDCA — Probe-Based Directional Credit Assignment — remains a strong starting point even if ALE becomes the broader umbrella.

PBDCA provides:

- an explicit intervention space,
- structured local perturbations,
- parallel branch evaluation,
- and a natural connection to delayed or nonlocal consequences.

Originally, one can view PBDCA as a way to project updates into a low-dimensional space defined by probe directions and evaluate outcomes at multiple points in that space. In its simplest form, it can produce pseudo-gradient-like updates from those measurements.

The ALE perspective opens a broader extension:

- fit a metamodel over the probe-configuration / loss pairs,
- use the metamodel rather than direct finite differences to infer a step,
- extract curvature information from the local fit,
- and use uncertainty in the fit to actively choose new probe configurations.

Under that interpretation, PBDCA becomes less "approximate backprop via probes" and more "active navigation of a projected local loss landscape."

That is a much more compelling identity.

Under the meta-framework view, PBDCA is one experimenter in a portfolio. It is the experimenter you activate when Tier 3 structural priors are present (delayed credit, discrete routing, memory writes) and Tier 1 indicators suggest SGD is struggling *specifically in those components*. PBDCA does not need to compete with SGD on well-behaved parameters — it slots into the regions of the architecture and the phases of training where active experimentation has a natural opening.

It also makes PBDCA a practical and conceptual bridge:

- practical, because it provides a concrete testbed,
- conceptual, because it demonstrates ALE in a regime where the broader idea is most likely to have teeth.

If ALE is the general paradigm, then PBDCA can be presented as one instantiation specialized to projected parameter-space interventions and directional credit assignment.

---

## 14. The optimizer's epistemic state as a first-class object

Perhaps the deepest version of the idea is this:

> The primitive object in learning need not be the gradient. It can instead be the learner's evolving epistemic state about how local interventions affect downstream outcomes.

This is what makes ALE feel larger than a specific optimizer trick.

In ALE, the optimizer has something like beliefs:

- a representation of plausible local response structure,
- uncertainty over that structure,
- a notion of which evidence would be most useful next,
- and a decision rule that maps its current epistemic state into both motion and additional measurement.

This can be implemented in many ways, from relatively crude structured surrogates to explicitly Bayesian versions. But the central idea survives across implementations.

This also makes priors more explicit. One can imagine priors over:

- smoothness,
- curvature structure,
- sparsity of interactions,
- temporal continuity of local geometry,
- or usefulness of particular intervention directions.

Unlike the hidden assumptions built into standard gradient methods, these assumptions become inspectable, revisable, and potentially learnable.

That is one of the most intellectually satisfying aspects of the framework.

---

## 15. Reasons for enthusiasm

There are several reasons this feels like a substantive idea rather than a clever tweak.

### 15.1 It changes the ontology of optimization

The central object is no longer just an update signal but a belief state over local intervention consequences.

### 15.2 It compresses many moving parts into one principle

Metamodels, uncertainty, active branch allocation, adaptive compute, basis adaptation, and decision-aware probing all become natural pieces of the same framework rather than disconnected additions.

### 15.3 It is generative

Once the framework is stated, a structured research program appears naturally:

- what makes a good intervention space,
- what uncertainty matters,
- what metamodel classes are sufficient,
- how to allocate branch budget,
- when to seek a better basis rather than a better estimate,
- what priors are useful,
- how to learn the controller itself.

### 15.4 It gives a principled middle ground

It avoids the false choice between rigid analytic differentiation and vague black-box search. ALE says the learner can actively investigate local intervention consequences under budget.

### 15.5 It seems especially promising where credit is delayed, discrete, or stateful

This may be exactly the class of problems where explicit local experimentation earns its compute.

### 15.6 It subsumes rather than competes

The meta-framework view means ALE does not claim to replace SGD. It recognizes SGD as an excellent default experimenter and provides a principled account of when to augment or override it. This is both a weaker claim (no need to beat SGD globally) and a stronger framework (it governs the entire training process, not just one optimizer).

---

## 16. Practical cautions

The framework is conceptually strong, but there are real risks.

### 16.1 The epistemic machinery must earn its cost

It is easy to build something richer than SGD that is also slower, noisier, harder to tune, and less effective in practice. The key question is whether explicit local experimentation pays for itself in the target regime. Triple locality (§7) is the primary safeguard here — overhead must stay small across space, time, and sophistication simultaneously.

### 16.2 Surrogates can become too expressive

If the metamodel layer grows too complex, it may quietly recreate the complexity the framework was meant to avoid. Starting with constrained local models is likely wiser.

### 16.3 Nonstationarity is real

The landscape is moving because the system itself is changing. Any local model must handle forgetting, trust regions, or temporal continuity assumptions carefully.

### 16.4 Basis quality matters even more than usual

A bad basis does not just slow progress; it can corrupt what the learner thinks the local landscape looks like.

### 16.5 The meta-learner must not become the bottleneck

The regime-detection system must be substantially cheaper than the experiments it governs. If tier 1 diagnostics alone cost more than a few percent of a training step, the framework is poorly instantiated. Piggybacking on existing optimizer state (Adam's moment estimates, gradient norms already being logged) is likely necessary.

### 16.6 The framework must stay sharp

If ALE expands so broadly that it covers everything, it loses bite. Its core commitments must remain visible:

1. explicit intervention space,
2. explicit uncertainty-bearing local model,
3. adaptive compute allocation for local evidence gathering,
4. update behavior conditioned on epistemic state,
5. triple locality constraining where, when, and how meta-reasoning occurs.

---

## 17. A possible hierarchy

A useful way to preserve both breadth and specificity is:

- **ALE**: the umbrella meta-framework — Adaptive Local Experimentation — governing regime detection, strategy selection, and the epistemic state that connects them.
- **SGD/Adam/Shampoo**: the default experimenters. They run when their assumptions hold. ALE's diagnostic hierarchy monitors whether those assumptions continue to hold.
- **PBDCA**: one concrete ALE experimenter focused on probe-defined intervention subspaces and directional credit assignment. Activated when structural priors and empirical diagnostics indicate that gradient-based methods are inadequate in specific components.
- **Future experimenters**: ALE methods with different intervention spaces, metamodels, acquisition rules, and decision rules — each suited to different regimes and failure modes.

This keeps continuity with existing work while allowing the larger idea to breathe. The hierarchy also provides a natural narrative for early experimental results: methods that show promising early gains but degrade later may be activating in the right regime but failing to yield back to SGD when the regime shifts. The meta-framework would know when to stop.

---

## 18. A simple working definition

A concise working definition might be:

> **ALE treats learning as adaptive, uncertainty-aware local experimentation over a constrained intervention space under a compute budget.**

Or, in a slightly more operational form:

> **In ALE, the learner maintains an uncertainty-bearing local model of how chosen interventions affect downstream outcomes, allocates experimental budget to reduce decision-relevant uncertainty, and uses the resulting epistemic state to decide both what to measure next and how to move.**

Or, in the meta-framework form:

> **ALE is a regime-aware meta-learning framework that maintains epistemic state about the adequacy of the current training strategy, uses a three-tier diagnostic hierarchy — cheap continuous indicators, moderate-cost validation probes, and structural priors — to detect when to transition between strategies, and deploys active local experimentation selectively in the components and phases of training where standard gradient-based methods are inadequate.**

The first two definitions capture the core mechanism. The third captures the operational architecture.

---

## 19. Near-term path

A sensible near-term path is not to write the grand unified ALE paper immediately, but to proceed on two fronts.

**Front 1: PBDCA as proof of life.** PBDCA remains the first serious instantiation. The path here is:

1. show PBDCA working in a regime with delayed credit, memory, or non-differentiable transitions,
2. add a simple metamodel layer,
3. add uncertainty-aware branch allocation,
4. demonstrate that the added epistemic machinery changes behavior in a meaningful and useful way.

**Front 2: Diagnostic hierarchy as standalone contribution.** The three-tier regime-detection framework is independently valuable and testable:

1. implement Tier 1 diagnostics piggybacking on Adam's existing state,
2. show that Tier 1 indicators correlate with known training pathologies (learning rate too high, gradient noise domination, plateau entry),
3. implement Tier 2 probes (predicted vs. actual loss drop) and show they add diagnostic power beyond Tier 1,
4. demonstrate that Tier 3 structural priors (e.g., flagging router parameters in MoE) make Tier 1 signals more interpretable — that the same raw symptom means different things in different components.

The diagnostic hierarchy could be published as its own contribution — a monitoring framework for training that makes the implicit regime-awareness of experienced practitioners explicit and systematic — even before PBDCA or any other ALE experimenter is fully validated.

Then the broader ALE framing can be written with empirical grounding from both fronts.

---

## 20. Closing thought

What makes ALE exciting is not merely that it offers another way to produce updates. It offers a different picture of what a learner can be.

Instead of a system that reflexively transforms whatever descent signal it is given, ALE points toward a system that:

- recognizes when its current strategy is working and lets it run,
- detects when that strategy is struggling, with diagnostic specificity about where and why,
- activates targeted experimentation in the right components at the right time with the right level of sophistication,
- intervenes, infers, tracks uncertainty, allocates attention and compute strategically,
- and acts only when it has learned enough locally to justify motion.

The triple locality principle — local in parameter space, local in time, local in epistemic sophistication — is what keeps this from blowing up. The three-tier diagnostic hierarchy — perception, investigation, prior knowledge — is what gives it teeth.

That may turn out to be expensive, temperamental, or useful only in certain regimes. But as a framework, it is substantive. It makes the optimizer's epistemic state explicit, and it suggests that learning may often be better understood not as reflexive descent, but as **budgeted local inquiry in a space of controllable interventions, governed by a regime-aware meta-learner that knows when to trust the gradient and when to look more carefully**.

That is the larger idea.

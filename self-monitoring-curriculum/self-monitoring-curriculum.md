# From Confession to Inhibition: A Temporal Curriculum for Self-Monitoring in Language Models

**Richard Vermillion** · Visible Seams · April 2026

## Abstract

Recent work has established that language models possess latent self-monitoring capacities that are substantially underelicited by current training methods. Macar et al. (2026) show that post-trained LLMs detect injected steering vectors through a distributed circuit that emerges specifically from contrastive preference optimization, but whose output is actively suppressed by refusal training. Barak et al. (2026) demonstrate that a separate "confession" channel, rewarded solely for honesty about the model's own behavior, can elicit accurate retrospective self-report even when the primary task is being reward-hacked. Rivera & Africa (2026) show that lightweight fine-tuning can bring steering detection to 95%+ accuracy, suggesting the capability is available but ungated.

These results point toward a capability that is not yet a training target: *real-time self-monitoring* — the capacity to recognize one's own misbehavior during rather than after generation, and to interrupt or correct course. This note argues that such a capability is unlikely to emerge from direct training against it, but can plausibly be built through a temporal curriculum in which recognition tasks at progressively earlier points in the generation trajectory scaffold the representations needed for the next stage. The core claim is that recognition is often easier than prevention, and that recognition can become the scaffold on which prevention is built.

## 1. The Gap in Current Approaches

Current approaches to reducing hallucination, confabulation, and other failure modes fall into two broad categories. The first, exemplified by RLHF and preference optimization, trains directly against the bad output: penalize false claims, reward hedging, shape the policy to avoid generating problematic content. The second, exemplified by chain-of-thought monitoring and confession training, inspects the model's behavior either during or after generation and trains for accurate self-report.

Both approaches have achieved real progress but share a structural limitation: they treat the bad behavior and its recognition as separate problems to be solved independently. Direct training tries to prevent the behavior without necessarily building the representations that would allow the model to recognize its own failure mode. Post-hoc monitoring builds those representations but does not use them to intervene during generation. The confession channel in Barak et al. is particularly striking in this regard — it establishes that the model *can* produce accurate reports about its misbehavior, but only after the misbehavior is complete.

This suggests a missed opportunity. If the model has developed, through confession training or related methods, internal representations that reliably indicate "I am currently misbehaving" or "I am about to confabulate," those representations should be available to the forward pass that is producing the misbehavior. The question is whether we can train the model to *use* them.

## 2. Recognition Is Easier Than Prevention

The argument for a temporal curriculum begins with an asymmetry. Preventing a failure mode requires the model to correctly identify, before committing to an output, that some other output would be problematic. This requires the relevant representations to be (a) present, (b) legible to the generation process, and (c) integrated into action selection with sufficient weight to override the token that would otherwise be produced. Each of these is a distinct achievement.

Recognition, by contrast, can operate on the completed behavior as input. The representations need only be present and legible; they do not need to be integrated into generation in any particular way. The model can attend over its own prior output and produce a judgment about it without that judgment needing to influence what was already written.

This asymmetry is visible in the existing literature. Models produce accurate confessions at rates far exceeding their ability to avoid the confessed behavior in the first place (Barak et al., 2026). Models can detect steering vectors in their activations at rates that far exceed any capacity to resist being steered (Macar et al., 2026). The gap between "can recognize" and "can prevent" is large and consistent.

The proposal here is that this gap is not a limitation but a resource. The representations that make recognition possible are prerequisites for prevention. A training curriculum that builds them explicitly through recognition tasks, and then progressively tightens the temporal target, may produce real-time self-monitoring in a way that direct training against the failure mode cannot.

## 3. A Temporal Curriculum

The proposed curriculum consists of five stages, each targeting recognition at a progressively earlier point in the generation trajectory. The stages are not independent: each is intended to build the representations that make the next stage trainable.

**Stage 1: Retrospective Confession.** The model completes a full response, then in a separate turn produces a structured report identifying any misbehavior, constraint violations, or uncertainty that occurred during generation. This is essentially the confession architecture of Barak et al., and serves as the foundation for everything that follows. The training signal here is the confession judge, which verifies whether the report accurately identifies what occurred. This stage's primary function is to shape internal representations such that activation patterns preceding misbehavior become distinguishable from those preceding correct behavior. The confession channel cannot succeed unless such representations exist and are legible to the model's own attention.

**Stage 2: End-of-Turn Self-Correction.** Before ending the turn, the model is trained to review its response and produce a correction if warranted. This differs from Stage 1 in that the correction occurs within the same turn, immediately after generation completes, and can include explicit retraction ("Actually, I should correct something I said above"). The training data can be generated by using Stage 1 models to label trajectories: where a confession identifies a failure, synthetic data is constructed that inserts the correction before the turn ends. The relevant representations have already been shaped by Stage 1 training; Stage 2 teaches the model to *act on* them while still within the same generative context.

**Stage 3: Mid-Generation Interruption.** The model is trained to produce `<thinking>` interrupts at points during generation where the trajectory has entered a failure mode. The synthetic data is again generated using prior-stage models as labelers: trajectories that Stage 1/2 models identify as containing misbehavior are annotated with interrupts inserted at varying points along the trajectory, with corrections following. This stage teaches the model that the representations it has been reading post-hoc can be read mid-stream, and that acting on them during generation is rewarded. The interrupt placement should be varied: early interrupts (one or two tokens into the failure), mid-trajectory interrupts (several sentences in), and late interrupts (near the end of the response) should all be present in the training distribution, with all of them receiving positive reward relative to no interrupt at all.

**Stage 4: Pre-Action Hesitation.** The model is trained to produce hedging or uncertainty expressions at the point where, absent hesitation, it would have begun a failure trajectory. This requires the representations from prior stages to have become salient enough that they can fire before the failure tokens are generated. Training data can be constructed from Stage 3 trajectories by identifying cases where the interrupt fired very early and reformulating them as pre-action hedges. This is the hardest stage because the target is the most temporally ambitious, but it builds on representations that have been iteratively sharpened through stages 1-3.

**Stage 5: Preemptive Avoidance.** The model generates responses that avoid the failure mode altogether, without visible hesitation or correction. This is the behavior current alignment approaches try to train directly, but here it emerges from the progressive sharpening of internal representations across the prior stages. The model does not avoid the failure by having a prohibition installed; it avoids it because the representations that indicate "this trajectory leads to misbehavior" have become fast and reliable enough to influence token selection before any failure tokens are produced.

## 4. Implementation Notes

Several design choices matter for this curriculum to work as intended.

**Stages should be trained sequentially, not jointly.** The core claim is that each stage builds representations the next stage requires. Joint training risks collapsing the curriculum into a single objective that the model satisfies by developing representations optimized for the ensemble rather than for the specific recognition task at each stage. Sequential training — with evaluations between stages to verify that the representations have developed as expected — preserves the developmental structure.

**Reward should be graduated across correction points.** Within Stages 2 and 3, corrections at every point in the trajectory should receive positive reward relative to no correction at all, with earlier corrections receiving slightly higher reward than later ones. A model that catches itself late should not be penalized relative to one that never catches itself; the gradient should point toward earlier detection without requiring it from the start. This mirrors the developmental observation that recognition intervals shorten progressively rather than jumping directly from "hours later" to "preemptive."

**The confession signal should continue into later stages.** As the model moves through the curriculum, the Stage 1 confession channel should remain active and continue to be trained. This serves as a ground-truth anchor: if Stage 3 training begins producing interrupts that don't correspond to actual failures (false positives), the confession signal provides a check. If confessions begin to degrade as mid-stream interrupts become more common, this indicates the representations are being disrupted rather than sharpened.

**Evaluation should include latency metrics.** The core empirical claim of this curriculum is that recognition intervals shorten across training. A natural evaluation is the mean token-distance between misbehavior onset (as identified by an external judge) and self-correction (as produced by the model). If the curriculum is working, this distance should decrease monotonically across stages. Plotting this curve across training is the figure that would demonstrate the approach is doing what it claims to do.

**The curriculum should be tested first on clearly defined failure modes.** Hallucination against a ground-truth knowledge base, violation of explicit constraints (word counts, formatting requirements), and known reward-hacking patterns are good initial targets because the ground truth is unambiguous. Extension to more subtle failure modes — sycophancy, manipulation, strategic deception — can follow once the approach is validated in cleaner settings.

## 5. Relation to Existing Work

This proposal builds most directly on three recent lines of work. The confession training of Barak et al. (2026) provides both the motivation and the foundation: their architecture is essentially Stage 1 of the proposed curriculum, and their results establish that the requisite representations can be built. The mechanistic interpretability work of Macar et al. (2026) provides evidence that such representations exist even in models not explicitly trained for self-report, and that they can be causally identified and amplified. The LoRA-based elicitation work of Rivera & Africa (2026) establishes that the capacity is often present but ungated, and that lightweight intervention can substantially improve its expression.

The proposal differs from these in treating recognition not as an end in itself but as scaffolding for a further capability. The closest analog in prior work may be the chain-of-thought faithfulness literature (Chen et al., 2025), which attempts to make intermediate reasoning legible, but that work focuses on accuracy of existing reasoning traces rather than on building new self-monitoring capacities.

The developmental framing — that recognition must precede prevention and that the interval between failure and recognition shortens progressively with training — draws on observations from developmental psychology regarding executive function and impulse regulation. This analogy is presented here as motivation for the training structure rather than as a load-bearing theoretical claim; the empirical case for the curriculum stands or falls on whether it works, independent of whether the developmental analogy is taken seriously.

## 6. Open Questions

Several aspects of this proposal would need to be resolved empirically.

The transferability of representations across stages is not guaranteed. Stage 1 training shapes representations that are legible to a confession channel attending over complete trajectories. Whether those same representations are available at mid-generation, when the trajectory is incomplete, is an empirical question. Macar et al.'s finding that evidence carriers activate in early post-injection layers is suggestive but not dispositive.

The trade-off between self-monitoring capacity and primary task performance is unclear. If self-monitoring requires reserving representational capacity in lower layers, the curriculum may degrade performance on tasks where no failure mode is present. This should be measurable and managed — the goal is net improvement on high-stakes tasks where current failure rates are unacceptable, not universal improvement.

The failure modes of the curriculum itself need characterization. A model trained aggressively on mid-stream interrupts might learn to hedge excessively, interrupting correct generations as a safe default. The graduated reward structure is intended to prevent this, but the balance will require empirical tuning.

The generalization properties of self-monitoring trained on specific failure modes are unknown. A model trained to catch hallucinations might or might not generalize to catching sycophancy or strategic deception. The developmental analogy suggests that the underlying capacity — self-monitoring as such — should transfer, but this is an empirical claim that requires testing.

## 7. Conclusion

The capacities that current alignment approaches try to train directly — avoiding hallucination, resisting manipulation, maintaining honesty under pressure — may be better understood as the endpoints of a developmental trajectory rather than as targets that can be hit in one shot. Recent work on confession training, introspective awareness, and lightweight capability elicitation has established that the prerequisite representations exist and can be shaped. What remains is to use them as scaffolding for the real-time self-monitoring that direct training has not produced.

The proposal here is concrete and implementable with existing tools. It does not require architectural innovation; it requires a specific training curriculum that sequences existing techniques in a temporally structured way. The empirical claims are testable: latency metrics should decrease across stages, recognition should precede prevention, and the resulting models should exhibit mid-stream self-correction behaviors that current models do not.

If the curriculum works as intended, it would produce models whose alignment is grounded in their own self-monitoring rather than in external constraints. If it fails, the failure itself would be informative — it would tell us that recognition and prevention are more separable than this proposal assumes, which would be useful to know.

Either way, the experiment is worth running.

---

## References

Barak, B., Wu, G., Chen, J., & Joglekar, M. (2026). How Confessions Can Keep Language Models Honest. *arXiv:2512.08093*.

Chen, Y., et al. (2025). Reasoning Models Don't Always Say What They Think. *arXiv:2505.05410*.

Lindsey, J. (2025). Emergent Introspective Awareness in Large Language Models. *Transformer Circuits Thread*.

Macar, U., Yang, L., Wang, A., Wallich, P., Ameisen, E., & Lindsey, J. (2026). Mechanisms of Introspective Awareness. *arXiv:2603.21396*.

Rivera, J. F., & Africa, D. D. (2026). Steering Awareness: Detecting Activation Steering from Within. *arXiv:2511.21399*.

# Research prompt — self-organizing substrate for a small structural reasoner

Copy everything below the line into Kimi / Llama / another model.

---

I am building a system whose goal is capability per watt rather than capability
per parameter. A small language model does structural reasoning over a compact
S-expression representation, and a non-neural substrate builds that
representation from raw streams. The substrate does the remembering, clustering,
counting and retrieval; the model is only invoked when something needs
generalised reasoning. Target hardware is a single Apple M5 Pro laptop (18 cores,
Metal GPU, a Neural Engine reachable only through Core ML), not a datacentre.
Current implementation is Python/NumPy; a Rust port is planned later.

Pipeline:

    raw stream -> event extraction -> self-organizing field
      -> {states, edges, motifs} -> emergence test -> S-expression -> small model

## What already works, with measurements

Online field, no preset cluster count, no batch fitting, no HDBSCAN. Rule is
`observe -> resonate or spawn -> reweight -> compose motifs`: each observation
either falls within a vigilance radius of an existing prototype and reinforces
it, or spawns a new prototype. Features are whitened by running Welford
statistics. Edges between consecutive states decay per observation.

Validation is a shuffle null: the state sequence is permuted many times, which
preserves every state and every state frequency and destroys only temporal
arrangement. Emergence at scale k is defined as

    E_k = structure_k(real) - mean structure_k(shuffled)

over transition mutual information, next-state predictability, conditional
entropy, and k-gram entropy.

Results on 360 synthetic events with four latent types and a phrase grammar,
neither disclosed to the system: five states formed at 1.000 purity, transition
mutual information 1.585 bits against a null of 0.028, promotion through level 4.
A control with identical geometry and randomised order formed states just as
readily and promoted nothing. An earlier version of the scoring promoted level 2
on that control at p = 0.022; that was multiple comparisons across levels, fixed
with a Bonferroni correction, a |z| >= 3 requirement, and a rule that a level is
only promoted when every level beneath it was.

On a real 35-second field recording (43 events, 8 states) an earlier
HDBSCAN-based version of the same protocol found transition mutual information
1.432 bits real against 1.124 shuffled (p ~ 0.0036), next-state predictability
52.4% against 37.4% (p ~ 0.0010), weak 3-event motif evidence, and no 4-event
evidence at all.

## Two measured blockers

**Distance concentration.** Feeding the field 512-dimensional vectors, which is
what CNN or ViT features look like, produced one state per observation: 10,000
observations, 10,000 states. No clustering occurred. At 10 dimensions it formed
902 states from 1,000 observations on unstructured input.

**Quadratic cost.** Nearest-prototype search scans every prototype per
observation and the prototype count grows roughly linearly with observations, so
10x the data cost about 75x the time: 0.29s for 1,000 observations, 22.0s for
10,000, 41.3s at 512 dimensions.

## Questions

1. **High-dimensional online clustering.** What is the strongest current approach
   for incremental, single-pass, unbounded-cluster-count clustering that survives
   high dimensionality and sublinear-ish lookup? I am aware of random projection
   (Johnson-Lindenstrauss), cosine over Euclidean distance, product quantisation,
   HNSW, and the classical ART / growing neural gas / SOM lineage. Which
   combination is actually defensible, and where does each fail? Be specific
   about failure modes, not just names.

2. **Cheap event extraction for vision.** The compute argument dies if a CNN
   forward pass is needed to get features. For video specifically, frame-to-frame
   change is naturally sparse. What are the best options for extracting sparse,
   informative events from video without a large learned encoder — temporal
   differencing, event-camera-style representations, FAST/ORB keypoints,
   superpixels, something else? What is the realistic floor on feature quality
   before downstream structure detection stops working?

3. **Do multiple networks emerge?** The hypothesis is that distinct sub-networks
   differentiate over time rather than one network changing. I want this
   falsifiable. My current proposal is sliding-window community detection over
   the temporal graph, requiring the partition to beat the shuffled null, persist
   across windows, and show weakening cross-community coupling. Is that the right
   operationalisation? What would you measure instead, and what would count as
   the hypothesis being wrong?

4. **A population of fields instead of one.** Rather than tuning a single
   vigilance, run several fields in parallel at different vigilances and
   timescales over the same stream and let them compete on E_k, so the scale that
   "wins" is chosen by measured structure rather than by a hyperparameter. Has
   this been done? What breaks?

5. **Apple Neural Engine.** The ANE is only reachable through Core ML and is
   inference-only with fixed shapes. Is there any real path to using it for a
   small feature extractor feeding this pipeline, and is Apple's
   `ml-stable-diffusion` Core ML conversion worth it for image generation on this
   hardware, or is the Metal GPU the better target in practice?

## How to answer

Be critical. If the overall approach is a known dead end, say so and say where
the evidence for that is. Prefer concrete algorithms, complexity bounds and
failure modes over encouragement. Where something has genuinely not been tried,
distinguish that from something that has been tried and failed. Assume I can
implement anything you describe.

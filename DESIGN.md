# Pilot design and explicit approximations

## Inspected references

The pilot was designed against these read-only references:

- DAPD, commit `05727b08da4cb4008a275123d7d9885dd5714f7c`:
  `dapd/core.py`, `dapd/dream_core.py`, `dapd/dream_generation.py`, and the
  Dream lm-eval wrapper.
- Official DAWN Dream inference, commit
  `19c32c28b5bf0475ccdfad853c74fc885f6410cd`:
  `dream/model/generation_utils.py`.
- Existing local Dream experiments in `safety-defusion-convergence`, used only
  to cross-check Dream's shifted logits, linear transfer schedule, and GSM8K
  answer extraction.
- Dream-v0-Instruct-7B is pinned to Hugging Face revision
  `05334cb9faaf763692dcf9d8737c642be2b2a6ae`.
- FlowBlock, arXiv `2607.17652`, and its public implementation were inspected
  for the distinction between window size, per-token commit confidence, and
  frontier spawn readiness.
- Mean-Field Parallel Decoding, arXiv `2606.15805`, was inspected for its exact
  predictive-overlap definition and within-active-set normalization.

The DAPD implementation is not copied into this repository. Vast setup checks
out the exact DAPD commit, and the pilot imports its Dream Q/K hook,
`shift_logits_dream`, and `build_dependency_graph` directly.

## What is faithful

- Generated positions alone are partitioned into fixed contiguous regions.
- Model visibility is unchanged. Every forward contains the prompt, every
  committed response token, and every remaining response mask.
- DAPD's Dream dependency is used unchanged: post-RoPE Q/K from the top 30% of
  layers, head/layer averaging, symmetric attention, diagonal removal,
  masked-pair filtering, and row-max normalization.
- Both the raw symmetric token matrix and DAPD's normalized matrix are saved.
- The Dream shifted-logit convention is preserved.
- `entropy`, `topk_margin`, and `maskgit_plus` use the ordinary Dream token
  sampling/confidence definitions. The default is Dream's `entropy` policy.
- A single global forward supplies predictions for every masked position. Only
  scheduler-selected regions apply a commitment.
- Region graph edges are oriented from lower to higher response position.

## Researcher-specified choices

These came directly from the requested pilot: fixed region sizes 16/32/64;
mean, top-10%-mean, and power-mean aggregators; gamma 2; a thresholded
undirected graph with positional parent orientation; sequential, always-on,
and lag 0/1/2/4 modes; dynamic graph recomputation; GSM8K; accuracy/NFE/time
metrics; and unchanged attention visibility.

## Implementation choices introduced for this pilot

These are not claims from DAPD or Dream:

1. **A local clock indexes a local linear transfer schedule.** Dream has no
   explicit diffusion-timestep input in this sampler. Its `t -> s` schedule
   only controls how many masks are committed. Therefore a regional clock is a
   regional reveal-budget clock, not a distinct neural-network noise time.
2. **`local_steps=region_size`.** With a 32-token region, its local transfer
   budget is 32 steps. Eight-region sequential execution therefore has the
   same 256-NFE budget as vanilla Dream, while selecting only inside the active
   region. Always-on executes those eight local schedules concurrently. The
   pinned checkpoint's vanilla decoder itself is global/full-canvas, not block
   sequential.
3. **Schedule cursor and backpressure clock are separate.** The official
   linear schedule can produce `int(...) == 0`, especially on its first step
   because `eps > 0`. An eligible attempt always advances the internal
   `schedule_step`, preventing the region from becoming stuck at that rounded
   zero. The backpressure `clock` advances only if at least one new token was
   actually committed. Children therefore cannot consume a parent's
   zero-information transition as progress.
4. **Completed-parent release.** Strictly enforcing
   `parent.clock >= child.clock + L` forever prevents a child from completing
   its last `L` steps once its parent stops at `local_steps`. The default treats
   a completed parent as no longer exerting backpressure. Set
   `release_completed_parents: false` to request strict semantics; a detected
   deadlock is raised explicitly.
5. **Raw token dependencies are aggregated by default.** DAPD first
   symmetrizes raw attention, then row-normalizes it for token graph selection.
   Row normalization destroys symmetry. The pilot defaults to the raw matrix
   so region scores retain the requested symmetric meaning; `matrix:
   normalized` is available as an ablation.
6. **Default region graph choice is top-10%-mean at threshold 0.005.** This is
   an uncalibrated pilot setting, not a published value. All three region
   matrices are logged at every saved dependency snapshot so this choice can
   be audited immediately on Vast.
7. **Graphs use only currently masked generated tokens.** Region aggregation
   excludes resolved positions rather than averaging DAPD's zero-filled
   non-mask entries into the denominator. As masks disappear, cross-region
   scores can vanish and topology can change. The last graph is retained
   between recomputation intervals.
8. **Always-on does not extract dependencies.** It is the zero-backpressure
   compute control. `async_lag0` does extract/recompute the graph but should
   have the same commitment schedule at temperature zero; their NFE/output
   comparison checks that the hook is observational, while wall time exposes
   dependency overhead.
9. **Fixed-canvas EOS handling.** Like ordinary Dream decoding, all canvas
   positions are filled, then display/evaluation truncates at the first
   EOS/EOT. Both canvas-token and effective-token counts are recorded.
10. **GSM8K is zero-shot chat-formatted.** The prompt asks for a final
    `#### <number>`. This is a fast hypothesis test and is not asserted to
    reproduce DAPD's paper prompting or published GSM8K scores.
11. **Dependency timing is instrumented-forward time, not isolated overhead.**
    DAPD obtains Q/K during the same model forward and then reconstructs
    attention. `dependency_seconds` measures those complete dependency-enabled
    forwards. It does not claim to subtract the counterfactual plain-forward
    cost. Diagnostic file/plot time is measured separately and excluded from
    reported decoding wall time.
12. **Graph density is a first-stage go/no-go diagnostic.** Every dependency
    snapshot records average degree, fraction of possible edges present,
    connected components, every region's parent count, root count, and maximum
    parent count. Density, degree, and component metrics are reported both over
    all fixed regions and over still-masked active regions, preventing completed
    isolated regions from creating misleading late-stage sparsity. The
    two-example probe recomputes this every forward. A dense
    graph that positional orientation turns into a near-total order is retained
    as a negative result; the scheduler does not sparsify it behind the scenes.
13. **The new wavefront probe is admission-only, not backpressure.** It begins
    with R0 and admits at most one next positional region per global iteration.
    Up to `W=8` admitted, unfinished regions evolve concurrently. There is no
    dependency-based pausing or clock merging yet; adding it before verifying
    graph dynamics would confound the diagnostic.
14. **The 15% gate follows FlowBlock's acceptance-ratio shape.** For each
    frontier region, readiness is the fraction of its currently masked tokens
    whose top-token probability is at least `0.5`. The next region is admitted
    if that fraction is at least `0.15`. The `0.5` cutoff is FlowBlock's
    reported math commit threshold; Dream's actual commit rule remains entropy
    scheduling. This is a cross-model approximation and is labelled as such.
15. **Graph interval defaults to `K=4`.** A 32-step local schedule then yields
    roughly eight to ten snapshots over a wavefront decode. This is a starting
    diagnostic resolution, not a tuned value. If edge turnover occurs entirely
    between snapshots, compare K=2; if graphs barely change, compare K=8.
16. **Mean-Field interaction is used as a dependency proxy, not as its paper's
    commit policy.** The paper defines
    `D_ij = 1 - JSD(pi_i, pi_j) / ln(2)`, max-normalizes `D` within the masked
    active set, and then uses it as an inhibitory term in a mean-field commit
    update. This pilot stops after `D`: it aggregates token pairs into the same
    fixed regions as DAPD and never applies the paper's mean-field commit rule.
17. **All-canvas JSD is explicitly approximate.** Exact pairwise JSD costs
    `O(m^2 |V|)` and the paper controls it with a 20-token active block. With up
    to 256 masks, the probe instead compares each pair on the union of the two
    top-256 supports plus a shared residual bucket. This coarsening can
    overestimate similarity when the omitted tails differ. Each snapshot logs
    mean/min/max retained top-k mass so the approximation can be rejected if
    needed.
18. **Mean-Field thresholds are swept, not pretended to be calibrated.** The
    diagnostic writes graphs at 0.50, 0.70, 0.80, 0.90, and 0.95. The 0.90 graph
    is only the initial visualization/combination choice. Region-level
    Mean-Field graphs use mean aggregation because top-10%-mean saturated on
    the first probe; DAPD retains top-10%-mean at its independently scaled
    0.005 threshold. Raw matrices from these unlike scales are never averaged.
19. **Signal combinations are boolean and auditable.** At the DAPD threshold
    and Mean-Field 0.90 threshold, the probe logs both union (either signal) and
    intersection (both signals). This avoids inventing an unjustified numeric
    weighting between attention scores near 0.005 and JSD similarities in
    `[0,1]`.
20. **“New dependency” is observable.** For every signal and snapshot, logs
    include added/removed region edges and Jaccard overlap with the preceding
    edge set. Active-region membership is also retained because an edge removed
    solely when one region finishes is not evidence that semantic dependence
    disappeared.
21. **The aggressive admission-only control is now explicit.**
    `loose_wavefront` uses W=8 and theta_spawn=0.15 but no positional or
    dependency backpressure. It directly tests whether GSM8K made a large,
    loose wavefront look safer than it is.
22. **Mean-Field is a separate commit-policy baseline.** `mean_field_repro`
    follows published Algorithm 1 with exact full-vocabulary JSD within one
    sequential active block, tau=0.90, R=2, and block size 32. It must not be
    interpreted as our JSD region graph: the former selects tokens to commit;
    the latter only adds scheduler backpressure.
23. **Mean-Field source and empty-set behavior are unresolved upstream.** The
    paper's linked repository returns 404, and its pseudocode does not define a
    zero-token commit case. The reproduction selects the maximum-q token only
    in that case, guaranteeing progress, and reports how often this choice was
    exercised. A high fallback count invalidates claims of paper fidelity.
24. **The first cross-dataset extension is math-only.** MATH-500 and ASDiv
    overlap FlowBlock's released benchmark suite and can be scored without
    executing model-generated programs. MATH-500 uses FlowBlock's pinned
    `math-verify==0.9.0`; code benchmarks remain deferred because their
    evaluators execute untrusted generated Python.
25. **A reduced-step global Dream sweep is required before scheduler claims.**
    The original vanilla row used 256 steps, while concurrent 32-step region
    schedules completed in roughly 71 global forwards. The new vanilla sweep
    holds the 256-token canvas and commit policy fixed and tests 128, 96, 72,
    64, and 32 global steps. The 72-step row is the primary compute-matched
    control; regional scheduling is useful only if it lies above this global
    accuracy/compute frontier.

## Decision rule after the dynamic graph probe

Continue to dependency-driven clock grouping only if at least one signal shows
repeatable, non-adjacent edge births that persist for more than one snapshot
and are not explained by regions completing. Prefer the intersection graph if
DAPD and predictive overlap agree; use the union only as a recall-oriented
ablation. If Mean-Field graphs are dense while retained top-k mass is low, first
increase top-k or reduce the compared active set rather than tuning the graph
threshold around an unreliable tail approximation.

If useful structure appears, the next scheduler should group connected
regions into one clock domain from that point forward while leaving unrelated
domains concurrent. Because Dream commitments are irreversible, grouping can
only synchronize future reveal budgets; it cannot revise tokens already
committed before the dependency appeared.

## First controlled scheduler

The implemented follow-up deliberately avoids round robin and graph coloring.
Every admitted region normally advances. For an oriented control edge
`parent -> child`, actual progress is the number of revealed tokens. The child
is paused only if it gets ahead of the parent; the parent is paused only if its
lead exceeds eight tokens. This is a loose bounded-skew controller, not clock
equality. Pausing a leader forces service of the lagging endpoint but does not
force Dream to reveal an extra low-confidence token.

Step-zero dependencies are diagnostic only. A dynamic edge becomes active
after both endpoints have nonzero revealed-token progress and two consecutive
observations; two consecutive misses remove it. Immediate positional-neighbor
edges supply the initial pipeline. The comparison holds this scheduler fixed
and varies only the dynamic graph source: DAPD, Mean-Field/JSD mean, or their
intersection. The loose readiness-only wavefront remains a diagnostic; the
50-example attribution control is the position-only bounded-skew strategy
described below.

For the 50-example evaluation, that loose wavefront is replaced as the
competitive admission comparison by `flowblock_proxy`: at most two unfinished
regions (`W=2`), a frontier readiness threshold of `0.60`, and per-token
readiness at maximum probability `0.50`. This is not official FlowBlock: the
proxy cannot reproduce its T2T editing, block-causal KV cache, or commit rule
without changing the model/sampler beyond this pilot. The exact settings and
the approximation name are written into every result row and metadata file.

The evaluation also includes `controlled_position`, which removes all dynamic
dependencies while holding the W=8, 15% admission gate, and positional
bounded-skew controller fixed. This was added to prevent a scheduler-only gain
from being attributed to DAPD or JSD. `controlled_jsd` uses a plain Dream
forward at graph observations and therefore does not pay the DAPD Q/K hook
cost; `controlled_combo` necessarily computes both signals.

Reported speed has three views. NFE measures full-canvas model calls; canvas
TPS divides the fixed 256-token canvas by measured decoding time; effective
TPS counts only tokens before EOS/EOT. Required dependency computation is in
decoding wall time. Diagnostic file and plot creation is separately measured
and excluded. The summary reports vanilla-relative NFE, wall-clock, and canvas
TPS speedups rather than inferring throughput from NFE alone.

## Mathematical limitation

Dream commitments are irreversible in the ordinary entropy/margin/MaskGIT+
decoder. A masked position's prediction can evolve on every global forward,
but after the policy commits it, that token does not diffuse or revise again.
Thus “regions evolve concurrently” means their unresolved positions are
re-predicted concurrently under changing global context. It does not mean
already committed regional tokens are continuously refined.

Because the model is not conditioned on a global diffusion timestep, there is
no conflict from passing several neural timesteps in one batch. Conversely,
the proposed clocks cannot have the stronger interpretation of independently
conditioning the pretrained network at different noise times. This pilot
tests asynchronous commitment budgets and backpressure—the closest faithful
interpretation available without changing Dream or training.

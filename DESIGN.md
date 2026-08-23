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

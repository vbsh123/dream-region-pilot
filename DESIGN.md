# Current design

This repository tests one narrow question: can fixed regions of a masked Dream
generation canvas make concurrent commitment progress without changing model
visibility or training?

## Invariants

- Prompt tokens are never part of generated regions.
- Every model forward sees the ordinary full Dream canvas.
- Committed Dream tokens are irreversible.
- Each region's transfer quota follows the same linear mask-to-token schedule
  as Dream, evaluated against that region's own schedule cursor and remaining
  masks.
- A region's progress is its number of actually revealed tokens. A zero-token
  transfer may consume a local schedule point, but it is not counted as real
  progress for backpressure.
- Token choice within a region uses Dream's configured confidence policy.
- Tail guarding, confidence deferral, and positional backpressure affect only
  which regional updates are applied after a shared forward.

## Positional controller

`controlled_position` starts with the leftmost region and admits at most one
new region after each forward when the current frontier passes the configured
readiness ratio. All admitted, unfinished regions may then update concurrently.

Adjacent admitted regions are coupled by revealed-token counts:

- a child that has revealed more tokens than its parent is paused;
- a parent whose lead reaches the configured limit is paused so the child can
  catch up;
- completed endpoints no longer constrain unfinished neighbors.

The always-on coupled-deferral variant uses the same adjacent positional
backpressure but admits every region at iteration zero.

## Startup deferral

For coupled-deferral strategies, the ordinary per-region transfer proposal is
constructed first. If its least-confident selected token is below the threshold,
the update may be deferred. A lagging endpoint at the positional gap bypasses
that confidence gate, and a bounded globally empty streak forces one update to
avoid a fixed-point deadlock. If `deferral_until_revealed_tokens` is set, this
confidence test applies only during the first configured number of real token
revelations in each region.

## Tail guard

From the current logits, the decoder finds the earliest still-masked position
whose top-1 prediction is a stop token. Its containing fixed region is treated
as the provisional tail. While any earlier region remains unfinished, that
region and every region to its right are withheld from commitment. Their logits
are still calculated by the shared full-canvas forward.

This is deliberately a coarse ablation, not a semantic answer detector.

## External decoding baselines

`mean_field_repro` is isolated from regional control. It implements the
published exact-JSD Mean-Field selection within one sequential active block.

DAWN modes are also separate token-selection experiments. The official DAWN
fork provides its averaged late-layer attention together with logits; the
regional harness can apply its selector independently within scheduled regions.

Inter-region DAPD/JSD graph construction, graph persistence, dynamic component
pooling, and fixed-lag graph scheduling were explored and then removed from the
active code because they added substantial complexity and overhead without a
useful accuracy gain in the pilot.

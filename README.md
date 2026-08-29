# Dream regional decoding pilot

A small, training-free research harness for testing regional commitment
schedules with `Dream-org/Dream-v0-Instruct-7B`. Model visibility is unchanged:
every forward still processes the full prompt and masked generation canvas.

The abandoned inter-region DAPD/JSD graph experiments have been removed. The
standalone Mean-Field reproduction remains because it is an external decoding
baseline, not an inter-region graph controller. DAWN remains as a separate
token-selection baseline and regional selector ablation.

## Core execution path

For a 256-token canvas with `region_size: 32`, `build_fixed_regions` creates
eight response-only regions. Each global iteration then does:

1. one full-canvas Dream forward;
2. optional tail prediction and positional admission/backpressure;
3. an ordinary Dream local transfer count for each scheduled region;
4. confidence-ranked commitment inside each scheduled region;
5. updates to actual revealed-token progress and the local schedule cursor.

The most useful code-reading path is:

1. `run_gsm8k.py`: CLI, benchmark loop, model loading, and strategy dispatch;
2. `decoding.py::decode_regional`: full decoding loop;
3. `scheduler.py::RegionScheduler`: admission and positional backpressure;
4. `commit.py::commit_active_regions`: per-region transfer quota and token
   choice;
5. `regions.py`: fixed response-position regions;
6. `diagnostics.py`: commitment and regional state logs.

## Strategies

- `vanilla` and `vanilla_steps{32,64,72,96,128}` call Dream's native global
  decoder.
- `fixed_sequential` completes fixed regions left-to-right.
- `always_on` schedules every unfinished region from iteration zero.
- `always_on_tail_guard` keeps the predicted terminal region and everything to
  its right masked while an earlier region remains unfinished.
- `always_on_bounded_defer{,_tail_guard}` may delay a low-confidence local
  update for a bounded number of iterations.
- `always_on_coupled_defer{,_tail_guard}` starts every region immediately,
  applies startup confidence deferral, and uses adjacent positional
  backpressure to bound revealed-token drift.
- `always_on_coupled_defer_stop_filter` lets only the predicted terminal
  region backfill a stop proposal with non-stop tokens whose proposed-token
  probability clears a configurable threshold.
- `always_on_coupled_defer_stop_defer` instead skips the whole regional update
  when a position in its ordinary selected quota has a stop token as raw
  top-1 or as the actual sampled proposal; it does not backfill.
  In both variants, later regions remain excluded, stop stalls override the
  normal maximum-gap force, and a committed stop latches the endpoint so its
  suffix does not need to be decoded.
- `loose_wavefront` admits at most one new region per iteration with the
  configured confidence-readiness gate.
- `controlled_position{,_tail_guard}` combines that admission rule with
  adjacent positional backpressure.
- `flowblock_proxy` is only a Dream-side admission proxy (`W=2`, spawn
  threshold `0.60` by default); it is not official FlowBlock.
- `mean_field_repro` reproduces the published Mean-Field Algorithm 1 inside
  sequential active blocks using exact JSD over the vocabulary.
- `official_dawn` calls the pinned official DAWN decoder.
- the regional strategy names containing `dawn` use DAWN token selection with
  this harness's regional scheduling.

## Vast setup

A 48 GB GPU is sufficient for the batch-size-one experiments used here.

```bash
git clone https://github.com/vbsh123/dream-region-pilot.git
cd dream-region-pilot
bash scripts/setup_vast.sh
source .venv/bin/activate
```

The setup script installs the package and creates pinned source-only checkouts
for DAWN, FlowBlock, and OpenAI HumanEval. DAPD is no longer downloaded or
required.

## Run the 50-example GSM8K-CoT comparison

```bash
source .venv/bin/activate
bash scripts/run_gsm8k_50.sh outputs/gsm8k_regional_50
```

Or select only the leading regional methods:

```bash
python -m dream_region_pilot.run_gsm8k \
  --config configs/gsm8k_cot_official_50.yaml \
  --output-dir outputs/gsm8k_leading_50 \
  --limit 50 \
  --strategies \
    vanilla \
    always_on_coupled_defer_tail_guard \
    always_on_coupled_defer_stop_filter \
    always_on_coupled_defer_stop_defer \
    controlled_position_tail_guard \
  --probe-window 8 \
  --spawn-readiness 0.15 \
  --readiness-confidence-threshold 0.5 \
  --max-progress-gap 4 \
  --deferral-confidence-threshold 0.4 \
  --stop-filter-confidence-threshold 0.4 \
  --deferral-until-revealed-tokens 2 \
  --max-global-deferral-iterations 4 \
  --diagnostic-examples 3
```

`configs/gsm8k_cot_official_50.yaml` uses Dream's released zero-shot
lm-eval-style GSM8K-CoT prompt and flexible extraction. The paired harness
reseeds every strategy for every example; it is suitable for attribution but
does not consume one continuous RNG stream exactly like upstream lm-eval.

For a separate comparison under DAWN's released Dream GSM8K baseline protocol
(lm-eval 0.4.8 `gsm8k`, five train demonstrations, no chat template, BOS prefix,
temperature zero), run:

```bash
python -m dream_region_pilot.run_gsm8k \
  --config configs/gsm8k_dawn_5shot.yaml \
  --output-dir outputs/gsm8k_dawn_protocol_terminal_50 \
  --limit 50 \
  --strategies \
    vanilla \
    always_on_coupled_defer_tail_guard \
    always_on_coupled_defer_stop_filter \
    always_on_coupled_defer_stop_defer \
  --diagnostic-examples 0
```

This protocol is intentionally separate from the official Dream zero-shot CoT
configuration. Its five-shot sampler uses lm-eval's seed 1234 and advances once
per test document, including skipped indices in targeted diagnostic runs.

## Other benchmarks

The same runner supports ASDiv, MATH-500, and HumanEval through their config
files. For example:

```bash
python -m dream_region_pilot.run_gsm8k \
  --config configs/humaneval_50.yaml \
  --output-dir outputs/humaneval_regional \
  --limit 164 \
  --strategies vanilla always_on always_on_tail_guard \
    always_on_coupled_defer_tail_guard \
  --max-progress-gap 4 \
  --deferral-confidence-threshold 0.4 \
  --deferral-until-revealed-tokens 2 \
  --max-global-deferral-iterations 4
```

HumanEval executes generated code in a child process. Use a disposable worker.

## Outputs

- `results.jsonl`: one full record per example and strategy;
- `summary.json`: accuracy/pass@1, NFE, wall time, throughput, speedups, and
  scheduler event averages;
- `metadata.json`: resolved config, model revision, and implementation notes;
- `diagnostics/example_*/<strategy>/iterations.jsonl`: scheduled/advanced
  regions, commitments, tail guard, and controller state;
- `commitments.jsonl`: chosen tokens with raw and sampling top-1/top-2
  probabilities;
- `region_state.csv` and `clocks_and_masks.png`: progress over time.

Wall time synchronizes CUDA before and after timed GPU work. Diagnostic token
conversion and file/plot generation are measured separately and excluded from
the reported decoding time.

## Static verification

No model is needed for the unit tests:

```bash
pytest -q
bash -n scripts/*.sh
```

# Dream fixed-region asynchronous decoding pilot

Minimal, training-free research code for testing whether fixed generated-token
regions can follow separate Dream commitment clocks under dependency-defined
backpressure.

Status: implemented locally, deliberately **not run**. No model, dataset,
environment setup, test, build, or experiment was executed on this PC.

## GSM8K protocol correction

Use `configs/gsm8k_cot_official_50.yaml` for every new GSM8K run. It mirrors
Dream's released zero-shot `gsm8k_cot` evaluation: `Q: ...\nA:`, the model chat
template, 256 canvas tokens and steps, temperature 0.1, top-p 0.9, entropy
selection, lm-eval stop strings, and both strict and flexible answer filters.
The package also pins Dream's officially tested Transformers 4.46.2 stack.
For fair paired comparisons, the pilot deliberately resets seed
`1234 + example_index` before each strategy; upstream lm-eval seeds once, so
the two paths need aggregate agreement rather than bit-identical samples.
The older `configs/gsm8k_50.yaml` is retained only to interpret completed
artifacts; its custom `####` instruction with deterministic sampling caused
premature direct-answer/EOS collapse and is not a valid published-Dream
baseline.

## Expanded baseline and dataset probe

### First priority: global Dream step sweep

Before interpreting any regional speedup, compare against the same official
Dream decoder with fewer global denoising steps. The 256-token canvas, prompt,
seed, entropy policy, and attention visibility remain unchanged; only `steps`
varies. In particular, `vanilla_steps72` is compute-matched to the earlier
regional mean of roughly 71 NFEs.

Run the two-example smoke test on Vast:

```bash
bash scripts/run_vanilla_step_sweep.sh \
  outputs/gsm8k_vanilla_step_sweep_2 2
```

Then run 50 examples:

```bash
bash scripts/run_vanilla_step_sweep.sh \
  outputs/gsm8k_vanilla_step_sweep_50 50
```

The sweep is `256, 128, 96, 72, 64, 32`. A regional method supports the
hypothesis only if it improves the accuracy/NFE or accuracy/wall-time frontier
over this global sweep. `vanilla_steps32` versus `always_on` will later isolate
global confidence allocation from equal per-region reveal budgets.

The next run separates the result we actually need to explain:

```text
loose_wavefront       W=8, theta=0.15, no progress-skew or graph control
controlled_position   same admission plus adjacent positional bounded skew
controlled_dapd       positional control plus dynamic DAPD edges
controlled_jsd        positional control plus dynamic JSD edges
controlled_combo      positional control plus persistent DAPD intersection JSD
controlled_dapd_dynamic  DAPD components with pooled commitment selection
controlled_jsd_dynamic   JSD components with pooled commitment selection
controlled_combo_dynamic DAPD-intersection-JSD components with pooled selection
mean_field_repro      paper Algorithm 1 commit policy, exact JSD per active block
flowblock_proxy       Dream-side W=2, theta=0.60 admission proxy
vanilla               official Dream diffusion_generate
```

The dynamic pooled variants are a separate ablation. At each dependency graph
refresh they rebuild connected components among active fixed regions. Every
component receives the sum of its members' ordinary local Dream transfer
counts, but chooses those tokens jointly across the component instead of
guaranteeing a separate quota to every member. Existing positional
backpressure still pauses an outrunning region. Graph splits and merges retain
real token progress and never remask revealed tokens.

Use `--example-indices` to rerun a particular paired failure without changing
its dataset index or `1234 + example_index` seed.

`loose_wavefront` is the missing ablation. If it also reaches roughly the same
accuracy/NFE as `controlled_position`, the large-window loose gate—not bounded
skew or dependencies—explains the earlier result. If it drops while position
or dependency control survives, the controller is doing real work.

The Mean-Field paper's code link currently points to
`github.com/AmeenAli/MDP`, which returns 404. `mean_field_repro` is therefore a
paper-faithful reproduction, not official code: exact full-vocabulary JSD,
log-top1/top2 unary margin, two mean-field updates, and threshold 0.90 inside
sequential 32-token active blocks. The one choice absent from its pseudocode is
deadlock handling: if no intensity crosses the threshold, the reproduction
commits the maximum-intensity token and logs
`mean_field_forced_progress_events`.

After pulling on Vast, refresh the environment once (this also checks out the
official FlowBlock source without installing its incompatible environment):

```bash
cd /workspace/dream-region-pilot
git pull
bash scripts/setup_vast.sh
```

Run two examples on each distribution first:

```bash
bash scripts/run_expanded_probe.sh configs/gsm8k_cot_official_50.yaml \
  outputs/gsm8k_expanded_probe_2
bash scripts/run_expanded_probe.sh configs/asdiv_50.yaml \
  outputs/asdiv_expanded_probe_2
bash scripts/run_expanded_probe.sh configs/math500_50.yaml \
  outputs/math500_expanded_probe_2
```

If those complete, run 50 examples of the new datasets:

```bash
bash scripts/run_reasoning_task_50.sh configs/asdiv_50.yaml \
  outputs/asdiv_expanded_50
bash scripts/run_reasoning_task_50.sh configs/math500_50.yaml \
  outputs/math500_expanded_50
```

The focused region-balance attribution can be run identically on GSM8K and
HumanEval:

```bash
bash scripts/run_balanced_attribution_50.sh configs/gsm8k_cot_official_50.yaml \
  outputs/gsm8k_balanced_attribution_50
bash scripts/run_balanced_attribution_50.sh configs/humaneval_50.yaml \
  outputs/humaneval_balanced_attribution_50
```

HumanEval uses the official OpenAI tests and deterministic pass@1. It executes
model-generated Python in a time-limited child process with the evaluator's
reliability guard. That guard is not a security sandbox: run this benchmark
only on a disposable Vast worker without secrets or trusted writable data.

MATH-500 uses `math-verify==0.9.0`, matching FlowBlock's pinned evaluator
dependency. ASDiv uses normalized numeric/fraction comparison when the gold is
numeric and normalized final-text comparison for its categorical answers.
These are the two additional math datasets shared with FlowBlock's official
suite.

## Minimal implementation plan

1. Use Dream-v0-Instruct-7B and its ordinary shifted logits and confidence
   commit rule.
2. Build response-relative fixed regions of size 16, 32, or 64.
3. Import the pinned public DAPD Dream attention hook and token dependency
   construction.
4. Aggregate every saved token matrix with mean, top-percent mean, and
   power mean; threshold the selected matrix into a positional DAG.
5. Compare vanilla Dream, fixed sequential regions, always-on regions, and
   fixed-lag regions at lags 0/1/2/4.
6. Write per-example JSONL, aggregate CSV/JSON, full dependency NPZ files,
   region CSVs, graph JSON/PNG, and clock/mask plots.

See [DESIGN.md](DESIGN.md) before interpreting results. It separates requested
design from choices introduced by this implementation and explains the exact
local-clock limitation.

## Recommended next step: dynamic graph probe

The next experiment is deliberately diagnostic-only. It starts a positional
wavefront with eight 32-token regions (`W=8`), admits at most one new region per
global iteration, and uses a loose FlowBlock-style admission threshold:

```text
token is ready:       max vocabulary probability >= 0.50
frontier is ready:    ready remaining masks / remaining masks >= 0.15
graph snapshot:       every K=4 global forwards
```

`0.15` is therefore the spawn fraction (`theta_spawn`), not “15% of the block
has already been committed.” The token commit policy remains Dream entropy;
confidence is used only to admit the next region. `W=8` is the maximum window,
so it does not bind once all eight regions have been admitted.

At every graph snapshot the same forward supplies DAPD attention scores,
Mean-Field-style predictive-overlap scores
`1 - JSD(p_i, p_j) / ln(2)`, and thresholded region graphs for each signal.
Union and intersection graphs are also written at the configured primary
thresholds. None of these graphs changes scheduling or commitments in this
phase. Clock merging is a later experiment.

Run two examples on Vast:

```bash
cd /workspace/dream-region-pilot
bash scripts/run_dynamic_graph_probe.sh outputs/gsm8k_dynamic_graph_probe_2
```

Inspect:

```text
outputs/gsm8k_dynamic_graph_probe_2/diagnostics/example_000/wavefront_probe/graph_metrics.csv
outputs/gsm8k_dynamic_graph_probe_2/diagnostics/example_000/wavefront_probe/graph_metrics_over_time.png
outputs/gsm8k_dynamic_graph_probe_2/diagnostics/example_000/wavefront_probe/graph_timeline.json
outputs/gsm8k_dynamic_graph_probe_2/diagnostics/example_000/wavefront_probe/region_state.csv
```

For a compact terminal comparison:

```bash
python scripts/inspect_dynamic_graphs.py \
  outputs/gsm8k_dynamic_graph_probe_2/diagnostics/example_000/wavefront_probe/graph_timeline.json
```

`graph_metrics.csv` is long-form by `iteration,signal`. It records edge
additions/removals, edge-set Jaccard similarity to the previous snapshot,
density, degree, components, parent counts, and adjacent versus non-adjacent
edges. The NPZ files retain both full token-token matrices.

The Mean-Field paper computes full-vocabulary JSD at cost
`O(masked_tokens^2 * vocabulary)`, while its practical experiments bound the
active block (reported block size 20). Applying that literally to all 256
canvas masks is a poor 4090 pilot. The default therefore uses a transparent
`topk=256` approximation: for each token pair it computes JSD on the union of
their top-k supports plus one shared residual-tail bucket. Every graph JSON
logs top-k probability mass and `paper_exact: false`. The shared tail can
overstate similarity; do not interpret a dense Mean-Field graph without first
checking retained mass.

## Controlled scheduling probe

After inspecting the dynamic graphs, run the first scheduling comparison:

```bash
cd /workspace/dream-region-pilot
bash scripts/run_controlled_scheduler_probe.sh \
  outputs/gsm8k_controlled_comparison_probe_2
```

This runs the same two examples under:

```text
vanilla               official Dream decoder
flowblock_proxy       W=2 and 60% readiness admission proxy
loose_wavefront       W=8 and 15% readiness, no graph/backpressure
mean_field_repro      published Mean-Field commit algorithm reproduction
controlled_position   W=8 positional bounded skew, no dynamic graph
controlled_position_tail_guard
                      same controller, provisional terminal region deferred
controlled_dapd       persistent DAPD edges
controlled_jsd        persistent Mean-Field/JSD mean edges
controlled_combo      persistent DAPD ∩ JSD edges
```

There is no round robin or graph coloring. All admitted, unblocked regions are
normally serviced together. Edges are oriented by response position. A child
is paused if it reveals more tokens than its parent; a parent is paused only
when its lead exceeds eight tokens. The lagging endpoint then receives its
ordinary next Dream local update. No extra token is forced, and a zero-token
update does not count as progress.

`controlled_position_tail_guard` is a deliberately coarse termination
ablation. On every forward it finds the earliest still-masked position whose
top-1 prediction is a stop token. While any earlier fixed region is unfinished,
the region containing that position and all later regions receive no forced
local update. The guard is removed once the earlier regions finish. This uses
only current model predictions; it does not use the reference answer or a
vanilla generation to infer response length.

`always_on_tail_guard` applies that identical termination rule without a
wavefront: all eight regions start on iteration zero, with no readiness
admission and no positional or dependency backpressure. Only a provisional
tail region and the regions after it can be paused.

Run the complete 164-task HumanEval split with paired baselines:

```bash
python -m dream_region_pilot.run_gsm8k \
  --config configs/humaneval_50.yaml \
  --output-dir outputs/humaneval_164_always_on_tail \
  --limit 164 \
  --strategies vanilla always_on always_on_tail_guard \
  --diagnostic-examples 0
```

The all-mask graph cannot control scheduling. A dependency edge activates only
after both endpoints have revealed at least one token and it appears in two
consecutive `K=4` snapshots; removal likewise requires two misses. Immediate
positional neighbors provide the initial loose pipeline once admitted.

Inspect each strategy's `region_state.csv`: `progress_tokens`, `blocked`, and
`urgent` expose every control decision. The result row also contains the full
`control_timeline` and final persistent edge set.

### Commitment-confidence audit

Diagnostic regional runs also write `commitments.jsonl`. Each committed token
records its response position and region, chosen token, raw untempered top-1
and top-2 probabilities, chosen-token probability, entropy, sampling
distribution statistics, and confidence rank within its region and across the
currently masked canvas. Diagnostic computation is synchronized, timed
separately, and excluded from reported decoding wall time.

Rerun only the vanilla-correct / controlled-position-wrong GSM8K examples:

```bash
python -m dream_region_pilot.run_gsm8k \
  --config configs/gsm8k_cot_official_50.yaml \
  --output-dir outputs/gsm8k_controlled_position_confidence_audit \
  --example-indices 3 4 6 9 10 17 28 30 41 45 46 47 \
  --strategies controlled_position \
  --diagnostic-examples 12 \
  --diagnostic-snapshot-interval 4
```

For example 41, inspect its least-confident visible commitments with:

```bash
effective=$(jq -r 'select(.example_index == 41) | .effective_generated_tokens' \
  outputs/gsm8k_controlled_position_confidence_audit/results.jsonl)
jq -s --argjson effective "$effective" '
  [.[] | select(.response_position < $effective)]
  | sort_by(.raw_top1_probability)
  | .[:20]
' outputs/gsm8k_controlled_position_confidence_audit/diagnostics/example_041/controlled_position/commitments.jsonl
```

The aggressive `wavefront_probe` is a diagnostic ablation, not the speed or
accuracy baseline for the 50-example run. Its two-example result is useful for
showing that a 15% gate can fail, but it should not be presented as FlowBlock.

## Current 50-example comparison

The headline pilot now compares:

```text
vanilla                 official Dream diffusion_generate
flowblock_proxy         W=2, theta_spawn=0.60, token readiness p>=0.50
loose_wavefront         W=8, theta_spawn=0.15, readiness only
mean_field_repro        exact-JSD Algorithm 1 reproduction, block size 32
controlled_position     W=8, theta_spawn=0.15, positional bounded skew only
controlled_dapd         position control plus persistent DAPD edges
controlled_jsd          position control plus persistent Mean-Field/JSD edges
controlled_combo        position control plus persistent DAPD intersection JSD
```

`flowblock_proxy` is deliberately named a proxy. It transfers FlowBlock's
admission settings into this fixed-region Dream harness, but retains Dream's
full attention visibility, entropy commit rule, and full-canvas forwards. It
does not implement FlowBlock's T2T editing, block-causal KV cache, or threshold
commit policy. It is the fair readiness-gate comparison available inside this
pilot, not a reproduction of official FlowBlock throughput.

`controlled_position` is essential attribution control: it uses exactly the
same W=8 admission and actual-token-progress bounded-skew scheduler as the
dependency modes but never extracts or activates dynamic graph edges. A
dependency strategy must improve on this row—not merely on the aggressive
wavefront—to support the graph hypothesis.

The official FlowBlock source is pinned under `external/FlowBlock` by setup.
It cannot be installed into the Dream virtual environment: its released engine
targets LLaDA-2.1-mini, SGLang/vLLM, and documents roughly 80 GB VRAM. On a
suitable GPU, create its separate documented environment and invoke its own
comparison entry point through:

```bash
bash scripts/run_official_flowblock.sh /path/to/LLaDA2.1-mini \
  --benchmark gsm8k --num-samples 50 --batch-size 1
```

That result is an official same-LLaDA FlowBlock-vs-LLaDA comparison. It is not
merged numerically with Dream TPS because model, canvas length, attention/KV
behavior, and hardware requirements differ.

## Vast setup

An RTX 4090 (24 GB) is the intended pilot GPU. Use batch size 1 (the harness
does this), at least 32 GB of system RAM, and about 40 GB of free disk for the
checkpoint, Python environment, DAPD checkout, and outputs. The default is
BF16. No quantization or multi-GPU setup is required.

On a fresh Vast instance:

```bash
cd /workspace
git clone https://github.com/vbsh123/dream-region-pilot.git
cd /workspace/dream-region-pilot
bash scripts/setup_vast.sh
```

The setup creates `.venv`, installs the package, checks out public DAPD at the
pinned revision under `external/DAPD`, and checks out official FlowBlock
source-only under `external/FlowBlock`. Hugging Face downloads the Dream
checkpoint and selected benchmark only when an experiment command is run.

The setup expects `git`, Python 3.10--3.12 with `venv`, a working NVIDIA
driver, and outbound access to GitHub, PyPI, and Hugging Face. If the first
dependency-enabled forward genuinely runs out of GPU memory, stop and record
the error rather than silently changing canvas length, precision, or attention
extraction. Those would be experimental changes, not setup fixes.

## Original fixed-lag graph probe

Before the full evaluation, run:

```bash
cd /workspace/dream-region-pilot
bash scripts/run_graph_probe.sh outputs/gsm8k_graph_probe_2
```

This runs `async_lag0`, `async_lag1`, `async_lag2`, and `async_lag4`,
recomputes dependencies every forward, and saves every graph snapshot. Inspect
these first:

```text
outputs/gsm8k_graph_probe_2/diagnostics/example_000/async_lag0/graph_metrics.csv
outputs/gsm8k_graph_probe_2/diagnostics/example_000/async_lag0/graph_metrics_over_time.png
outputs/gsm8k_graph_probe_2/diagnostics/example_000/async_lag1/graph_timeline.json
```

The timeline records average region degree, percentage of possible edges,
connected components, every region's parent count, root count, maximum parent
count, and whether the graph is complete. A dense graph is retained as a
negative result; the code does not sparsify it automatically. Inspect this
probe before choosing the threshold or launching the full sweep.

## Exact 50-example command

```bash
cd /workspace/dream-region-pilot
bash scripts/run_gsm8k_50.sh outputs/gsm8k_expanded_50_r32
```

The strategy set is:

```text
vanilla flowblock_proxy loose_wavefront mean_field_repro controlled_position controlled_dapd controlled_jsd controlled_combo
```

Resume an interrupted run with:

```bash
bash scripts/run_gsm8k_50.sh outputs/gsm8k_expanded_50_r32 --resume
```

To test another supported fixed region size, copy the YAML and change
`region_size` and `local_steps` together to 16 or 64. Keep outputs in a new
directory.

The main ablations are also direct command-line flags. For example:

```bash
bash scripts/run_gsm8k_50.sh outputs/gsm8k_50_r64_power \
  --region-size 64 \
  --dependency-aggregator power_mean \
  --gamma 2 \
  --dependency-threshold 0.00005
```

Available dependency flags are `--dependency-aggregator` (all three requested
aggregators), `--dependency-matrix`, `--dependency-threshold`,
`--dependency-recompute-interval`, `--topk-percent`, and `--gamma`.

## Outputs

- `results.jsonl`: accuracy inputs plus per-example NFE, commitment, token,
  timing, canvas/effective TPS, dependency-cost, clock, and graph-density
  measurements.
- `summary.csv` and `summary.json`: accuracy, NFE, aggregate canvas/effective
  TPS, mean pre-EOS output length, wall time, and vanilla-relative
  NFE/wall/TPS speedups for every mode.
- `metadata.json`: resolved model commit when exposed by Transformers, pinned
  DAPD revision, complete config, and interpretation notes.
- `diagnostics/example_*/async_lag*/`: full raw/normalized token dependency
  matrices (`.npz`), all three region matrices (`.csv` and `.npz`), graph
  timelines with edge scores, per-iteration clocks/mask ratios/advanced
  regions, and matplotlib plots.

No result table is checked into the repository because the requested local
workflow explicitly forbids running the experiment here.

NFE remains useful because all strategies still run the same 256-token Dream
canvas: fewer forwards means less core model work. It is not treated as the
only speed result. End-to-end decoding wall time includes required DAPD/JSD
computation, and the decisive throughput columns are `canvas_tokens_per_second`
and `effective_tokens_per_second`. Diagnostic serialization/plotting is timed
separately and excluded. Strategy order rotates by example to reduce systematic
warm-cache and thermal bias.

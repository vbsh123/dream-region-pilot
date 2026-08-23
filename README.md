# Dream fixed-region asynchronous decoding pilot

Minimal, training-free research code for testing whether fixed generated-token
regions can follow separate Dream commitment clocks under dependency-defined
backpressure.

Status: implemented locally, deliberately **not run**. No model, dataset,
environment setup, test, build, or experiment was executed on this PC.

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

The setup creates `.venv`, installs the package, and checks out public DAPD at
the pinned revision under `external/DAPD`. Hugging Face downloads the Dream
checkpoint and GSM8K only when the experiment command is run.

The setup expects `git`, Python 3.10--3.12 with `venv`, a working NVIDIA
driver, and outbound access to GitHub, PyPI, and Hugging Face. If the first
dependency-enabled forward genuinely runs out of GPU memory, stop and record
the error rather than silently changing canvas length, precision, or attention
extraction. Those would be experimental changes, not setup fixes.

## Two-example graph probe

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
bash scripts/run_gsm8k_50.sh outputs/gsm8k_50_r32
```

The default strategy set is:

```text
vanilla fixed_sequential always_on async_lag0 async_lag1 async_lag2 async_lag4
```

Resume an interrupted run with:

```bash
bash scripts/run_gsm8k_50.sh outputs/gsm8k_50_r32 --resume
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
  timing, dependency-cost, clock, and graph-density measurements.
- `summary.csv` and `summary.json`: accuracy-vs-NFE comparison for every mode.
- `metadata.json`: resolved model commit when exposed by Transformers, pinned
  DAPD revision, complete config, and interpretation notes.
- `diagnostics/example_*/async_lag*/`: full raw/normalized token dependency
  matrices (`.npz`), all three region matrices (`.csv` and `.npz`), graph
  timelines with edge scores, per-iteration clocks/mask ratios/advanced
  regions, and matplotlib plots.

No result table is checked into the repository because the requested local
workflow explicitly forbids running the experiment here.

# qwen3.5-quant-bench

Benchmarking quantization methods for [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B)
on the [`MME-RealWorld`](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld)
dataset, served with [vLLM](https://github.com/vllm-project/vllm), run entirely on
Google Colab.

The goal isn't just "does quantization hurt accuracy" — it's whether a given
quantization method is actually worth using, judged on latency, memory, and quality
together.

## Methodology

**Model & dataset.** `Qwen3.5-2B` is the base model, evaluated against quantized
checkpoints of the same model.
Evaluation uses `MME-RealWorld`, a multiple-choice, image-grounded VQA benchmark
split into `Reasoning` and `Perception` tasks. A stratified 10% sample (same
proportion pulled from each task) is used for fast iteration; full-dataset runs are
a separate, more expensive pass and are not mixed with sample-run numbers.

**Inference.** All models are served through vLLM (`LLM.chat`, batched multimodal
messages, `temperature=0.0` for reproducible outputs). Each quantization method is
loaded, benchmarked, and fully torn down (`destroy_model_parallel`, CUDA cache
cleared) before the next one loads, so measurements aren't contaminated by a
previous model's residency.

**Metrics.** For every quantization method under test:

| Metric | What it captures |
|---|---|
| Latency (avg) | Mean end-to-end request latency |
| Latency (time-weighted) | Latency weighted by output length, so long generations don't get diluted by short ones in the average |
| Latency (p50/p90/p99) | Median and tail end-to-end latency — averages hide the slow requests that matter most for user-facing behavior |
| TTFT (avg + p50/p90/p99) | Time to first token — responsiveness / prefill cost, distribution not just mean |
| TPOT (avg + p50/p90/p99) | Time per output token — steady-state decode throughput, distribution not just mean |
| Memory (load) | Peak/allocated VRAM once the model is resident |
| Memory (unload delta) | VRAM not reclaimed after teardown — catches quantization backends that leak or fragment memory |
| Accuracy | MME-RealWorld score via the benchmark's own `eval_your_results.py`, so numbers stay comparable to published results |

Additional metrics get added here as they turn out to matter (e.g. throughput under
concurrent load, quantization/load time itself) — this table is the current set, not
a fixed spec.

**Accuracy scoring** always goes through the upstream `MME-RealWorld` evaluation
script rather than a custom scorer, to avoid silently drifting from how the
benchmark is meant to be graded.

## Infrastructure

Everything runs on Google Colab, with all persistent state — model weights, the HF
cache, extracted dataset images, sampled question sets, and per-run result JSONs —
stored on Google Drive instead of Colab's ephemeral local disk. This is the main
speed lever: weights, dataset, and installed packages only need to be fetched once
and are reused across sessions, so a fresh Colab runtime can pick up where a
previous one left off without re-downloading or reinstalling anything already
present.

Package installs use [`uv`](https://github.com/astral-sh/uv) instead of pip for
faster environment setup. Model weights and dataset come from Hugging Face Hub.

## Outline

1. **Setup** — mount Drive, set up `PROJECT_DIR`/`HF_HOME`, install dependencies
   with `uv`, authenticate to Hugging Face.
2. **Fetch** — download base + quantized model weights and the dataset via
   `snapshot_download`, with broken-symlink detection/retry for interrupted
   downloads.
3. **Preprocess** — reassemble/extract dataset archives, flatten images into a
   single directory, load and filter the English question set, build a stratified
   sample for fast iteration.
4. **Inference** — run each model variant (fp16 baseline, then each quantization
   method) over the question set in batches, writing results incrementally so a
   killed kernel doesn't lose progress.
5. **Evaluate** — score each result set with the official MME-RealWorld evaluator.
6. **Benchmark** — collect latency/TTFT/TPOT/memory metrics per quantization method
   (in progress — see `CLAUDE.md` for current status).
7. **Record** — commit the notebook (with outputs) and result artifacts after every
   run, so both the numbers and the trail of what was tried/fixed are preserved in
   git history.

## Repo layout

- `qwen3.5-quant-bench.ipynb` — the entire pipeline (setup through benchmarking).
- `README.md` — this file.
- `CLAUDE.md` — working conventions for this repo (env quirks, storage layout,
  benchmarking/commit conventions) for anyone (human or AI) picking this up.

Large artifacts (weights, HF cache, dataset images, result JSONs) intentionally live
on Google Drive, not in this repo.

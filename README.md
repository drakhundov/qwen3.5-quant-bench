# qwen3.5-quant-bench

Benchmarking quantization methods for [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B)
on the [`MME-RealWorld`](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld)
dataset, served with [vLLM](https://github.com/vllm-project/vllm), run entirely on
Google Colab.

The goal isn't just "does quantization hurt accuracy" — it's whether a given
quantization method is actually worth using, judged on latency, memory, and quality
together.

## Methodology

**Model & dataset.** `Qwen3.5-2B` is the base model, evaluated in three variants: full
precision (fp16), AWQ ([`QuantTrio/Qwen3.5-2B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.5-2B-AWQ)),
and GPTQ — GPTQ is currently a placeholder in the notebook (`MODEL_CONFIGS`) since no
public Qwen3.5-2B GPTQ checkpoint exists yet; it's skipped automatically (loudly, not
silently) until one is filled in. Which variants actually run is controlled by the
`BENCHMARK_MODELS` toggle at the top of the notebook.
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
The Colab runtime this notebook targets is a T4 GPU (16GB VRAM) — a real constraint
on several of the choices below, not just Colab's default.

## Optimization methods

Speed work falls into three buckets: not re-fetching things, not re-computing things,
and getting more out of the GPU per second it's busy. In roughly the order they pay
off:

- **Downloads are concurrent and accelerated.** All enabled models' weights download
  in parallel (`ThreadPoolExecutor` around `snapshot_download`) instead of one after
  another, and `HF_HUB_ENABLE_HF_TRANSFER=1` swaps in HF's Rust-based accelerated
  downloader for the actual transfer. Combined with everything landing on Drive (see
  Infrastructure above), a repo/weight/dataset is fetched at most once, ever, per
  Drive.
- **Sampled images are staged to local disk before decoding.** `EXTRACT_DIR` on
  Drive is a single flat directory holding every image in the full dataset, not
  just the sample — per-file lookups against a Drive-mounted directory that large
  are slow, and Drive's per-user rate limiting can add further backoff under
  concurrent access. The ~10% sample's images are bulk-copied to local disk once
  (skipping any already staged, so an interrupted copy resumes), and the actual
  decode step reads from that local copy instead of Drive.
- **Images are decoded once, not once per model.** The sample question set and every
  model variant (fp16/AWQ/GPTQ) share the exact same images and prompts — the only
  thing that changes between runs is which `LLM` processes them. Chat payloads are
  built once up front with a thread pool and reused across all three inference
  passes, instead of being rebuilt from scratch for each one. Trade-off: this holds
  every decoded image in memory for the run, which is fine for the 10% sample but
  worth revisiting for a full-dataset run.
- **`max_model_len` is sized to the actual workload, not the model's ceiling.**
  MME prompts are one image plus a short question, nowhere near vLLM's reported
  context ceiling. A needlessly large `max_model_len` reserves KV-cache block
  space for sequence lengths that never occur, directly capping how many requests
  can run concurrently — so it's set to 8192, the single largest throughput lever
  here, especially on a 16GB T4.
  - `max_num_seqs` is set explicitly (64) and `gpu_memory_utilization` is held
    constant across fp16/AWQ/GPTQ on purpose — letting a smaller quantized model
    grab extra KV-cache headroom would make it look faster for a reason unrelated to
    quantization, which would make the latency/TTFT/TPOT comparison unfair rather
    than actually reflecting the method.
- **One inference call per chunk, not one per item.** Requests within a chunk are
  handed to `llm.chat()` together so vLLM's own continuous-batching scheduler
  interleaves prefill/decode across them, instead of the caller synchronizing on
  small sub-batches and leaving the GPU idle between them. The chunk size is tied to
  `max_num_seqs` so each chunk is actually large enough to saturate the scheduler —
  incremental result checkpointing (so a killed Colab kernel doesn't lose progress)
  still happens once per chunk.
- **CUDA graphs are captured for a range of batch sizes, not just one.** vLLM
  only gets the CUDA-graph speedup for batch sizes it captured a graph for; a
  partial trailing chunk (say 47 of 64 items) needs its own captured size, or it
  falls back to slow eager execution. `cudagraph_capture_sizes` covers a full
  range (`[1, 2, 4, 8, 16, 32, 64]`), not just the endpoints, so odd-sized
  trailing chunks stay on the fast path.
- **Prefix caching is on.** Every request shares the same chat-template preamble;
  `enable_prefix_caching=True` lets vLLM skip recomputing it. Modest, since the
  image and question content still differ per request, but free.
- **`dtype="float16"`, deliberately.** T4 (Turing) has no native bf16 tensor cores,
  so fp16 isn't just a default here — it's the dtype that actually runs fast on this
  hardware.
- **A known ceiling: Marlin kernels need Ampere+.** vLLM auto-upgrades AWQ/GPTQ to
  its fast Marlin kernel on Ampere-or-newer GPUs; T4 (sm_75, Turing) doesn't qualify,
  so quantized runs on this notebook's hardware fall back to a slower (but correct)
  kernel. This isn't something the code controls — worth checking the vLLM startup
  log for which kernel actually got selected, since it matters more for quantized
  throughput than anything above.
- **Metrics come from vLLM's own per-request timestamps**, not wall-clock timing
  wrapped around a batch call — see the Metrics table above and `CLAUDE.md` for why
  that distinction matters for TTFT/TPOT specifically.

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
6. **Benchmark** — collect latency/TTFT/TPOT/memory metrics per quantization method,
   sourced from vLLM's own per-request timestamps, and write both per-model and a
   combined `benchmark_summary.json`.
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

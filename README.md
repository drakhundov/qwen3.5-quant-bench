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
| Model load time | Wall time to construct `vllm.LLM(...)`, kept separate from inference wall time |
| Latency (avg) | Mean end-to-end request latency |
| Latency (time-weighted) | Latency weighted by output length, so long generations don't get diluted by short ones in the average |
| Latency (p50/p90/p99) | Median and tail end-to-end latency — averages hide the slow requests that matter most for user-facing behavior |
| TTFT (avg + p50/p90/p99) | Time to first token — responsiveness / prefill cost, distribution not just mean |
| TPOT (avg + p50/p90/p99) | Time per output token — steady-state decode throughput, distribution not just mean |
| Queue / prefill / decode time | vLLM's own breakdown of where request time goes — separates scheduler backpressure from actual prefill/decode cost |
| Throughput (req/s, tokens/s) | Aggregate throughput over the whole pass, the headline number for comparing methods |
| Preemptions | Count of requests evicted-and-recomputed under KV-cache pressure, and how many requests were affected — explains latency outliers that aren't about raw decode speed |
| Corrupted requests | Count of requests vLLM itself flagged as corrupted — a nonzero count means the accuracy number for that run may not be trustworthy |
| Memory (load) | Peak VRAM used during load + inference |
| Memory (unload delta) | VRAM not reclaimed after teardown — catches quantization backends that leak or fragment memory |
| Accuracy | MME-RealWorld score via the benchmark's own `eval_your_results.py`, so numbers stay comparable to published results |

Additional metrics get added here as they turn out to matter — this table is the
current set, not a fixed spec.

**Where these numbers actually come from, and why it isn't as simple as reading
`RequestOutput.metrics`.** vLLM's offline batch API doesn't hand you per-request
timing for free — its `LLM` class defaults per-request metric collection to *off*,
and even with it explicitly enabled, the object you get back has gone through more
than one redesign across vLLM versions, so the field names aren't the ones you'd
find in older docs or examples. Latency/TTFT/TPOT here are computed from that
object with both of those accounted for explicitly, not assumed. Memory numbers
come from querying the GPU device directly (NVML) rather than the Python process's
own CUDA accounting, since vLLM runs the actual model in a separate subprocess —
`torch.cuda.*` in the calling process can't see memory that process never
allocated. See `CLAUDE.md` for the specifics if either of these need touching
again.

**Where these numbers actually come from, and why it isn't as simple as reading
`RequestOutput.metrics`.** vLLM's offline batch API doesn't hand you per-request
timing for free — its `LLM` class defaults per-request metric collection to *off*,
and even with it explicitly enabled, the object you get back has gone through more
than one redesign across vLLM versions, so the field names aren't the ones you'd
find in older docs or examples. Latency/TTFT/TPOT here are computed from that
object with both of those accounted for explicitly, not assumed. Memory numbers
come from querying the GPU device directly (NVML) rather than the Python process's
own CUDA accounting, since vLLM runs the actual model in a separate subprocess —
`torch.cuda.*` in the calling process can't see memory that process never
allocated. See `CLAUDE.md` for the specifics if either of these need touching
again.

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

- **The full question set is only loaded when there's no cached sample yet.**
  Checking for `sample_questions.json` happens *before* touching the ~23,600-question
  source file, not after — a run with an already-cached sample never parses the full
  file at all.
- **Downloads are concurrent and accelerated.** All enabled models' weights download
  in parallel (`ThreadPoolExecutor` around `snapshot_download`) instead of one after
  another, and `HF_HUB_ENABLE_HF_TRANSFER=1` swaps in HF's Rust-based accelerated
  downloader for the actual transfer. Combined with everything landing on Drive (see
  Infrastructure above), a repo/weight/dataset is fetched at most once, ever, per
  Drive.
- **vLLM's kernel-compilation cache persists across sessions too, not just
  weights/dataset.** Each engine startup normally recompiles the Triton/CUDA
  kernels behind its `torch.compile`/CUDA-graph path from scratch; that cache
  is pointed at local disk (fast) and round-tripped to a single tarball on
  Drive once per model pass, so a fresh Colab runtime restores it instead of
  paying full compilation cost on every startup for every model, every
  session.
- **Weights are staged to local disk before vLLM loads them.** Unlike the
  many-small-image problem below, weight files are few and large (a handful of
  safetensors shards) — bulk sequential reads are Drive's best case, not its worst.
  They're still copied to local disk once per session before `vllm.LLM(...)` reads
  them, as a precaution: safetensors loading commonly memory-maps its files, and
  mmap'd access over a network-backed FUSE mount can turn into scattered small
  reads depending on access order, which would reintroduce the same problem the
  image staging below fixes — just hidden inside one file instead of spread across
  many.
- **Raw images are only fetched for what isn't already cached, and never
  extracted on Drive.** Once the sample and the base64 cache (below) are known, the
  notebook works out which needed images the cache doesn't cover. If there are none,
  no raw image is touched. Otherwise only those are staged from the flattened
  image folder already on Drive (one directory listing, then 8 retrying threads);
  and only if that folder is missing or incomplete does it copy the dataset archives
  to local disk and extract them there — never onto Drive, since thousands of small
  writes over Drive's FUSE mount are as slow and failure-prone as thousands of
  small reads.
- **The base64-encoded images themselves are cached and persisted to Drive.**
  Reading and base64-encoding an image happens once per model pass by design (see
  the windowed chat-building point below) and again on every fresh Colab session —
  `IMAGE_CACHE` (`{basename: data_url}`) means that work happens at most once,
  ever, per image. It's filled lazily inside `build_chat` (a cache miss is
  encoded and stored right there, no separate build pass), and saved back to
  `PROJECT_DIR/base64_image_cache.json` on Drive after every model pass — not just
  once at the end — so a crash partway through a run doesn't throw away
  cache-filling work already done.
- **Images are read as base64, not decoded locally.** Each cached image is read as
  raw bytes and base64-encoded into vLLM's `image_url` content type, rather than
  opened with PIL and passed as `image_pil`. This skips a local decode step
  entirely (vLLM decodes on its own side) and keeps each chat payload close to the
  compressed file size in memory, instead of a full decoded RGB array.
- **Chat payloads are built in small windows, immediately before they're used, not
  for the whole sample up front.** `run_inference_pass` builds chats a few chunks
  at a time (`GROUP_BATCHES`, currently 3 × 64 = 192 items) right before running
  inference on that window, then moves on — instead of building all ~2,300 chats
  before any inference starts. This bounds memory to one window's worth of data
  regardless of sample size. Trade-off: since chats aren't cached across models
  anymore, each of the three passes re-reads and re-encodes the same images — but
  since staging already made those reads local, that repeat cost is small compared
  to what it would have cost against Drive.
- **The next window is prefetched while the current one runs on the GPU.** Building
  a window of chats (local reads + base64 encode) is fast compared to running that
  window through the model, so the *next* window is built on a background thread
  while `llm.chat()` is still working on the current one, instead of waiting for
  inference to finish before starting the next build. This hides most of the
  build time behind GPU time rather than paying for it serially between windows.
- **`max_model_len` is sized to the actual workload, not the model's ceiling —
  but "the actual workload" turned out bigger than expected.** MME-RealWorld
  images vary widely in resolution, and vision tokenization scales with it; at
  least one image in the sample needs 16000+ tokens once encoded, well past a
  "one image, short question" assumption. `max_model_len` is set to 32768 —
  real headroom above that observed floor, not tuned to an exact minimum — while
  staying far below vLLM's reported native ceiling for this model. Leaving it
  unset entirely would size the KV cache for that full native context across
  every concurrent sequence, which can fail engine startup outright rather than
  just being wasteful.
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
- **Prefix caching is off.** It would have been a modest win (every request
  shares the same chat-template preamble, though the image and question content
  still differ per request) — but this model uses a Mamba-hybrid architecture,
  and vLLM logs that enabling prefix caching switches it into a special Mamba
  cache mode that fails engine startup outright for this model. Not worth a
  broken engine for a modest win; revisit once vLLM's hybrid-model support for
  this matures.
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
6. **Benchmark** — collect load-time/latency/TTFT/TPOT/queue-prefill-decode/
   throughput/preemption/memory metrics per quantization method, sourced from
   vLLM's own per-request timestamps, and write both per-model
   (`benchmarks/<key>_metrics.json`) and a combined
   (`benchmarks/benchmark_summary.json`) file.
7. **Record** — commit the notebook (with outputs) and the `benchmarks/`
   artifacts after every run, so both the numbers and the trail of what was
   tried/fixed are preserved in git history.

If any cell raises an uncaught exception and `BREAK_ON_ERR` (top of the
notebook) is set, the Colab runtime disconnects itself rather than sitting
idle/billing after an unattended run dies partway through — set it to `0`
while iterating interactively. Between sessions, vLLM's `torch.compile`
kernel-compilation cache round-trips to a single tarball on Drive so a fresh
runtime doesn't recompile from scratch on every engine startup.

## Repo layout

- `qwen3.5-quant-bench.ipynb` — the entire pipeline (setup through benchmarking).
- `README.md` — this file.
- `CLAUDE.md` — working conventions for this repo (env quirks, storage layout,
  benchmarking/commit conventions) for anyone (human or AI) picking this up.
- `benchmarks/` — per-model metrics JSON, a combined summary JSON, and the
  MME-RealWorld eval script's output, one set per benchmarked variant. Small
  and git-friendly, committed after every run (see Commit discipline in
  `CLAUDE.md`). The notebook writes them to `PROJECT_DIR/benchmarks/` on Drive so
  they survive the runtime disconnecting; copy that folder here to commit it.

Large, per-run artifacts — model weights, the HF cache, dataset images, and the
raw per-item model output JSON that `benchmarks/` is summarized from —
intentionally live on Google Drive, not in this repo.

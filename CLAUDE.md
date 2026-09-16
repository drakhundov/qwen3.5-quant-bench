# qwen3.5-quant-bench

Benchmarking quantization methods for `Qwen/Qwen3.5-2B` on the `MME-RealWorld` dataset,
using vLLM for inference, entirely inside Google Colab notebooks.

## Repo shape

- `qwen3.5-quant-bench.ipynb` — the whole pipeline lives in one notebook: env setup →
  dataset/weight download → preprocessing → inference → evaluation → benchmarking.
  There is no separate `.py` source tree; treat the notebook as the codebase.
- `README.md` — methodology and outline, aimed at a human reader (or a future you)
  landing on the repo cold.
- Everything heavy (model weights, HF cache, dataset archives/images, per-run JSON
  results) lives in Google Drive under `PROJECT_DIR`, **not** in git. Only the notebook,
  README, and small config/markdown files belong in the repo.

## Environment (Google Colab)

- Storage: Drive is mounted at `/content/drive`, and `PROJECT_DIR =
  /content/drive/MyDrive/qwen3.5-quant-bench` is the root for everything persistent —
  HF cache (`HF_HOME = $PROJECT_DIR/hf_cache`), extracted dataset images
  (`$PROJECT_DIR/mme_dataset_images`), sampled question sets, and `test_results/`.
  The point of this is to avoid re-downloading weights/dataset and reinstalling
  packages every session — if you add new persistent state, put it under
  `PROJECT_DIR`, and check for existing output before recomputing/redownloading it
  (see the `sample_questions.json` and broken-symlink-retry cells for the pattern).
- Package manager: `uv` (not pip/conda) for speed — `uv venv --python 3.12 --seed`,
  then `uv pip install`. `torchaudio` is explicitly uninstalled because vLLM's CUDA
  version check conflicts with it and it isn't needed.
- Secrets: `HF_TOKEN` (and `HUGGINGFACE_API_KEY`, same value) come from Colab's
  `userdata.get(...)`, not hardcoded or `.env` files. This only works when the
  cell calling it is run from the actual Colab browser UI — it times out (not a
  clean error) when driven from an external client like VSCode's Jupyter extension
  connected to the same Colab kernel, since the vault needs a round-trip to the
  browser tab itself. Deliberately not worked around with a token file on Drive: a
  file there is readable by anything with Drive access to that folder (any other
  notebook, any app granted Drive access, Drive Desktop sync, anyone the folder
  gets shared with) — not scoped to this notebook the way the vault is. If driving
  this notebook from VSCode, the env-setup cell needs to be run from the actual
  Colab UI once per fresh runtime; the token then stays in that kernel's memory
  for the rest of the session.
- HF auth + downloads go through `huggingface_hub` (`login`, `snapshot_download`,
  `hf_hub_download`), cached under `HF_HOME`.

## Dataset handling gotchas

- `yifanzhang114/MME-RealWorld` on HF ships as `.tar.gz` / `.tar.gz.part_*` shards;
  the notebook reassembles parts only when a plain archive isn't already present, then
  extracts everything into a single flat `EXTRACT_DIR` (directory structure inside the
  archive is irrelevant — every question references an image by basename).
- HF snapshot downloads can leave broken symlinks (interrupted downloads); there's a
  `check_all_targets` / re-download loop for this — always leave that check in place
  before extraction rather than assuming a snapshot is complete.
- Google Drive's FUSE mount can lag behind newly extracted/written files, so image
  loads use `safe_open_drive_img` with retries instead of a bare `PIL.Image.open`.
  Keep using it (or equivalent retry wrapping) for anything reading files that were
  just written to Drive in the same run.
- The English-only questions JSON is picked by filename matching `MME` and excluding
  `CN` — if you add other MME-RealWorld variants, keep this filter honest.

## Inference conventions

- Which variants run is controlled by `BENCHMARK_MODELS` (top of the notebook) and
  `MODEL_CONFIGS` (repo id + vLLM `quantization` kwarg per key, in the download
  cell). Add a new quantization method by adding a `MODEL_CONFIGS` entry and a
  matching `BENCHMARK_MODELS` flag — don't hardcode a new per-model cell, the
  inference/eval loops iterate `MODEL_CONFIGS` generically.
- `qwen3.5-2b-gptq`'s `repo_id` is intentionally empty — there's no public
  Qwen3.5-2B GPTQ checkpoint yet (checked HF search 2026-09). It's left
  enabled-but-empty on purpose so the download/inference loops print a loud skip
  instead of a model silently missing from the benchmark. Fill it in once a
  checkpoint exists, or point it at a self-quantized (GPTQModel/AutoGPTQ) path.
- Models are loaded via `vllm.LLM(...)`, `dtype="float16"` (this project's Colab
  runtime is a T4 — Turing has no native bf16 tensor cores, so this isn't just a
  default), `trust_remote_code=True`, `max_model_len=8192` (MME prompts are one
  image + a short question; don't casually raise this — it directly trades away
  KV-cache concurrency), `max_num_seqs=64`, `enable_prefix_caching=True`, and a
  `compilation_config` with `cudagraph_capture_sizes` covering a full range (not
  just the endpoints — a partial trailing batch with no matching captured graph
  silently falls back to slow eager execution). Quantized variants pass
  `quantization=<method>` (`"awq"` or `"gptq"`); `gpu_memory_utilization` (0.85) is
  held constant across all variants deliberately, so a smaller quantized model
  doesn't get an unfair throughput boost from extra KV-cache headroom that has
  nothing to do with quantization.
- On this notebook's T4 hardware specifically, vLLM's fast Marlin kernel for
  AWQ/GPTQ is unavailable (Marlin needs Ampere+/sm_80, T4 is sm_75) — expect a
  slower fallback kernel and check the vLLM startup log for which one actually got
  selected before drawing conclusions from quantized throughput numbers.
- Sampled images are staged from Drive to local disk (`/content/mme_sample_images`)
  once, before chat payloads are built. `EXTRACT_DIR` on Drive is a single flat
  directory holding every image in the *full* dataset, not just the sample —
  looking up files in a Drive-mounted directory that large is slow per-file, and
  Drive's per-user rate limiting can add further backoff under concurrent access.
  Copying just the ~10% sample locally first turns per-image reads into local disk
  reads for the actual decode step, which are effectively free by comparison. The
  staging cell skips files that already exist locally, so it's safe to rerun if
  interrupted. `build_chat`/`safe_open_drive_img` resolve images relative to the
  global `EXTRACT_DIR`, so the chat-prebuild cell temporarily repoints `EXTRACT_DIR`
  at the local staged copy and restores it afterward — keep that restore if you
  touch this cell, since other cells (and any future full-dataset run) expect
  `EXTRACT_DIR` to mean the Drive copy.
- Chat payloads (`build_chat`) are built **once**, before the per-model loop, and
  reused across every model variant — the images/questions are identical across
  fp16/AWQ/GPTQ, so rebuilding them per model is pure repeated decode work. If you
  change what varies per model run, keep this precompute-once-reuse-across-models
  structure; don't fold chat-building back inside `run_inference_pass`.
- Inference runs in chunks sized to `MAX_NUM_SEQS` (currently 64, shared with
  `max_num_seqs` above so each chunk actually saturates vLLM's scheduler) using a
  single `llm.chat(...)` call per chunk with multimodal messages (`image_pil` +
  text) — one call per chunk lets vLLM's continuous batching interleave
  prefill/decode across the whole chunk, rather than many small serialized calls
  that would leave the GPU idle between them. Results are written to the results
  JSON incrementally after every chunk — keep this pattern for any new run loop
  so a killed Colab kernel doesn't lose completed work.
- Between loading different model variants in the same session, always tear down
  the previous one first: `destroy_model_parallel()`, `del llm`, `gc.collect()`,
  `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`. This matters even more
  now that memory (`memory_gb.peak_allocated`/`unload_residual`) is measured per
  run — any measurement taken without this teardown is contaminated by the prior
  model's residency.
- `sample_questions.json` (10% stratified by `Task`, cached under `PROJECT_DIR`) is
  the fast-iteration dataset. Full-dataset runs are a separate, more expensive pass —
  don't conflate the two when comparing numbers across runs. Note the pre-built
  chat cache above holds every decoded image in memory, which is fine for the
  sample but should be reconsidered (e.g. build chats per-chunk) for a full-dataset
  run.

## Evaluation & benchmarking

- Accuracy: the cloned `MME-RealWorld` repo's own
  `evaluation/eval_your_results.py` is run as a subprocess against each results JSON.
  Don't reimplement MME scoring — shell out to the upstream script so numbers stay
  comparable to published results.
- Benchmarking metrics captured per quantization method (`run_inference_pass`
  returns them; the driving loop writes `<key>_metrics.json` per model plus a
  combined `benchmark_summary.json` under `test_results/`):
  - Latency — average, time-weighted (i.e. weighted by output length, since longer
    generations shouldn't count the same as short ones in an average), and p50/p90/p99
    (tail latency, not just the mean).
  - TTFT (time to first token) — average and p50/p90/p99.
  - TPOT (time per output token) — average and p50/p90/p99.
  - Memory: peak/allocated VRAM during load, and delta after unload (to catch
    quantization methods that don't fully release memory).
  - Anything else that materially affects the latency/memory/quality tradeoff is
    fair game to add — note it in the README's methodology section when you do.
- When adding new timing instrumentation, prefer vLLM's own request-level metrics
  (per-request timestamps in `RequestOutput`/engine stats) over wall-clock timing
  wrapped around the whole batch — batch-level timing hides TTFT/TPOT and
  under-counts latency variance.
- Percentiles (p50/p90/p99) must be computed over the raw per-request metric
  arrays, not over already-averaged per-batch numbers — averaging first and then
  taking percentiles of those averages silently collapses the tail you're trying to
  measure. Keep the raw per-request latency/TTFT/TPOT lists around (or dump them to
  the results file) so percentiles can be recomputed later without rerunning
  inference.

## Commit discipline

- Every run gets committed — including the notebook itself (with fresh outputs) and
  any new result files under version control. This is deliberate: it's the audit
  trail for what was tried, what broke, and what the numbers were, not just a backup.
  When committing after a run, write commit messages that say what changed and why
  (a fix, a new quant method, a metric added), not just "update notebook".
- Don't clean up or "simplify" earlier notebook cells as drive-by edits — if
  something looks wrong, flag it and fix it explicitly (with a comment saying what
  was broken and why), don't silently rewrite history-bearing cells while doing
  something else.

## General conventions

- Keep the notebook linear and runnable top-to-bottom on a fresh Colab runtime with
  Drive already populated from a prior run (idempotent downloads/extraction is the
  existing pattern — preserve it for new steps).
- No test suite / CI here — correctness is judged by running the notebook on Colab
  against real GPU + Drive + HF. Don't invent local mocks for vLLM/Drive/HF; if you
  can't run it on Colab yourself, say so rather than claiming it works.
- This is a solo research/benchmarking project, not a library — prefer the shortest
  path that produces a correct, comparable number over building reusable
  abstractions the notebook doesn't need yet.

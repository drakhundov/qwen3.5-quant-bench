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
- Package manager: `uv` (not pip/conda) for speed — `uv pip install --system`.
  The `--system` flag is load-bearing, not optional: the setup cell runs as a
  `%%bash` cell, which executes in its own throwaway subprocess, so a
  `uv venv` + `source .venv/bin/activate` there (tried first) never
  propagates to the notebook's actual Python kernel process — anything
  installed into that isolated venv would be invisible to every `import
  vllm`/`import pynvml`/etc. in the cells below. Installing straight into the
  kernel's own system Python with `--system` is what makes those imports
  work at all, and it's naturally idempotent on a rerun, unlike `uv venv`
  (which errors if `.venv` already exists). `torchaudio` is explicitly
  uninstalled because vLLM's CUDA version check conflicts with it and it
  isn't needed. The cell starts with `set -e`: without it, a failed command
  partway through (e.g. a flaky `vllm` install) doesn't stop the script — bash
  keeps going, and the cell's exit status ends up being whatever the *last*
  command returned, silently masking the real failure until `import vllm`
  breaks confusingly cells later. Keep `set -e` if you edit this cell, both
  for that reason and because `BREAK_ON_ERR` (below) can only see failures
  that actually raise.
- Secrets: `HF_TOKEN` (and `HUGGINGFACE_API_KEY`, same value) are entered via a
  hidden `getpass` prompt each session, not hardcoded, not in a `.env` file, and
  not persisted to Drive or local disk — the token exists only in that kernel's
  memory for the session. This works the same whether the notebook is driven from
  the Colab browser UI or an external client (e.g. VSCode's Jupyter extension),
  unlike Colab's `userdata.get()` secrets vault, which needs a round-trip to the
  actual browser tab and times out otherwise. The cost — re-entering the token
  once per fresh runtime — is deliberate: a token file (on Drive or local disk)
  would be more convenient but stays readable by anything with access to that
  storage for as long as the file exists, not just this notebook for one session.
- `snapshot_download`/`hf_hub_download` pick up `HF_TOKEN` from the environment
  automatically (no explicit `huggingface_hub.login()` call needed) — downloads
  are cached under `HF_HOME`.
- `BREAK_ON_ERR` (top of the notebook, next to `BENCHMARK_MODELS`) controls a
  global `get_ipython().set_custom_exc((Exception,), ...)` handler, registered
  in the cell right after that config cell so it covers every cell below —
  Drive mount, pip install, downloads, inference, eval, all of it. When set to
  `1`, any uncaught exception disconnects the runtime (`runtime.unassign()`)
  instead of leaving a GPU-attached Colab runtime idle/billing after an
  unattended run dies partway through. It's registered for `Exception`, not
  `BaseException`, so a manual interrupt (`KeyboardInterrupt`) never triggers
  it. `StopCellExecution` (defined in that same cell) is the escape hatch for
  a deliberate, expected early-stop that *shouldn't* count as a failure — the
  handler special-cases it. Because this depends on `set -e` actually being
  present in the `%%bash` setup cell (see below) to catch install failures,
  don't remove that flag either.

## Dataset handling gotchas

- `yifanzhang114/MME-RealWorld` on HF ships as `.tar.gz` / `.tar.gz.part_*` shards,
  downloaded (via `snapshot_download`) into the Drive-backed `HF_HOME` cache under
  `SNAPSHOT_ROOT` — that download step is a one-time, few-large-files transfer, and
  HF's own cache already makes it a no-op on later sessions, so there's no separate
  "already downloaded" check needed for it.
- Raw images are only fetched for what `IMAGE_CACHE` doesn't already cover, cheapest
  path first (the "make sure every image the sample still needs is on local disk"
  cell, which runs after `questions` and `IMAGE_CACHE` are known): (1) if the cache
  covers every image the sample needs, no raw images are fetched at all; (2) else the
  missing ones are staged, with `_stage_file` retries and 8 threads, from the flattened
  copy already on Drive (`PROJECT_DIR/mme_dataset_images`) into local `EXTRACT_DIR`
  (`/content/mme_dataset_images`), using one `os.listdir` of the Drive folder rather
  than a per-file existence check; (3) only if that Drive folder is missing or lacks a
  needed image does it fall back to `_extract_archives_locally()`, which copies the
  (few, large) archives from `SNAPSHOT_ROOT` to `/content/mme_archives`, extracts them
  there, and flattens locally. Extraction is always local: `tar` writing ~23,600 small
  files onto Drive's FUSE mount is the write-side version of the many-small-files
  problem below, and an earlier version of the extraction cell had a broken "already
  extracted" check that silently redid that onto Drive every session. Nothing
  downstream touches Drive for images.
- The fallback's "already extracted" check is a per-archive `<archive>.extracted`
  marker file, not `os.path.exists(<archive>)` — a plain (non-parted) archive already
  exists the moment it's copied, so its own existence says nothing about extraction.
  Since it runs on local disk (wiped every fresh runtime) the marker only guards a
  same-session rerun. Keep the marker check if you touch that function.
- HF snapshot downloads can leave broken symlinks (interrupted downloads); there's a
  `check_all_targets` / re-download loop for this — always leave that check in place
  before extraction rather than assuming a snapshot is complete.
- Google Drive's FUSE mount can lag behind newly written files, and can also fail a
  read outright with a plain `OSError` under load, not just be slow — don't assume a
  bare read from a Drive-mounted path is reliable; wrap it with retries (see
  `_stage_file` in the weight-download cell) for anything reading files that were
  just written to Drive in the same run, or reading many files from Drive at all.
- `_stage_file(src, dst, retries=3, delay=1.0)`, defined once in the
  weight-download cell, is the single shared staging helper — every Drive-to-local
  copy in this notebook (weights, dataset archives) calls it directly with full
  paths, rather than each having its own near-identical copy. It copies to a
  `.part`-suffixed temp name and `os.replace()`s it into place atomically, rather
  than writing the destination filename directly. This matters because a copy
  interrupted partway (kernel restart, disconnect — both have happened in this
  project) would otherwise leave a truncated file under the final filename, which
  the "skip if already staged" `os.path.exists(dst)` check would then treat as
  complete forever. If a future staging need doesn't fit `_stage_file`'s signature,
  extend it rather than forking a new near-copy — that's exactly the duplication
  this consolidation removed.
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
- Each model's weights are staged from the Drive-backed HF cache to local disk
  (`/content/model_weights/<key>`) right after download, and `weights_path` (what
  `MODEL_CONFIGS` and `run_inference_pass` actually use) points at that local
  copy, not the Drive one. Weight files are few and large, unlike the many-small
  image files elsewhere in this notebook, so this isn't working around the same
  problem — it's a precaution against safetensors' common use of `mmap` for
  loading, which can turn into scattered small reads over a network-backed FUSE
  mount depending on access order. The staging loop skips files already present
  locally, so it's safe to rerun. vLLM itself never touches Drive for weights —
  it only ever reads `weights_path`, the local copy — so this is already "store
  weights in Colab storage" for the part that actually matters (the access
  pattern that would be slow/risky over FUSE). The Drive-backed `HF_HOME` cache
  underneath it stays, deliberately: it's the only reason a fresh Colab runtime
  doesn't re-download several GB of weights per model from HF every session.
  RAM-backed storage (e.g. a `/dev/shm` path) wouldn't add anything here either
  — each model's weights are read once per pass (load, use for the whole pass,
  tear down), never re-read within a session, so there's no repeated-access
  pattern for RAM to speed up that local disk (plus the OS's own page cache)
  doesn't already cover.
- Models are loaded via `vllm.LLM(...)`, `dtype="float16"` (this project's Colab
  runtime is a T4 — Turing has no native bf16 tensor cores, so this isn't just a
  default), `trust_remote_code=True`, `max_model_len=32768`, `max_num_seqs=64`,
  and a `compilation_config` with `cudagraph_capture_sizes` covering a full
  range (not just the endpoints — a partial trailing batch with no matching
  captured graph silently falls back to slow eager execution). Quantized
  variants pass `quantization=<method>` (`"awq"` or `"gptq"`);
  `gpu_memory_utilization` (0.85) is held constant across all variants
  deliberately, so a smaller quantized model doesn't get an unfair throughput
  boost from extra KV-cache headroom that has nothing to do with quantization.
- `max_model_len` is a real, load-bearing value, not a knob to casually change in
  either direction. MME-RealWorld images vary widely in resolution, and vision
  tokenization scales with it — at least one image in the sample needs 16000+
  tokens once encoded, so `max_model_len` needs real headroom above that, not
  just above a "one image + a short question" assumption (that assumption was
  wrong; a plain short-prompt guess of 8192 hit a hard failure on the larger
  images). At the same time, leaving `max_model_len` unset entirely sizes the KV
  cache for this model's full native 262144-token context across every one of
  `max_num_seqs` concurrent sequences, which can fail engine startup outright
  with an opaque "Engine core initialization failed" rather than a clear OOM.
  32768 is chosen with headroom above the observed 16000+ floor, not tuned to an
  exact minimum — if an even larger outlier image surfaces, raise it further
  rather than assuming the current value is exact.
- `enable_prefix_caching` is `False`. This model uses a Mamba-hybrid architecture
  (attention + state-space layers), and vLLM logs that it switches into a special
  Mamba cache "align" mode specifically because prefix caching is on — engine
  startup fails immediately after that log line. The win from prefix caching here
  was always modest (only the shared chat-template preamble benefits; the actual
  image+question content differs per request), not worth trading for a working
  engine. Revisit once vLLM's hybrid-model prefix caching support matures — don't
  flip it back on without confirming that first.
- On this notebook's T4 hardware specifically, vLLM's fast Marlin kernel for
  AWQ/GPTQ is unavailable (Marlin needs Ampere+/sm_80, T4 is sm_75) — expect a
  slower fallback kernel and check the vLLM startup log for which one actually got
  selected before drawing conclusions from quantized throughput numbers.
- `EXTRACT_DIR` is local disk (`/content/mme_dataset_images`) — see Dataset handling
  gotchas for how images get there and when they don't need to. `build_chat` reads
  from it only on an `IMAGE_CACHE` miss, so every raw-image read during the run is
  a local disk read.
- Images are read as raw bytes and base64-encoded into a `data:` URL
  (`image_to_data_url`, `image_url` content type) rather than opened via PIL and
  passed as `image_pil` — this skips a local image decode entirely (vLLM decodes
  server-side) and keeps a built chat payload's memory footprint close to the
  compressed file size instead of a full decoded RGB array.
- The encoded strings themselves are cached in `IMAGE_CACHE` (`{basename:
  data_url}`), persisted to `PROJECT_DIR/base64_image_cache.json` on Drive
  (same atomic `.part` + `os.replace()` write as `_stage_file`, for the same
  reason). Without this, the same image gets read and re-encoded once per
  model pass (chat payloads are rebuilt per pass, see below) and again on
  every fresh Colab session — the cache means that work happens at most once,
  ever, per image. `build_chat` checks the cache first and fills a miss
  inline, rather than a separate cache-building pass up front: since
  `build_chat` already runs inside `run_inference_pass`'s `img_pool`, filling
  a miss happens concurrently with GPU inference on whatever window is
  currently running, the same way window-prefetching already overlaps
  chat-building with inference. Up to 8 `img_pool` threads can write
  `IMAGE_CACHE[basename] = ...` concurrently — safe under the GIL for
  distinct keys; the only risk is two threads independently re-encoding the
  *same* basename if it's referenced by more than one question in the same
  window (confirmed this happens — the image-prep cell dedupes basenames
  for exactly this reason), which wastes a little work but never corrupts
  the dict, so it isn't worth a lock. `save_image_cache()` (defined in the
  cell right before the driving loop, alongside the vLLM compile-cache load)
  is called from a `finally` wrapped around *each model's* `run_inference_pass`
  call in the driving loop — same placement as `sync_vllm_cache_to_drive()`,
  and for the same reason: a crash on model N shouldn't throw away
  cache-filling work already done by models before it, or done so far within
  model N itself. Don't move this to a single call after the whole loop —
  that would lose an in-progress pass's cache-filling work on a crash, and
  don't add a per-window save either; that would turn one large sequential
  Drive write into many, reintroducing the exact problem staging exists to
  avoid.
- `ThreadPoolExecutor`, not `torch.utils.data.DataLoader`, parallelizes the
  per-item read+encode work. DataLoader's value is parallel tensor collation for
  a loop where *we* construct the tensors fed to a model — we don't; we hand
  `llm.chat()` a list of dicts, and vLLM's engine does its own tokenization and
  tensorization. DataLoader workers would also mean `multiprocessing`, which has
  to pickle the (sizeable) base64 strings across a process boundary, and spawning
  worker processes after CUDA is already initialized in the main process (which
  it will be, since vLLM owns the GPU) is a known source of flaky interactions.
  Threads are the right tool here: this work is I/O-bound (local file reads),
  and there's no tensor/GPU boundary in our code for pinned memory to help with
  either — that's a different abstraction level than "build a request payload."
- Chat payloads are built inside `run_inference_pass`, a `GROUP_BATCHES` window
  (currently 3 chunks, `CHAT_GROUP_SIZE` items) at a time, immediately before that
  window's inference calls — not for the whole sample up front. This bounds memory
  to one window's worth of chats regardless of sample size, at the cost of
  redoing the read/encode once per model pass instead of once for all three;
  that's an acceptable trade specifically because staging already made those reads
  local and cheap. If you change this, keep chat-building windowed and immediately
  followed by its own inference calls — don't go back to building the whole
  sample's chats before any inference starts.
- The window *after* the one currently running inference is prefetched on a
  dedicated single-worker thread pool while `llm.chat()` runs, rather than being
  built only once inference for the current window finishes. GPU inference for a
  window of `CHAT_GROUP_SIZE` multimodal requests takes meaningfully longer than
  reading+base64-encoding the next window's local images, so this hides most of
  the build cost behind GPU time. If you change the chat-building loop, keep this
  one-window-ahead prefetch rather than reverting to build-then-run-then-build.
- Inference runs in chunks sized to `MAX_NUM_SEQS` (currently 64, shared with
  `max_num_seqs` above so each chunk actually saturates vLLM's scheduler) using a
  single `llm.chat(...)` call per chunk — one call per chunk lets vLLM's continuous
  batching interleave prefill/decode across the whole chunk, rather than many small
  serialized calls that would leave the GPU idle between them. Results are written
  to the results JSON incrementally after every chunk — keep this pattern for any
  new run loop so a killed Colab kernel doesn't lose completed work.
- Between loading different model variants in the same session, always tear down
  the previous one first: `destroy_model_parallel()`, `del llm`, `gc.collect()`,
  `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`. This matters even more
  now that memory (`memory_gb.peak_allocated`/`unload_residual`) is measured per
  run — any measurement taken without this teardown is contaminated by the prior
  model's residency.
- `sample_questions.json` (10% stratified by `Task`, cached under `PROJECT_DIR`) is
  the fast-iteration dataset. Full-dataset runs are a separate, more expensive pass
  — don't conflate the two when comparing numbers across runs. Because image fetching is
  demand-driven (only images referenced by `questions` and missing from
  `IMAGE_CACHE` are fetched), a full-dataset run means fetching, encoding and caching
  ~10x more images on its first cold run, and `IMAGE_CACHE` growing accordingly
  (written to Drive after every model pass) — that's the part this design hasn't
  been sized for. Sampling is seeded
  (`random.seed(0)`) — the cached JSON is the real cross-model "comparable"
  guarantee within one run, but the seed also makes a *freshly regenerated* sample
  (cache deleted, or `DS_COMPR_RATE` changed) reproducible run-to-run instead of
  silently comparing different 10% subsets.
- vLLM's `torch.compile` cache (compiled Triton/CUDA kernels — not the CUDA
  graphs themselves, which are cheap to re-capture fresh every process start;
  it's the *compilation* feeding them that's slow and actually worth
  persisting) is pointed at local disk via `VLLM_CACHE_ROOT` (env-setup cell),
  not left at its `~/.cache/vllm` default, and not put on Drive directly
  either — engine startup touches many small files in that cache, exactly the
  access pattern Drive's FUSE mount handles worst (same reasoning as the image
  staging above). Instead, the local cache directory is round-tripped to a
  single tarball on Drive (`VLLM_CACHE_STORE`): loaded once at session start
  (cell right before the inference loop; a corrupt/truncated tarball logs a
  warning and falls back to a cold cache instead of raising, since this is a
  speed optimization and shouldn't be able to trip `BREAK_ON_ERR`), and synced
  back via `sync_vllm_cache_to_drive()` after *every* model pass in the driving
  loop, not just once at the end, so a crash on a later model (or a
  `BREAK_ON_ERR` disconnect) doesn't throw away compile work already paid for
  by models that finished first. Written with the same temp-name-then-
  `os.replace()` atomic pattern as `_stage_file`, for the same reason — a tar
  write interrupted mid-stream shouldn't leave a corrupt tarball under the
  final name.

## Evaluation & benchmarking

- Accuracy: the cloned `MME-RealWorld` repo's own
  `evaluation/eval_your_results.py` is run as a subprocess against each results JSON.
  Don't reimplement MME scoring — shell out to the upstream script so numbers stay
  comparable to published results.
- Two separate output locations, not one — don't merge them:
  - `PROJECT_DIR/test_results/` on Drive holds the raw per-item model output
    (`<key>_MME_res.json`, what the eval script actually reads). This is the
    "heavy, per-run" artifact called out in Repo shape above — it stays on
    Drive, not in git.
  - `PROJECT_DIR/benchmarks/` on Drive holds the small, git-friendly artifacts:
    `<key>_metrics.json` per model, a combined `benchmark_summary.json`, and
    `<key>_output.txt` (the eval script's stdout/stderr). They are written
    straight to Drive, never to a relative path: a relative `benchmarks/`
    resolves to the Colab VM's local disk, which is wiped when the runtime
    disconnects — and `BREAK_ON_ERR` or the final `runtime.unassign()` disconnect
    it, so a "copy to Drive at the end" step is exactly the kind of step that
    never runs (this lost a complete run's metrics once). After a run, copy this
    folder into the repo's `benchmarks/` and commit it — these are the "result
    files" Commit discipline below means, just numbers/text.
- Benchmarking metrics captured per quantization method (`run_inference_pass`
  returns them as one dict; the driving loop writes each model's dict to
  `PROJECT_DIR/benchmarks/<key>_metrics.json`, plus a combined `benchmark_summary.json`):
  - `model_load_seconds` — wall time to construct `vllm.LLM(...)`, separate from
    inference wall time, so a quantization method's load-time cost (dequant
    overhead vs. smaller download) doesn't get mixed into throughput numbers.
  - Latency — average, time-weighted (i.e. weighted by output length, since longer
    generations shouldn't count the same as short ones in an average), and p50/p90/p99
    (tail latency, not just the mean).
  - TTFT (time to first token) — average and p50/p90/p99.
  - TPOT (time per output token) — average and p50/p90/p99.
  - `queue_time` / `prefill_time` / `decode_time` — vLLM's own internal
    breakdown of request time (see `IterationStats` in
    `vllm/v1/metrics/stats.py`: `queued_time = scheduled_ts - queued_ts`,
    `prefill_time = first_token_ts - scheduled_ts`,
    `decode_time = last_token_ts - first_token_ts`). A high `queue_time` under
    this notebook's 64-concurrent-sequence load points at scheduler
    backpressure, not slow per-token generation — don't attribute a latency
    regression to decode speed without checking this breakdown first.
  - `throughput.requests_per_second` / `throughput.output_tokens_per_second` —
    aggregate throughput over the whole pass's wall-clock time, the headline
    number most quantization comparisons actually want.
  - `preemptions.total` / `preemptions.requests_affected` — vLLM can preempt
    (evict and recompute) a request under KV-cache pressure; a preempted
    request's latency includes that wasted recompute, so a spike here explains
    an otherwise-confusing latency outlier rather than it being "the model got
    slower."
  - `corrupted_requests` — count of requests vLLM flagged `is_corrupted`
    (`RequestStateStats.is_corrupted`); `run_inference_pass` also prints a loud
    warning if this is nonzero, since a corrupted request's output shouldn't be
    trusted for the accuracy eval either.
  - `truncated_requests` — count of requests whose `finish_reason`
    (`CompletionOutput.finish_reason`, `vllm/outputs.py` — a stable field, not
    part of the still-experimental `RequestStateStats`) was `"length"` rather
    than a natural stop, i.e. they hit `max_tokens=512` before finishing.
    That both means their latency/TPOT reflect an artificial cutoff rather
    than the model's real stopping behavior, and risks the accuracy eval
    scoring an answer that got cut off mid-generation — `run_inference_pass`
    prints a loud warning if this is nonzero, same treatment as
    `corrupted_requests`.
  - Memory: peak VRAM during load+inference, and the delta after unload (to
    catch quantization methods that don't fully release memory). The
    post-unload reading is taken after polling until GPU memory stops
    dropping (bounded to ~10s), not a single snapshot immediately after
    teardown — vLLM's actual GPU memory lives in a spawned engine subprocess
    (see the NVML note below), which takes a moment to actually exit, and a
    snapshot taken mid-exit would overstate `unload_residual`.
  - Anything else that materially affects the latency/memory/quality tradeoff is
    fair game to add — note it in the README's methodology section when you do.
- GPU teardown (`destroy_model_parallel()`/`del llm`/`gc.collect()`/
  `torch.cuda.empty_cache()`/`torch.cuda.synchronize()`, plus the memory-
  stabilization poll above) runs in a `finally` block around the inference
  loop in `run_inference_pass`, not just after it — a crashed pass (bad
  request, engine error, a `BREAK_ON_ERR` disconnect about to fire) still
  releases the model instead of leaving it resident on the GPU and
  contaminating the next model's `mem_before` baseline if the notebook gets
  rerun in the same kernel session.
- When adding new timing instrumentation, prefer vLLM's own request-level metrics
  (per-request timestamps in `RequestOutput`/engine stats) over wall-clock timing
  wrapped around the whole batch — batch-level timing hides TTFT/TPOT and
  under-counts latency variance.
- `RequestOutput.metrics` is only populated when `disable_log_stats=False` is
  passed explicitly — the offline `LLM` class defaults this to `True` (disabled),
  which silently means every request comes back with `metrics=None` and every
  latency/TTFT/TPOT field in the output JSON is `null`, with no error anywhere
  (this happened once already — a full run's worth of metrics came back empty).
  `run_inference_pass` sets `disable_log_stats=False` explicitly for exactly this
  reason; don't remove it. Also note the field names on the populated object:
  vLLM's v1 engine originally removed per-request metrics from the offline API
  entirely, then restored them under a new type (`RequestStateStats`, not the
  older `RequestMetrics`) with different field names —
  `first_token_ts`/`last_token_ts`, not `first_token_time`/`finished_time` that
  older vLLM docs/examples may still show. vLLM's own maintainers have flagged
  this restored API as not yet stable, so `_request_metrics` reads these fields
  defensively (`getattr(..., None)`, not direct attribute access) so a future
  rename degrades to "no metrics for that request" instead of crashing the run.
  If metrics come back empty again, check `_request_metrics`'s field names
  against whatever vLLM version is actually installed before assuming the cause
  is something else — `run_inference_pass` also prints a loud warning if every
  request in a run comes back with no usable metrics, specifically so this
  doesn't go unnoticed again.
- `memory_gb` is read via NVML (`pynvml`, from the `nvidia-ml-py` package) polled
  on a background thread throughout `run_inference_pass`, not `torch.cuda.*`.
  vLLM's engine runs actual model execution (weights, KV cache, activations) in a
  separate spawned subprocess (visible in the engine startup log: "must use the
  spawn multiprocessing start method... Reasons: CUDA is initialized") — the
  parent process's own CUDA context stays essentially empty the whole time, so
  `torch.cuda.max_memory_allocated()`/`memory_allocated()` in this process always
  read back 0, regardless of how much GPU memory the model is actually using.
  NVML queries the GPU device directly, which is correct regardless of which
  process is using it.
- This NVML memory profiling hardcodes device index 0
  (`pynvml.nvmlDeviceGetHandleByIndex(0)`), which only means "the GPU" on a
  single-GPU runtime — true for every Colab accelerator runtime, not
  necessarily true anywhere else this notebook might run. It's gated behind
  `SINGLE_GPU_DEVICE` (config cell, default `1`): when set to `0`, NVML is never
  initialized, `_gpu_mem_used_bytes()` returns `None` unconditionally, the
  background polling thread never starts, and `memory_gb` comes back
  `{"peak_allocated": null, "unload_residual": null}` instead of silently
  reporting one GPU's numbers as if they were the whole picture. Don't remove
  this gate to "simplify" — reaching for `nvmlDeviceGetHandleByIndex(0)`
  unconditionally is exactly the wrong-on-multi-GPU behavior it exists to
  prevent.
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

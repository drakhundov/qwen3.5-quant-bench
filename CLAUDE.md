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

## Dataset handling gotchas

- `yifanzhang114/MME-RealWorld` on HF ships as `.tar.gz` / `.tar.gz.part_*` shards;
  the notebook reassembles parts only when a plain archive isn't already present, then
  extracts everything into a single flat `EXTRACT_DIR` (directory structure inside the
  archive is irrelevant — every question references an image by basename).
- HF snapshot downloads can leave broken symlinks (interrupted downloads); there's a
  `check_all_targets` / re-download loop for this — always leave that check in place
  before extraction rather than assuming a snapshot is complete.
- Google Drive's FUSE mount can lag behind newly extracted/written files, and can
  also fail a read outright with a plain `OSError` under load, not just be slow —
  don't assume a bare read from a Drive-mounted path is reliable; wrap it with
  retries (see `_stage_one` in the staging cell) for anything reading files that
  were just written to Drive in the same run, or reading many files from Drive at
  all.
- Both staging cells (`_stage_one` for images, `_stage_file` for weights) copy to
  a `.part`-suffixed temp name and `os.replace()` it into place atomically, rather
  than writing the destination filename directly. This matters because a copy
  interrupted partway (kernel restart, disconnect — both have happened in this
  project) would otherwise leave a truncated file under the final filename, which
  the "skip if already staged" `os.path.exists(dst)` check would then treat as
  complete forever. For images specifically this is a silent-corruption risk, not
  just a crash risk — a truncated JPEG often still decodes via PIL, just with
  visible corruption, no exception. Keep this pattern for any new staging code.
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
  locally, so it's safe to rerun.
- Models are loaded via `vllm.LLM(...)`, `dtype="float16"` (this project's Colab
  runtime is a T4 — Turing has no native bf16 tensor cores, so this isn't just a
  default), `trust_remote_code=True`, `max_model_len=32768`, `max_num_seqs=64`,
  and a `compilation_config` with `cudagraph_capture_sizes` covering a full
  range (not just the endpoints — a
  partial trailing batch with no matching captured graph silently falls back to
  slow eager execution). Quantized variants pass `quantization=<method>` (`"awq"`
  or `"gptq"`); `gpu_memory_utilization` (0.85) is held constant across all
  variants deliberately, so a smaller quantized model doesn't get an unfair
  throughput boost from extra KV-cache headroom that has nothing to do with
  quantization.
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
  `cudagraph_capture_sizes` covering a full range (not just the endpoints — a
  partial trailing batch with no matching captured graph silently falls back to
  slow eager execution). Quantized variants pass `quantization=<method>` (`"awq"`
  or `"gptq"`); `gpu_memory_utilization` (0.85) is held constant across all
  variants deliberately, so a smaller quantized model doesn't get an unfair
  throughput boost from extra KV-cache headroom that has nothing to do with
  quantization.
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
- Sampled images are staged from Drive to local disk (`/content/mme_sample_images`)
  once, before any chat payloads are built. `EXTRACT_DIR` on Drive is a single flat
  directory holding every image in the *full* dataset, not just the sample —
  looking up files in a Drive-mounted directory that large is slow per-file, Drive's
  per-user rate limiting can add further backoff under concurrent access, and Drive
  can outright fail a read with an `OSError` under load. Copying just the ~10%
  sample locally first (with retries, see above) turns every image read for the
  rest of the run into a local disk read, isolating all Drive access to this one
  step. The staging cell skips files that already exist locally, so it's safe to
  rerun if interrupted. Once staging finishes, it repoints the global `EXTRACT_DIR`
  at the local copy for the rest of the run — `build_chat` always resolves images
  relative to `EXTRACT_DIR`, so nothing downstream needs to know staging happened.
- Images are read as raw bytes and base64-encoded into a `data:` URL
  (`image_to_data_url`, `image_url` content type) rather than opened via PIL and
  passed as `image_pil` — this skips a local image decode entirely (vLLM decodes
  server-side) and keeps a built chat payload's memory footprint close to the
  compressed file size instead of a full decoded RGB array.
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
  — don't conflate the two when comparing numbers across runs, and note that a
  full-dataset run means staging (and re-staging on every fresh runtime) a much
  larger set of images, which this design hasn't been sized for.

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

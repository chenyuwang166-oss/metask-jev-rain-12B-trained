# metask-jev-rain-12B-trained

A self-hosted decision service for JevBench. It serves `POST /v1/systemone` for the unchanged
`typesafe` adapter of the official harness, and `GET /health`. The model has two parts that are
loaded together and never merged: an int4 AWQ quantization, made by us, of
`google/gemma-4-12B-it-qat-q4_0-unquantized`, and our LoRA adapter, which vLLM applies to every
request. Facts, use of public items and measurements are in [DISCLOSURE.md](DISCLOSURE.md).

## Requirements

- One NVIDIA GPU used only by this service: compute capability 8.0 or newer, NVIDIA driver 580 or
  newer. The measurements of this package were made on one NVIDIA A100-SXM4-80GB; other GPUs have
  not been measured with it. The commands use the host's GPU 0 (`--gpus '"device=0"'`).
- Docker with the NVIDIA Container Toolkit.
- Runtime image `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`
  (Python 3.12.3, vLLM 0.31.0, torch 2.13.0+cu130), the linux/amd64 image of vLLM release
  `v0.31.0`. Nothing is installed and the service has no network access.
- Model folder: the files of https://huggingface.co/MeTask-Rain/metask-jev-rain-12B-trained/tree/cf3706fdd57e23025a8c97fbde86bd5b4a547010 as regular files,
  downloaded with

  ```bash
  hf download MeTask-Rain/metask-jev-rain-12B-trained --revision cf3706fdd57e23025a8c97fbde86bd5b4a547010 --local-dir "$MODEL"
  ```

  The repository root holds the int4 base; `adapter/` holds the adapter (`adapter_config.json`,
  `adapter_model.safetensors`). A Hugging Face cache `snapshots/<revision>` directory holds links
  and cannot be mounted on its own. At startup the service checks the SHA-256 of the small model
  files, the size of the weight shards and the full SHA-256 of both adapter files
  (`MODEL_MANIFEST.json`), and refuses other weight, tokenizer, processor or template files in the
  folder; it never writes to the folder.
- Code: this repository at the commit given in the submission email.
- Harness: https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc

## Run

Copy the commands as they are, in the same Bash terminal.

1. Set four absolute host paths and the task list (container paths; the value below is the public
   set; task paths cannot contain commas). Start with an empty `WORK` on a local filesystem the
   container user can write to and that is not mounted `noexec`:

   ```bash
   PKG=/abs/path/to/metask-jev-rain-12B-trained   # this repository, mounted read-only
   MODEL=/abs/path/to/new-model-dir                # the model folder, mounted read-only
   WORK=/abs/path/to/new-work-dir                  # new for each evaluation; the only place anything is written
   JEVBENCH=/abs/path/to/jevbench                  # the harness checkout above, mounted read-only
   TASKS=/jevbench/datasets/public/easy.jsonl,/jevbench/datasets/public/original.jsonl,/jevbench/datasets/public/hard.jsonl
   ```

2. Download the model folder:

   ```bash
   hf download MeTask-Rain/metask-jev-rain-12B-trained --revision cf3706fdd57e23025a8c97fbde86bd5b4a547010 --local-dir "$MODEL"
   ```

3. Start the service:

   ```bash
   docker run -d --name jev12b --init --gpus '"device=0"' --network none --read-only --shm-size 8g \
     -v "$PKG":/pkg:ro -v "$MODEL":/model:ro -v "$WORK":/work -v "$JEVBENCH":/jevbench:ro \
     --entrypoint bash vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b \
     /pkg/serve.sh --model /model --state-dir /work/state --run-id run1
   ```

4. Follow the log until the service is ready:

   ```bash
   docker logs -f jev12b
   ```

   When a line starting with `{"status": "ok"` appears, press Ctrl-C (this ends only the log
   display) and go to step 5. If the log ends before that line, the container has exited: the
   reason is in `docker logs jev12b` and in `$WORK/state/run1-startup/attempt-*.log`; go to step 6.
   After correcting the cause, use a new `WORK` and restart from step 3.

5. Run the official harness in the same container, then summarize:

   ```bash
   docker exec -e PYTHONPATH=/jevbench jev12b python3 -B -m jevbench.cli run --adapter typesafe \
     --endpoint http://127.0.0.1:8000 --model metask-jev-rain-12B-trained --key-env '' --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl --raw-dir /work/run1/raw \
     --manifest /work/run1/manifest.json --cap-usd 15 --price-in-per-m 0.05 --price-out-per-m 0.15
   docker exec -e PYTHONPATH=/jevbench jev12b python3 -B -m jevbench.cli summarize --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl
   ```

6. Stop and remove the container:

   ```bash
   docker stop -t 40 jev12b && docker rm jev12b
   ```

Each evaluation uses a new `WORK` and a new container. A run name that already exists below the
state directory is refused at startup ("use a new --state-dir"); with these commands that means a
new `WORK`. The slow-path quota is counted per run in arrival order, so a run always starts from a
freshly started service.

## What to expect

- Health line: `"status": "ok"`, `"warnings": []`, `"model_key": "metask-gemma4-12b-lora-awq"`,
  `"mode": "routed"`, `"prefix_caching": true`, `"supports_slow_session": true`, and
  `threshold_identity.adapter` equal to `/model/adapter`, the verified adapter folder.
- Startup: `launch.py` verifies the model folder and the adapter, writes the run configuration
  (with the adapter folder filled in) to `$WORK/state/run1-startup/config.runtime.json`, and starts
  vLLM with `enable_lora` and `max_lora_rank=16`; it sets `VLLM_BATCH_INVARIANT=1` and
  `VLLM_USE_FLASHINFER_SAMPLER=0`. Each startup attempt waits at most 1800 s for readiness: the
  first lets vLLM select the attention backend; only if the engine fails to start does a second
  attempt request `TRITON_ATTN`. The log of attempt N is `$WORK/state/run1-startup/attempt-N.log`.
- Writes: the service writes files only below `$WORK/state` (plus shared memory in `/dev/shm`) and
  the harness only below `$WORK/run1`. The package and model mounts are read-only. Cache and
  configuration path variables (`HF_HOME`, `XDG_*`, `TRITON_*`, `TORCH_HOME` and similar) must not
  be passed to the container: a value outside `/work/state` is refused at startup. No script of
  this package deletes directories recursively.
- Responses: an item the service cannot process returns HTTP 422 without answer and without usage;
  after an engine failure, and from the third consecutive item failure on, the service answers
  every `POST /v1/systemone` with HTTP 503 and `/health` reports `"status": "error"`. No substitute
  answer is ever returned.

## Package checks (offline, no GPU or network)

```bash
python3 -B pkg_check.py             # package integrity, configuration and adapter contract
python3 -B selftest.py --offline    # synthetic software tests
python3 -B pkg_check.py --release   # lists what is still missing; always exits 2
```

`MODEL_MANIFEST.json` pins the repository, revision and every model and adapter file; `FREEZE.json`
records the freeze of the fitted configuration; `MANIFEST.sha256` binds all package files.
`MODEL_CARD.md` is the model card of the Hugging Face repository.

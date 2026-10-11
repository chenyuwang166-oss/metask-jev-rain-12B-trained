# metask-jev-rain-12B-trained: disclosure

## Model

- **Base model:** Gemma 4 12B (`google/gemma-4-12B-it`), official QAT checkpoint
  `google/gemma-4-12B-it-qat-q4_0-unquantized` at revision `b6ed86275a6a5735884e208bfed95b445a684ca2`.
- **Our weights:** a LoRA adapter (rank 16, alpha 32, q/k/v/o/gate/up/down projections) trained by us
  in BF16 on that unquantized checkpoint, using our own engineering data. The adapter file is
  byte-identical to the selected training checkpoint; `adapter_config.json` names the official base
  instead of a local path.
- **Quantization of the base:** our int4 quantization of the official checkpoint
  (llmcompressor AWQ, compressed-tensors W4A16 pack-quantized, group size 32, asymmetric).
  Embeddings, vision/audio projections and `lm_head` are not quantized.
- **Serving:** the adapter is not merged. vLLM 0.31.0 loads the int4 base and applies the adapter
  with `enable_lora` (`max_lora_rank=16`) on every request.

## Service

- The service reads answer-slot probabilities in a single forward pass; up to 15% of items take a
  second pass with a short reasoning draft before the answer. The second-pass share is lowered when
  the running cost reaches 85% of the cost limit.
- Output probabilities are calibrated. Routing and calibration parameters were fitted only on the
  231 public JevBench items; the adapter's training data is our own engineering data.

## Measurement conditions

- Hardware: one NVIDIA A100 80GB; vLLM 0.31.0, torch 2.13.0 (CUDA 13.0); `VLLM_BATCH_INVARIANT=1`.
- Cost: engine-basis token usage multiplied by an assumed price of USD 0.05 / 0.15 per million
  input / output tokens; this is an estimate, not a published tariff.
- Latency: adjusted p50 = 2 x raw p50 + 0.15 s.

| Check | Result |
|---|---|
| Engine-basis cost per 1,000 decisions (limit 0.0646) | USD 0.0461 (measured on 2,005 of our own evaluation items, second pass on 14.4% of them) |
| Adjusted p50 latency (limit 1.233 s) | 0.452 s |
| Reproducibility, 50 public items x 4 fresh processes, max probability difference (limit 0.005) | 0.0041 |
| Readout errors | 0 of 2,005 |
| Package launcher serves the bound adapter (smoke test) | passed: same predictions and routes as the reproducibility runs; largest probability difference 0.0039 |

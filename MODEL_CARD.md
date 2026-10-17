---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-12B-it-qat-q4_0-unquantized
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
- gemma4
- compressed-tensors
- int4
- lora
- vllm
- jevbench
---

# metask-jev-rain-12B-trained

A decision model for the JevBench service package `metask-jev-rain-12B-trained`. It has two parts
that are loaded together and not merged:

| Part | Location | Content |
|---|---|---|
| Base | repository root | int4 quantization (compressed-tensors W4A16, AWQ, group size 32) of [`google/gemma-4-12B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) at revision `b6ed86275a6a5735884e208bfed95b445a684ca2`, the official QAT checkpoint of Gemma 4 12B (`google/gemma-4-12B-it`) |
| Adapter | `adapter/` | our own LoRA weights (PEFT format, rank 16, alpha 32, q/k/v/o/gate/up/down projections), trained in BF16 on the unquantized base above |

## Serving

The service package serves this repository with vLLM 0.31.0, `enable_lora` and `max_lora_rank=16`,
using `adapter/` as the LoRA for every request:

```python
from vllm import LLM
from vllm.lora.request import LoRARequest
llm = LLM(model=MODEL_DIR, dtype="bfloat16", enable_lora=True, max_lora_rank=16)
adapter = LoRARequest("adapter", 1, MODEL_DIR + "/adapter")
# llm.generate(prompts, sampling_params, lora_request=adapter)
```

Loading only the root files gives the base checkpoint without our decision adapter.

## Measured service properties

Measured on one NVIDIA A100 80GB with `VLLM_BATCH_INVARIANT=1`, by calling the package's service
code directly, one request at a time; model loading and warm-up are excluded. These are local
measurements, not official benchmark results.

| Property | Value |
|---|---|
| Engine-basis cost per 1,000 decisions (USD 0.05 / 0.15 per million input / output tokens) | USD 0.0461 (measured on 2,005 of our own evaluation items, second pass on 14.4% of them) |
| Latency p50, raw / adjusted (2 x raw + 0.15 s) | 0.151 s / 0.452 s |
| Reproducibility: 50 public items x 4 fresh processes, max probability difference | 0.0041 |
| Readout errors | 0 of 2,005 |

For comparison, the [organizer's leaderboard](https://benchmarkheaven.com/jev-models) reports an on-GPU p50 of 0.115 s and an adjusted p50 of 0.38 s (checked 2026-10-17). The local A100 measurements above use a different setup and are not interchangeable with the organizer's measurements.

## Data and license

The adapter was trained on our own engineering data. The 231 public JevBench items were not used to train the adapter. They were used
to fit service routing and calibration parameters and for self-tests and
regression/reproducibility checks. These are in-sample service checks, not independent
evaluation results; see [DISCLOSURE.md](DISCLOSURE.md). License: Apache 2.0, following the
base model (see the Gemma 4 license link above).

# Remediation Summary: byteshape_devstral_small_2_24b_instruct_2512_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[byteshape_devstral_small_2_24b_instruct_2512_gguf/causal_lm/pytorch-24B_Instruct_2512_GGUF-single_device-inference]

## Result
XFAIL — 24B model dequantized from IQ4_XS GGUF to BF16 (~48 GB) exceeds single-device DRAM (~32 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
oom-24b-model-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fixes): RuntimeError: Value out of range (expected to be in range of [-15, 14], but got -4095)

Post-fix failure (hardware class): RuntimeError: TT_FATAL @ tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false — Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks, where each bank needs to store 41943040 B, but bank size is 4273390016 B (allocated: 4196980928 B, free: 76409088 B, largest free block: 37026240 B)

## Root cause

Two loader bugs blocked model loading; after those were fixed, the model OOMs on device.

**Bug 1 — mistral3 GGUF architecture unsupported (loader):** The GGUF file `Devstral-Small-2-24B-Instruct-2512-IQ4_XS-4.04bpw.gguf` declares architecture `mistral3`. transformers 5.2.0 does not include `mistral3` in `GGUF_SUPPORTED_ARCHITECTURES`, causing `ValueError: GGUF model with architecture mistral3 is not supported yet.` The architecture is structurally identical to `mistral`.

**Bug 2 — sliding_window out of XLA slice bounds (loader):** Devstral-Small-2 has `sliding_window=4096`. With `max_length=128` (seq_len=15 after tokenization), `SlidingWindowCache.update` slices with `start = -sliding_window+1 = -4095`. XLA enforces strict slice bounds of `[-seq_len, seq_len-1] = [-15, 14]`, unlike PyTorch which clamps silently. This is the original reported error.

**Hardware capacity (XFAIL):** After both loader fixes, model loading and compilation succeed. Execution OOMs: the 24B-parameter model, dequantized from IQ4_XS to BF16 by transformers (~24B × 2 bytes = ~48 GB), exceeds the device's ~32 GB DRAM. This is not a compiler bug.

## Fix

**Loader fix 1** (`tt_forge_models/byteshape_devstral_small_2_24b_instruct_2512_gguf/causal_lm/pytorch/loader.py`): Patch `GGUF_TO_TRANSFORMERS_MAPPING` and `GGUF_TO_FAST_CONVERTERS` at import time to register `mistral3` as an alias for `mistral`. Also patches `load_gguf_checkpoint` in all four call sites within transformers to rewrite `model_type=mistral3` → `mistral` in the returned config dict. Commit `76e0848720` in `tt-forge-models`.

**Loader fix 2** (same file, same commit): In `load_inputs`, after tokenizing, clamp `model.config.sliding_window` to `input_ids.shape[1]` when `sliding_window > seq_len`. This preserves semantics (full attention over all `seq_len` tokens when `seq_len < sliding_window`) while keeping XLA slice indices within `[-seq_len, seq_len-1]`.

**XFAIL marking** (`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`): Added `KNOWN_FAILURE_XFAIL` entry for this model variant. Commit `aee594313` in `tt-xla`.

## Verification
- pytest exit: FAIL (OOM — hardware class)
- Hardware:    blackhole-p150b
- Duration:    277.35s (0:04:37)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/byteshape_devstral_small_2_24b_instruct_2512_gguf/causal_lm/pytorch/loader.py` (mistral3 GGUF patch + sliding_window clamp)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aa2dce61f876ce78bc3d3cb6ec006c68c277a077 |
| tt-forge-models | 76e0848720e28a6c86a961d07e04a46f209baac7 |

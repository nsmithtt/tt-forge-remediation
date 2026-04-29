# Remediation Summary: devstral_gguf-causal_lm-pytorch-Small_2_24B_Instruct_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[devstral_gguf/causal_lm/pytorch-Small_2_24B_Instruct_GGUF-single_device-inference]

## Result
XFAIL — Devstral 24B GGUF dequantizes to ~48 GB BF16 weights; device DRAM (32 GB p150b) exhausted before execution-time buffers can be allocated. Two loader bugs were fixed along the way.

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-get-weights-map-mistral3-not-found

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_FATAL @ /home/nsmith/tt-forge-remediation/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
TT_FATAL: Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks, where each bank needs to store 41943040 B, but bank size is 4273390016 B (allocated: 4196981824 B, free: 76408192 B, largest free block: 37026240 B)
 --- ttnn::prim::tilize(...)
 --- ttnn::tilize(...)
```

Original CI failure (before loader fixes):
```
NotImplementedError: Unknown gguf model_type: mistral in gguf-py.
```

## Root cause

**Original CI failure — loader bug 1 (gguf-get-weights-map-mistral3-not-found):**
`gguf>=0.10.0` renamed `'mistral'` to `'mistral3'` in `MODEL_ARCH_NAMES`. The existing
`_patch_transformers_mistral3_gguf()` patch rerouted `model_type='mistral3'` → `'mistral'`
so the config loaded correctly, but `get_gguf_hf_weights_map` then searched for
`model_type='mistral'` in `MODEL_ARCH_NAMES`, which no longer contains that key, and raised
`NotImplementedError`. Fix: add a `patched_get_weights_map` wrapper that re-routes
`model_type='mistral'` → `'mistral3'` before calling the original function.

**Loader bug 2 (aten-slice-tensor-out-of-bounds-start):**
After fixing bug 1, execution failed with `RuntimeError: Value out of range
(expected to be in range of [-128, 127], but got -4095)`. `SlidingWindowCache.update`
computes `full_states[:, :, -sliding_window+1:, :]`; with `sliding_window=4096` and
`seq_len=128`, the start index `-4095` falls outside XLA's strict valid range
`[-seq_len, seq_len-1] = [-128, 127]`. Fix: clamp `model.config.sliding_window` to
`inputs["input_ids"].shape[1]` in `load_inputs` (semantically equivalent since full
and sliding attention are identical when `seq_len ≤ sliding_window`).

**Final failure — hardware capacity:**
After both loader bugs were fixed, the model compiled and attempted execution. Device
DRAM (32 GB total across 8 banks of 4 GB) was ~97% consumed by the model weights
(31.3 GB allocated), leaving only ~580 MB free with no contiguous block large enough
(37 MB max) for a 320 MB execution-time buffer. The 24B parameter model dequantizes
from Q4_K_M on load — `AutoModelForCausalLM.from_pretrained` produces standard
`nn.Linear` modules with float32 (or BF16) tensors, discarding the GGUF quantization.
At BF16: 24B × 2 = 48 GB exceeds the device's 32 GB, at F32: 96 GB. This is a
hardware capacity ceiling, not a compiler bug.

## Fix

**Loader (tt_forge_models commit `9b0bfb6c4ea26e9c7823381d26c6011be1c30949`):**
`devstral_gguf/causal_lm/pytorch/loader.py` — two changes:
1. Added `patched_get_weights_map` inside `_patch_transformers_mistral3_gguf()` that
   routes `model_type='mistral'` → `'mistral3'` before calling
   `orig_get_weights_map`, fixing the `NotImplementedError` from `gguf>=0.10.0`.
2. Added sliding-window clamp in `load_inputs`: if
   `model.config.sliding_window > seq_len`, set
   `model.config.sliding_window = inputs["input_ids"].shape[1]`, fixing the XLA
   out-of-bounds slice error.

**Test config (tt-xla commit `191a601819cb7a6ec33e69672e99475d1c55db34`):**
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added
`KNOWN_FAILURE_XFAIL` entry for
`devstral_gguf/causal_lm/pytorch-Small_2_24B_Instruct_GGUF-single_device-inference`
with OOM reason citing the exact bank_manager error.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    707.64s (0:11:47)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/devstral_gguf/causal_lm/pytorch/loader.py` (tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 191a601819cb7a6ec33e69672e99475d1c55db34 |
| tt-forge-models | 9b0bfb6c4ea26e9c7823381d26c6011be1c30949 |

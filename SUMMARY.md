# Remediation Summary: deepseek_r1_distill_llama_70b_gguf-causal_lm-pytorch-DeepSeek_R1_Distill_Llama_70B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_llama_70b_gguf/causal_lm/pytorch-DeepSeek_R1_Distill_Llama_70B_GGUF-single_device-inference]

## Result
XFAIL — 70B model exceeds single n150 device DRAM; OOM during inference activation allocation

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure (before fix): `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

After loader fix (actual hardware ceiling):
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 469762048 B DRAM buffer across 8 banks,
where each bank needs to store 58720256 B, but bank size is 4273390016 B
(allocated: 4113318208 B, free: 160071808 B, largest free block: 45351360 B)
```

## Root cause
Two issues were found:

1. **Loader bug (fixed):** 26 GGUF loaders in tt-forge-models monkey-patch `load_gguf_checkpoint` at import time with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 added a `model_to_load=None` kwarg to this function. When any of these patchers is active in a pytest session, all subsequent GGUF model loads raise `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Fix: change the patcher signature to `(*args, **kwargs)` and forward them to the original.

2. **Hardware capacity ceiling (XFAIL):** After the loader fix, the 70B model loads and compiles successfully but runs OOM during inference. The model weights consume ~30 GB of the device's 32 GB DRAM, leaving only ~152 MB free per bank — insufficient for the tilize activation buffers needed during execution. DeepSeek-R1-Distill-Llama-70B, even with Q4_K_M GGUF quantization dequantized to bfloat16, cannot run single-device inference on a wormhole n150.

## Fix
**Loader fix** (`tt-forge-models`, branch `remediation/deepseek_r1_distill_llama_70b_gguf-causal_lm-pytorch-DeepSeek_R1_Distill_Llama_70B_GGUF-single_device-inference`):
- Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` → `def _patched_load_gguf_checkpoint(*args, **kwargs):` in 26 loader files
- Changed `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(*args, **kwargs)` in the same 26 files

**XFAIL config** (`tt-xla`, committed on `remediation/...` branch):
- Added `deepseek_r1_distill_llama_70b_gguf/causal_lm/pytorch-DeepSeek_R1_Distill_Llama_70B_GGUF-single_device-inference: status: KNOWN_FAILURE_XFAIL` to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Verification
- pytest exit: FAIL (OOM — hardware capacity, correctly classified as XFAIL)
- Hardware:    n150
- Duration:    1465.75s (0:24:25) — model loading + compilation + OOM during first inference
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: 26 files under `*/causal_lm/pytorch/loader.py` — `_patched_load_gguf_checkpoint` signature and call
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f60d3fb99a340ac44d528f0dfcba269fbfd9f17a |
| tt-forge-models | c35bff15dc62ec612a7375f916ebf1401fad39af |

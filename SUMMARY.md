# Remediation Summary: asmodeus_24b_v2_gguf-causal_lm-pytorch-24B_v2_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[asmodeus_24b_v2_gguf/causal_lm/pytorch-24B_v2_i1_GGUF-single_device-inference]

## Result
XFAIL — Asmodeus-24B-v2 in BF16 exceeds single-device n150 DRAM capacity; OOM during input tilize at execution time

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-24b-model-exceeds-n150-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure: `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`

Reproduced as: `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

After loader fix: `TT_FATAL @ bank_manager.cpp:439: Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks, where each bank needs to store 41943040 B, but bank size is 4273390016 B (allocated: 4196976960 B, free: 76413056 B, largest free block: 37030336 B)`

## Root cause
Two bugs stacked:

1. **Loader bug (fixed):** 26 loader files in tt_forge_models that globally monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` used a narrow signature `(gguf_path, return_tensors=False)`. transformers 5.x added `model_to_load` as a kwarg to this call. When any of these loaders was imported during pytest collection (which imports all models), the patched function caused a TypeError for the asmodeus test (and any other GGUF test in the same session). The fix adds `model_to_load=None, **kwargs` to all 26 signatures and threads `model_to_load` through to the original function.

2. **Hardware capacity ceiling:** After the loader bug is fixed, the test loads Asmodeus-24B-v2 in Q4_K_M GGUF, which transformers dequantizes to BF16 (~48 GB). The n150 single device has ~34 GB DRAM (8 banks × 4.27 GB). After model weights fill 4.20 GB per bank, only 76 MB remains — insufficient to tilize a 320 MB input tensor during execution. This is hardware-class: the model is too large for a single n150.

## Fix
1. **tt_forge_models** (`remediation/asmodeus_24b_v2_gguf-causal_lm-pytorch-24B_v2_i1_GGUF-single_device-inference`):
   - Fixed `_patched_load_gguf_checkpoint` signature in 26 loader files from `(gguf_path, return_tensors=False)` to `(gguf_path, return_tensors=False, model_to_load=None, **kwargs)`, threading `model_to_load` to the original. Commit `e37576eb57`.
   - Added missing `asmodeus_24b_v2_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`. Commit `3697a69ca6`.

2. **tt-xla** (`remediation/asmodeus_24b_v2_gguf-causal_lm-pytorch-24B_v2_i1_GGUF-single_device-inference`):
   - Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` for the hardware OOM. Commit `0754c3a3e`.
   - Updated tt_forge_models submodule pointer to remediation branch. Commit `935c9cec7`.

## Verification
- pytest exit: PASS (1 xfailed)
- Hardware:    n150
- Duration:    621.35s (0:10:21)
- Tier A attempts: N/A

## Files changed
- `asmodeus_24b_v2_gguf/causal_lm/pytorch/requirements.txt` (new, in tt_forge_models)
- 26 × `*/causal_lm/pytorch/loader.py` `_patched_load_gguf_checkpoint` signatures (in tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 935c9cec721a0df94fe6dc339f6ca57d64498985 |
| tt-forge-models | 3697a69ca6cef0f595518abdf1e94f28fa968997 |

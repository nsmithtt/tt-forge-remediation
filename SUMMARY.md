# Remediation Summary: gemma_3_27b_it_vl_heretic_gguf-causal_lm-pytorch-27B_IT_ULTRA_UNCENSORED_HERETIC_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_27b_it_vl_heretic_gguf/causal_lm/pytorch-27B_IT_ULTRA_UNCENSORED_HERETIC_GGUF-single_device-inference]

## Result
XFAIL — 27B BF16 model (~54 GB) exceeds BH p150b single-device DRAM (~34 GB); hardware-class capacity ceiling

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-27b-bf16-exceeds-p150b-dram

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
info:
Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks, where each bank needs to store 28901376 B,
but bank size is 4273390016 B (allocated: 4225214784 B, free: 48175232 B, largest free block: 13683008 B)
```

## Root cause

Two loader/compiler bugs were fixed before reaching the hardware ceiling:

**Bug 1 (loader): `model_to_load` kwarg TypeError** — transformers 5.2.0+ calls `load_gguf_checkpoint` with `model_to_load=dummy_model`, but 26 GGUF loaders monkey-patched it with the narrow signature `(gguf_path, return_tensors=False)`. Fixed in tt_forge_models remediation branch (commit `1709acd0e`): widened all 26 patches to `(*args, **kwargs)`.

**Bug 2 (tt-xla): XLA lazy slice out-of-bounds negative start** — Gemma3's `SlidingWindowCache.update()` executes `full_value_states[:, :, -self.sliding_window + 1 :, :]`. With `sliding_window=1024` and a freshly-populated cache with 23 elements, the start index is `-1023`, outside XLA's allowed range of `[-23, 22]`. PyTorch eager silently clamps; XLA raises `RuntimeError: Value out of range`. Fixed in tt-xla remediation branch (commit `2fd2c7120`): pre-clamped out-of-bounds negative slice starts in `TorchFunctionOverride.__torch_function__` in `torch_overrides.py`.

**Final failure (hardware-class)** — After both bugs were fixed, the model compiled and began allocating device tensors. The 27B parameter BF16 model requires ~54 GB DRAM (27×10⁹ × 2 bytes). The BH p150b has 8 DRAM banks of ~4.27 GB each = ~34 GB total. The allocator fails with 48 MB free across 8 banks. This is a fundamental hardware capacity ceiling, not a compiler bug.

## Fix

1. **tt_forge_models** (`remediation/...` branch, commit `1709acd0e`):
   - Widened `_patched_load_gguf_checkpoint` signatures in 26 GGUF loaders from `(gguf_path, return_tensors=False)` / `_orig(gguf_path, return_tensors=return_tensors)` to `(*args, **kwargs)` / `_orig(*args, **kwargs)`

2. **tt-xla** (`remediation/...` branch, commits `2fd2c7120` + `adaae0c84`):
   - `python_package/tt_torch/torch_overrides.py`: Added pre-clamping of out-of-bounds negative slice start/end indices in `TorchFunctionOverride.__torch_function__` before forwarding to XLA lazy backend
   - `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` entry for this test

## Verification
- pytest exit: FAIL (hardware OOM after both bugs fixed)
- Hardware:    blackhole-p150b
- Duration:    1018.19s (0:16:58) — third run after both fixes applied
- Tier A attempts: N/A

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models/*/loader.py` (26 files, via tt_forge_models submodule)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | adaae0c84b499ba8697ffa5d592a6724cf466da5 |
| tt-forge-models | 1709acd0e471f44effd6f963fea59be4bf0532f2 |

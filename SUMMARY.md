# Remediation Summary: ddh0_gemma_3_40b_gguf-causal_lm-pytorch-DDH0_GEMMA_3_40B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ddh0_gemma_3_40b_gguf/causal_lm/pytorch-DDH0_GEMMA_3_40B_GGUF-single_device-inference]

## Result
XFAIL — 40B BF16 model (~80 GB) exceeds single-device DRAM capacity (~34 GB on BH; ~12 GB on n150)

## Stack layer
loader, tt-xla, hardware-class

## Tier
N/A

## Bug fingerprint
dram-buffer-capacity-40b-bf16-model

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original:
```
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
```

After loader and tt-xla fixes:
```
E   RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
E   info:
E   Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks,
E   where each bank needs to store 28901376 B, but bank size is 4273390016 B
E   (allocated: 4225171776 B, free: 48218240 B, largest free block: 13801280 B)
```

## Root cause
Two bugs blocked the test before hitting the hardware ceiling:

1. **GGUF `model_to_load` kwarg (loader)** — transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`. Twenty-six GGUF loader wrappers in tt_forge_models had a narrow signature `(gguf_path, return_tensors=False)` that caused `TypeError` when the new kwarg was passed. Fixed in the tt_forge_models submodule.

2. **`aten.slice.Tensor` OOB negative start (tt-xla)** — Gemma3 `SlidingWindowCache` calls `full_value_states[:, :, -1023:, :]` on a dim-23 tensor. PyTorch eager silently clamps out-of-range negative indices (returning all 23 elements), but the XLA lazy backend raises `Value out of range` instead. Fixed by pre-clamping `start`/`end` to `[-size, size]` in `TorchFunctionOverride.__torch_function__`.

After both fixes, the test compiled and executed on device for 23 min 59 s, then failed with DRAM OOM when the runtime tried to allocate a 27.6 MB/bank tilized input buffer. The BH device has ~34 GB total DRAM (8 banks × 4.27 GB); with the compiled 40B BF16 model consuming ~33.7 GB, only 46 MB remained free (fragmented: largest block 13.2 MB). The 40B model in BF16 (~80 GB) fundamentally exceeds the device's single-device DRAM capacity.

## Fix
1. **tt_forge_models** (`e6f68b0fd493ba15abdb06c1965cc7565e381136`): Fixed `_patched_load_gguf_checkpoint` narrow signature — forwards `*args, **kwargs` so the `model_to_load` kwarg passes through.

2. **tt-xla** (`86648f2add50653b11206d6f2322de9224ac86c7`): Clamped out-of-bounds negative slice indices in `python_package/tt_torch/torch_overrides.py` `TorchFunctionOverride.__torch_function__`.

3. **tt-xla** (`8a421171c1b9a84b95e13178536f097fdb0730fb`): Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL (TT_FATAL DRAM OOM)
- Hardware:    blackhole-p150b
- Duration:    1439.17s (0:23:59)
- Tier A attempts: N/A

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models` (submodule pointer)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8a421171c1b9a84b95e13178536f097fdb0730fb |
| tt-forge-models | e6f68b0fd493ba15abdb06c1965cc7565e381136 |

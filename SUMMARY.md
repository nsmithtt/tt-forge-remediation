# Remediation Summary: gemma_3_27b_it_abliterated_normpreserve_gguf-causal_lm-pytorch-27B_IT_Abliterated_NormPreserve_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_27b_it_abliterated_normpreserve_gguf/causal_lm/pytorch-27B_IT_Abliterated_NormPreserve_GGUF-single_device-inference]

## Result
XFAIL — 27B bfloat16 model exceeds single-device DRAM capacity; KNOWN_FAILURE_XFAIL added after two intermediate bugs were fixed

## Stack layer
loader, tt-xla, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-27b-dram-oom

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure (transient):
```
2026-04-23 23:02:26.362 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)
```

On reproduction, two underlying bugs blocked reaching the model forward pass:

1. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

2. After fixing #1: `ValueError: Value out of range (expected to be in range of [-23, 22], but got -1023)` on `aten.slice.Tensor` — Gemma 3 SlidingWindowCache with window=1024 and 23 tokens computes start=-1001.

After fixing both, the 27B model OOMs during inference:
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks,
where each bank needs to store 28901376 B, but bank size is 4273390016 B
(allocated: 4225214784 B, free: 48175232 B, largest free block: 13683008 B)
```

## Root cause
Three separate issues in sequence:

1. **Loader bug** (`gguf-load-checkpoint-model-to-load-kwarg`) — 26 tt_forge_models loaders monkey-patch `load_gguf_checkpoint` at import time with a narrow `(gguf_path, return_tensors=False)` signature. transformers 5.2.0 added `model_to_load=dummy_model` to the call site. Because pytest collects all loaders, one of these patches the global before the test runs.

2. **tt-xla bug** (`aten-slice-tensor-out-of-bounds-start`) — `TorchFunctionOverride` did not intercept `aten.slice.Tensor`. Gemma 3 sliding window attention (window=1024) with a 23-token input computes start = -1023, below the valid range `[-23, 22]`. XLA lazy backend raises `ValueError`; PyTorch eager silently clamps.

3. **Hardware capacity** — Gemma 3 27B in bfloat16 requires approximately 54 GB of DRAM. A single n150 device has approximately 12 GB; a p150b has approximately 34 GB. The model exceeds single-device capacity on both hardware classes.

The originally reported `TT_FATAL: connects to a remote mmio device` is a transient device initialization error. The CI framework explicitly excludes this string from the `tt_fatal` failure category; it is not model-specific and not the root cause.

## Fix
**Fix 1 — loader (tt_forge_models, commit `20203371ca60f2dc8f0cae512eb571ce01edefa2`):**
Changed all 26 GGUF loader files from narrow signature:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to variadic:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```

**Fix 2 — tt-xla (`python_package/tt_torch/torch_overrides.py`, commit `7e9b7236f`):**
Added `aten.slice.Tensor` intercept in `TorchFunctionOverride.__torch_function__` to pre-clamp `start` and `end` to `[-size, size-1]` when they fall below `-size` and `size` is a statically-known int, matching PyTorch eager silent-clamp semantics.

**Fix 3 — test config (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`, commit `a91a6ff5e`):**
Added `KNOWN_FAILURE_XFAIL` entry with the DRAM OOM message as the reason.

## Verification
- pytest exit: PASS (1 xfailed)
- Hardware:    n150
- Duration:    145.90s (0:02:25)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — aten.slice.Tensor clamp fix
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — XFAIL entry
- `tt-xla/third_party/tt_forge_models` — submodule pointer to commit with GGUF narrow-signature fix
- 26 × `tt_forge_models/*/pytorch/loader.py` — GGUF narrow-signature fix (in tt_forge_models repo)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a91a6ff5ea060a0c098930ee7704e28043b457f1 |
| tt-forge-models | 20203371ca60f2dc8f0cae512eb571ce01edefa2 |

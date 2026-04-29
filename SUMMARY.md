# Remediation Summary: gemma_3_27b_abliterated_i1_gguf-causal_lm-pytorch-27B_ABLITERATED_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_27b_abliterated_i1_gguf/causal_lm/pytorch-27B_ABLITERATED_I1_GGUF-single_device-inference]

## Result
XFAIL — 27B bfloat16 model exceeds single-device DRAM capacity (31.5 GB allocated of 34.2 GB before OOM)

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
Original reported failure:
```
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
```

On reproduction, two prior bugs blocked reaching the slice error:

1. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`
   — 26 GGUF loaders in tt_forge_models patched `load_gguf_checkpoint` at import time with
   a narrow `(gguf_path, return_tensors=False)` signature; transformers 5.2.0 added
   `model_to_load=dummy_model` to the call.

2. After fixing #1, the test loaded the model and reached the Gemma 3 SlidingWindowCache
   path where `slice(tensor, dim, kv_seq_len - window_size, ...)` produces a start index of
   `23 - 1024 = -1001` on a dim-23 KV cache. XLA lazy backend rejects indices outside
   `[-size, size-1]` (unlike PyTorch eager which silently clamps).

After fixing both #1 and #2, the test loads and runs the 27B model but OOMs during inference:
```
Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks,
where each bank needs to store 28901376 B, but bank size is 4273390016 B
(allocated: 4225214784 B, free: 48175232 B, largest free block: 13683008 B)
```

## Root cause
Three separate issues in sequence:

1. **Loader bug** — 26 tt_forge_models loaders that monkey-patch `load_gguf_checkpoint`
   used a fixed `(gguf_path, return_tensors=False)` signature. Because pytest imports all
   loaders during collection, one of these patched the global before our test ran.

2. **tt-xla bug** (`aten-slice-tensor-out-of-bounds-start`) — `TorchFunctionOverride` in
   `torch_overrides.py` did not intercept `aten.slice.Tensor`. Gemma 3 uses sliding window
   attention (window=1024); on a 23-token input, `start = seq_len - window = 23 - 1024 = -1001`,
   which is below the valid range for dim-size 23. PyTorch eager silently clamps it; XLA lazy
   raises `ValueError: Value out of range`.

3. **Hardware capacity** — Gemma 3 27B in bfloat16 requires ~54 GB; the device (Blackhole
   P150B, ~34 GB DRAM) can hold ~31.5 GB before OOM during inference activation allocation.
   This is hardware-class, not a compiler bug.

## Fix
**Fix 1 — loader (tt_forge_models):**
In `remediation/gemma_3_27b_abliterated_i1_gguf-causal_lm-pytorch-27B_ABLITERATED_I1_GGUF-single_device-inference` branch of tt_forge_models, changed all 26 loader files from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```

**Fix 2 — tt-xla (`python_package/tt_torch/torch_overrides.py`):**
Added `aten.slice.Tensor` intercept in `TorchFunctionOverride.__torch_function__` to clamp
`start` and `end` to `[-size, size-1]` when they are `< -size` and `size` is a known int.
This matches PyTorch eager's silent-clamp semantics.

**Fix 3 — test config (XFAIL):**
Added `KNOWN_FAILURE_XFAIL` entry in `tests/runner/test_config/torch/test_config_inference_single_device.yaml` with the OOM error message.

## Verification
- pytest exit: FAIL (OOM, hardware-class)
- Hardware:    blackhole-p150b
- Duration:    1067.66s (0:17:47)
- Tier A attempts: N/A

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — aten.slice.Tensor clamp fix
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — XFAIL entry
- 26 × `tt_forge_models/*/causal_lm/pytorch/loader.py` — GGUF narrow-signature fix

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6f48b99a2135cf30d63d5599055dfd4fd16a7f2a |
| tt-forge-models | fed628507eb4d45ffb1bb7569196e4629fc2c3a4 |

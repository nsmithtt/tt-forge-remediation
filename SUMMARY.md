# Remediation Summary: gemma3_1b_gguf-causal_lm-pytorch-1B_IT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_1b_gguf/causal_lm/pytorch-1B_IT_GGUF-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: GGUF loader model_to_load kwarg + XLA slice OOB clamping

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
(The originally reported error "RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)" is the second bug, surfaced after the loader bug is fixed.)

## Root cause
Two independent bugs stacked:

1. **Loader layer** — 26 GGUF model loaders in tt_forge_models patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time using a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 added a `model_to_load` keyword argument to this call site. Because test collection imports all loaders, any GGUF loader with the old signature patches the global function and breaks every subsequent GGUF model load in the same session.

2. **tt-xla compiler frontend** — PyTorch eager semantics silently clamp out-of-range negative slice indices (e.g. `t[:, :, -511:, :]` on a dim-23 tensor returns all 23 elements). The XLA lazy tensor backend in `torch/csrc/lazy/core/helpers.cpp` raises "Value out of range" instead of clamping. Gemma3's `SlidingWindowCache` produces `window_size=512` slice indices on short inputs (23-token sequence), triggering this error.

## Fix
1. **tt_forge_models** (`remediation/gemma3_1b_gguf-causal_lm-pytorch-1B_IT_GGUF-single_device-inference`): Changed the narrow `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signature to `(*args, **kwargs)` in all 26 affected loaders, forwarding to the original function transparently.

2. **tt-xla** (`remediation/gemma3_1b_gguf-causal_lm-pytorch-1B_IT_GGUF-single_device-inference`, commit `32aba9675`): In `python_package/tt_torch/torch_overrides.py`, intercept `aten.slice.Tensor` calls in `TorchFunctionOverride.__torch_function__` and pre-clamp `start`/`end` to `[-size, size]` for statically-known dimensions before dispatching to XLA.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    310.31s (0:05:10)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/*/causal_lm/pytorch/loader.py` (26 files, narrow-signature fix)
- `tt-xla/python_package/tt_torch/torch_overrides.py` (slice OOB clamping)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d5640f421d2b2284f7d744b2e9828f9789f3a893 |
| tt-forge-models | 53313542e7257ac757482f51834ae171a501d9a0 |

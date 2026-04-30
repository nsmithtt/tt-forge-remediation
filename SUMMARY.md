# Remediation Summary: gemma3-uncensored-gguf-4b-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_uncensored_gguf/causal_lm/pytorch-4B_IT_UNCENSORED_V2_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
N/A

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
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_37, 2, -1023, 9223372036854775807), kwargs = {})

Original traceback pointed to transformers/cache_utils.py line 214:
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

(Preceded by a loader bug: TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load')

## Root cause
Two bugs chained:

1. **Loader bug (tt_forge_models):** 26 GGUF loaders on this branch patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 added a `model_to_load` keyword argument to that call, raising TypeError. Since the patch is applied at import time and persists for the entire pytest session, any test whose loader is collected before the gemma3_uncensored_gguf test is affected.

2. **tt-xla frontend bug:** Gemma 3's `SlidingWindowCache.update()` computes `full_value_states[:, :, -self.sliding_window + 1:, :]`. With `sliding_window=1024` and a 23-token sequence, start becomes `-1023`, which is outside the XLA lazy backend's valid range `[-23, 22]`. PyTorch/NumPy eager semantics clamp out-of-range negative indices to 0 (returning all elements), but the XLA lazy backend raises "Value out of range" instead.

## Fix
**Fix 1 — loader layer (tt_forge_models):**
All 26 narrow-signature `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` functions were widened to `(*args, **kwargs)` and the inner `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` calls updated to `_orig_load_gguf_checkpoint(*args, **kwargs)`.

Branch: `remediation/gemma3-uncensored-gguf-4b-single-device-inference` in tt-forge-models
Commit: 394b50bfe8

**Fix 2 — tt-xla frontend:**
Added slice-index clamping to `TorchFunctionOverride.__torch_function__` in `python_package/tt_torch/torch_overrides.py`. When `func is torch.ops.aten.slice.Tensor` and the dimension size is a statically-known int, `start` and `end` are pre-clamped to `[-size, size]` before being dispatched to XLA, matching Python eager semantics.

Branch: `remediation/gemma3-uncensored-gguf-4b-single-device-inference` in tt-xla
Commit: 0187f7ba0

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    423.45s (0:07:03)
- Tier A attempts: N/A

## Files changed
- tt-xla: `python_package/tt_torch/torch_overrides.py`
- tt-forge-models: 26 loader.py files (all GGUF loaders with narrow `_patched_load_gguf_checkpoint` signature)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 0187f7ba0 |
| tt-forge-models | 394b50bfe8 |

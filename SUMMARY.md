# Remediation Summary: gemma_3_12b_it_vl_deepseek_v3_1_heretic_uncensored_thinking_i1_gguf-causal_lm-pytorch-12B_IT_VL_DEEPSEEK_V3_1_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_12b_it_vl_deepseek_v3_1_heretic_uncensored_thinking_i1_gguf/causal_lm/pytorch-12B_IT_VL_DEEPSEEK_V3_1_I1_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
N/A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start, gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Two sequential failures:

1. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`
   — Blocked model loading entirely.

2. `RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)`
   — XLA lazy backend rejected out-of-range slice indices from SlidingWindowCache.

## Root cause
**Bug 1 (loader):** 26 GGUF loaders in tt-forge-models monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time
with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0
added `model_to_load=dummy_model` to the call site; any of these loaders imported
during test collection replaces the real function with one that rejects the new
kwarg, breaking all GGUF model loading in the session.

**Bug 2 (tt-xla):** Gemma3 `SlidingWindowCache.update()` slices the KV cache with
`start = position - sliding_window + 1`. When `seq_len=32 < sliding_window=1024`,
this yields `start = -1023` on a dim of size 24, which is outside `[-24, 23]`.
PyTorch eager silently clamps out-of-range indices; the XLA lazy backend
(`torch/csrc/lazy/core/helpers.cpp`) raises `RuntimeError: Value out of range`
instead.

## Fix
**tt-forge-models** (`remediation/gemma_3_12b_it_vl_deepseek_v3_1_heretic_uncensored_thinking_i1_gguf-causal_lm-pytorch-12B_IT_VL_DEEPSEEK_V3_1_I1_GGUF-single_device-inference`):
- 26 loaders with `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):`
  widened to `def _patched_load_gguf_checkpoint(*args, **kwargs):` with inner call
  updated to `_orig_load_gguf_checkpoint(*args, **kwargs)`.
- Files: all `*/causal_lm/pytorch/loader.py` that had the narrow signature.

**tt-xla** (`python_package/tt_torch/torch_overrides.py`):
- Added intercept in `TorchFunctionOverride.__torch_function__` for
  `func is torch.ops.aten.slice.Tensor`: reads `tensor`, `dim`, `start`, `end`
  from args/kwargs; when `size = tensor.shape[dim]` is a known int and
  `start < -size`, clamps `start = max(start, -size)` (same for `end`), then
  calls `func` with the clamped args. Guard: `not torch.compiler.is_compiling()`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    726.31s (0:12:06)
- Tier A attempts: N/A

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/third_party/tt_forge_models` (submodule pointer bump)
- 26 files in `tt-forge-models/*/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d916e38e4fc4589f06f7f70881f9032fd5b6c8a1 |
| tt-forge-models | 9f13186ae230ab5ba952079cf61a08fbb278d0dc |

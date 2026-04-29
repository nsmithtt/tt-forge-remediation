# Remediation Summary: gemma3_12b_gguf-causal_lm-pytorch-chatpdflocal_12B_IT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_12b_gguf/causal_lm/pytorch-chatpdflocal_12B_IT_GGUF-single_device-inference]

## Result
SILICON_PASS — two loader/frontend bugs fixed: _patched_load_gguf_checkpoint signature and aten.slice negative index clamping

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

(Preceded by: TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load')

## Root cause
**Bug 1 (loader):** During pytest collection, all model loader modules are imported.
Several Qwen3.5/GPT-OSS loaders (qwen_3_5_imatrix_gguf, mradermacher_qwen3_5_27b_gguf,
and 24 others) define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`
at module level and monkey-patch it onto `transformers.modeling_gguf_pytorch_utils`.
In transformers 5.2.0, `modeling_utils.py:4010` does a lazy `from .modeling_gguf_pytorch_utils
import load_gguf_checkpoint` inside `from_pretrained`, picking up the last module-level
patch — the old-signature function that rejects the new `model_to_load` kwarg.

**Bug 2 (tt-xla):** `transformers.cache_utils.SlidingWindowCache.update` computes
`full_value_states[:, :, -self.sliding_window + 1 :, :]`. With `sliding_window=1024`
on a 23-token sequence, this produces `start=-1023` on dim 2 of size 23. PyTorch eager
silently clamps the index; the XLA lazy backend (torch/csrc/lazy/core/helpers.cpp) raises
"Value out of range (expected to be in range of [-23, 22], but got -1023)".

## Fix
**Bug 1:** `tt-forge-models` — `remediation/gemma3_12b_gguf-causal_lm-pytorch-chatpdflocal_12B_IT_GGUF-single_device-inference`
Updated 26 loader files to change `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`
to `def _patched_load_gguf_checkpoint(*args, **kwargs)` and forward `*args, **kwargs`
to the original. This allows the new `model_to_load` kwarg from transformers 5.2.0 to pass through.

**Bug 2:** `tt-xla` — `python_package/tt_torch/torch_overrides.py`
Added an `aten.slice.Tensor` intercept in `TorchFunctionOverride.__torch_function__`
that pre-clamps `start` and `end` to `max(index, -size)` when the tensor shape dim
is a known int, matching PyTorch eager's silent clamp behavior.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    831.93s (0:13:51)
- Tier A attempts: N/A

## Files changed
- tt-forge-models: 26 loader files under various `*/causal_lm/pytorch/loader.py` paths
- tt-xla: `python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b6934df45b8773eee05b4bc72f114876fceba47e |
| tt-forge-models | 9235e4ce41e9068932a4179c1ac065c9e53dc6c3 |

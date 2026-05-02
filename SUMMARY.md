# Remediation Summary: libretranslate_gemma3-causal_lm-pytorch-1B_IT_Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[libretranslate_gemma3/causal_lm/pytorch-1B_IT_Q4_0-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, xla-lazy-slice-oob

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

After fixing the loader, a second failure appeared:
RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
While executing %slice_6 : call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_29, 2, -1023, 9223372036854775807))
from cache_utils.py:214: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
where sliding_window=1024, so start = -1023, but the seq_len dimension is only 23.

## Root cause
Bug 1 (loader): 26 Qwen3.5 GGUF model loaders registered a monkey-patched
_patched_load_gguf_checkpoint with the narrow signature (gguf_path, return_tensors=False).
transformers 5.2.0 added a model_to_load keyword argument to load_gguf_checkpoint.
Because these loaders install their patch globally, any model loaded in the same pytest
session after one of these loaders is collected encounters the narrow-sig patch, causing a
TypeError when transformers calls load_gguf_checkpoint(..., model_to_load=dummy_model).

Bug 2 (tt-xla): The XLA lazy backend validates slice bounds strictly; start=-1023 on a
23-element dimension raises "Value out of range" instead of clamping to the dim boundary
as PyTorch eager does. Gemma3 sliding-window cache sets start = -(sliding_window-1) = -1023.

## Fix
Fix 1 - tt-forge-models: Changed signature of _patched_load_gguf_checkpoint in all 26
affected loaders from (gguf_path, return_tensors=False) to (gguf_path, return_tensors=False, **kwargs)
and threaded **kwargs through to _orig_load_gguf_checkpoint.

Fix 2 - tt-xla: Added slice-start clamping in TorchFunctionOverride.__torch_function__ in
python_package/tt_torch/torch_overrides.py. When func is torch.ops.aten.slice.Tensor,
start and end indices are clamped to [-size, size] for statically-known dimensions before
passing to the XLA backend.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    337.06s (0:05:37)
- Tier A attempts: 1

## Files changed
- tt-xla/python_package/tt_torch/torch_overrides.py
- 26 GGUF loader files in tt-xla/third_party/tt_forge_models/*/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 97d5c80d2fa7d404df466bd0a43842d2d2d578e7 |
| tt-forge-models | 50255ca5e15febae11c281f79fd53dfd0df97e9e |

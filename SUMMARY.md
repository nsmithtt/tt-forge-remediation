# Remediation Summary: gemma_2_mitra_e_i1_gguf-causal_lm-pytorch-Mitra_E_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_mitra_e_i1_gguf/causal_lm/pytorch-Mitra_E_i1_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
Two bugs were present:

1. **Loader (tt_forge_models)**: 26 loaders each monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time. These patched functions had signature `(gguf_path, return_tensors=False)` but transformers 5.x added a third positional argument `model_to_load=None`. When pytest collects all tests it imports all model loaders, so one of these 26 loaders patched `load_gguf_checkpoint` before the gemma_2_mitra test ran, causing `TypeError` when transformers called `load_gguf_checkpoint(..., model_to_load=dummy_model)`.

2. **Loader (tt_forge_models)**: The gemma_2_mitra tokenizer loaded from the GGUF file has no `chat_template` set. The `load_inputs()` method unconditionally called `apply_chat_template()`, which raised `ValueError`.

3. **tt-xla (TorchFunctionOverride)**: Gemma-2 uses sliding-window attention with `sliding_window=4096`. During inference, the KV-cache trimming computes `full_value_states[:, :, -sliding_window+1:, :]` = `aten.slice.Tensor(tensor, dim=2, start=-4095, end=inf)`. With a 12-token input the tensor has only 12 elements in dim 2. PyTorch eager silently clamps this slice start to `-12`, but XLA's lazy backend raises `RuntimeError: Value out of range (expected to be in range of [-11, 10], but got -4095)`.

## Fix

**Fix 1 — loader (tt_forge_models remediation branch)**:
Updated all 26 `_patched_load_gguf_checkpoint` function signatures from
`(gguf_path, return_tensors=False)` to `(gguf_path, return_tensors=False, model_to_load=None)`,
and updated each call-through to pass `model_to_load=model_to_load` to `_orig_load_gguf_checkpoint`.
Files: 26 loader files under `third_party/tt_forge_models/*/causal_lm/pytorch/loader.py`

**Fix 2 — loader (tt_forge_models remediation branch)**:
In `gemma_2_mitra_e_i1_gguf/causal_lm/pytorch/loader.py`, wrapped `apply_chat_template()`
call with `if self.tokenizer.chat_template is not None:` guard, falling back to raw
`sample_text` when no template is set.
File: `third_party/tt_forge_models/gemma_2_mitra_e_i1_gguf/causal_lm/pytorch/loader.py`

**Fix 3 — tt-xla (TorchFunctionOverride)**:
Added out-of-bounds slice clamping in `TorchFunctionOverride.__torch_function__` for
`aten.slice.Tensor`. When `start` or `end` are negative integers that exceed `-size`, they
are clamped to `-size` before dispatch. This matches PyTorch eager semantics and prevents
the XLA "Value out of range" error.
File: `python_package/tt_torch/torch_overrides.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    481.00s (0:08:01)
- Tier A attempts: 1

## Files changed
- `third_party/tt_forge_models/gemma_2_mitra_e_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/*/causal_lm/pytorch/loader.py` (26 files — model_to_load kwarg fix)
- `python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 88e28ca05c2470bff5efb738edaea444c6ed3068 |
| tt-forge-models | d035606776eb1ca89835479b09d8952736075ee0 |

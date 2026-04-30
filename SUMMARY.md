# Remediation Summary: gemma3_12b_glm_heretic_grande_gguf-causal_lm-pytorch-12B_GLM_Heretic_GRANDE_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_12b_glm_heretic_grande_gguf/causal_lm/pytorch-12B_GLM_Heretic_GRANDE_GGUF-single_device-inference]

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
what():  Timeout waiting for ARC msg request queue.

## Root cause
Two bugs were blocking the test:

1. **Loader — GGUF patcher missing `**kwargs`**: 26 loaders in tt_forge_models define
   `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` without `**kwargs`.
   Transformers 5.x added `model_to_load=None` to `load_gguf_checkpoint`; the old fixed
   signature raises `TypeError` when the full pytest session collects these loaders and
   installs them in the global patcher chain. A `requirements.txt` with `gguf>=0.10.0`
   was also absent from the `gemma3_12b_glm_heretic_grande_gguf` loader directory.

2. **tt-xla — `aten.slice.Tensor` out-of-bounds start index**: Gemma3's
   `SlidingWindowCache.update()` in `transformers/cache_utils.py:214` does
   `full_value_states[:, :, -self.sliding_window + 1 :, :]`. With `sliding_window=1024`
   and input `seq_len=23` (chat-template tokens for "What is your favorite city?"), the
   start index is `–1023` but the tensor dimension is only 23, making the start
   out-of-range for the XLA backend (valid range `[–23, 22]`). PyTorch silently
   clamps this to 0, but the XLA/TT backend validates strictly and raises
   `RuntimeError: Value out of range`. This Python-level exception during
   `TorchFunctionOverride.__torch_function__` propagated as a device-level hang in the
   original CI run, causing the ARC management-coprocessor queue to time out.

## Fix
**Loader fixes** (tt_forge_models, remediation branch):
- `gemma3_12b_glm_heretic_grande_gguf/causal_lm/pytorch/requirements.txt` — created
  with `gguf>=0.10.0`.
- 26 GGUF loaders (various `*_gguf/causal_lm/pytorch/loader.py`) — updated
  `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to
  `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):` and
  forwarded `**kwargs` to the `_orig_load_gguf_checkpoint(...)` call.

**Compiler-frontend fix** (tt-xla, remediation branch):
- `python_package/tt_torch/backend/passes.py` — added
  `clamp_out_of_range_slice_starts(gm)`: iterates `aten.slice.Tensor` nodes, reads
  the sliced dimension size from `node.args[0].meta["val"].shape`, and clamps any
  static negative `start` value that falls below `–dim_size` to `–dim_size`.
- `python_package/tt_torch/backend/backend.py` — imported and called
  `clamp_out_of_range_slice_starts(compiled_graph)` after `bypass_assert_tensor_metadata`
  in `torch_pass_pipeline`. No C++ rebuild required (pure Python backend change).

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    766.16s (0:12:46)
- Tier A attempts: 1

## Files changed
**tt_forge_models (remediation branch):**
- `gemma3_12b_glm_heretic_grande_gguf/causal_lm/pytorch/requirements.txt` (created)
- 26 `*_gguf/causal_lm/pytorch/loader.py` files (GGUF patcher `**kwargs` fix)

**tt-xla (remediation branch):**
- `python_package/tt_torch/backend/passes.py` (added `clamp_out_of_range_slice_starts`)
- `python_package/tt_torch/backend/backend.py` (import + call)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 991b503aca3577c7ea28294751976d0e81c3bb26 |
| tt-forge-models | 6f2801a34a30df91b195a2db6ad11e74572a9e4b |

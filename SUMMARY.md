# Remediation Summary: gemma_3_12b_it_max_horror_gguf-causal_lm-pytorch-12B_IT_MAX_HORROR_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_12b_it_max_horror_gguf/causal_lm/pytorch-12B_IT_MAX_HORROR_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
broken-gguf-filename, gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
OSError: DavidAU/Gemma-3-12b-it-MAX-HORROR-Imatrix-GGUF does not appear to have a file named Gemma-3-12b-it-MAX-HORROR-Imatrix-Q4_K_M.gguf
→ TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
→ RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
  (aten.slice.Tensor on %cat_51, dim=2, start=-1023)
```
(Reported failure `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` was the last warning line, not the actual error.)

## Root cause
Three bugs in sequence:

1. **Loader — wrong GGUF filename** (`broken-gguf-filename`): The loader had `GGUF_FILE = "Gemma-3-12b-it-MAX-HORROR-Imatrix-Q4_K_M.gguf"` but the HuggingFace repo names all files with the `D_AU-` prefix and `-imat` suffix. File not found.

2. **Loader — missing `**kwargs` in patched function** (`gguf-load-checkpoint-model-to-load-kwarg`): 26 loaders in tt_forge_models patch `gguf_utils.load_gguf_checkpoint` at import time using a fixed signature `(gguf_path, return_tensors=False)` without `**kwargs`. Transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, causing `TypeError`. All 26 patches are applied to the global `gguf_utils.load_gguf_checkpoint` during pytest collection, so any model loaded in the same session hits this bug.

3. **tt-xla — aten.slice.Tensor out-of-bounds start** (`aten-slice-tensor-out-of-bounds-start`): `SlidingWindowCache.update()` computes `full_value_states[:, :, -self.sliding_window + 1:, :]` = `start=-1023` when `seq_len=24` and `sliding_window=1024`. PyTorch silently clamps such out-of-range indices; the XLA/TT backend validates strictly and raises `RuntimeError: Value out of range`. Tier A fix: a post-export FX pass clamps negative start indices to `-dim_size`.

## Fix
1. **`tt_forge_models/gemma_3_12b_it_max_horror_gguf/causal_lm/pytorch/loader.py`**: Changed `GGUF_FILE` to the correct filename `Gemma-3-12b-it-MAX-HORROR-D_AU-Q4_K_M-imat.gguf`. Commit `2ad2178d9e` on tt-forge-models branch `remediation/gemma3_heretic_gguf-causal_lm-pytorch-4B_IT_HERETIC_GGUF-single_device-inference`.

2. **26 loaders in tt_forge_models** (bulk): Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to add `**kwargs` and forward it to the original call. Commit `70d0022578` on tt-forge-models.

3. **`tt-xla/python_package/tt_torch/backend/passes.py`** and **`tt-xla/python_package/tt_torch/backend/backend.py`**: Added `clamp_out_of_range_slice_starts` FX pass that iterates over `aten.slice.Tensor` nodes and clamps any `start < -dim_size` to `-dim_size`. Commit `fbd671748` on tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    720.07s (0:12:00)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/gemma_3_12b_it_max_horror_gguf/causal_lm/pytorch/loader.py`
- 26 files matching `*/loader.py` in tt_forge_models with `_patched_load_gguf_checkpoint` — `**kwargs` forwarding fix
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fbd671748a0c9c0d35885039a0177637de247d2f |
| tt-forge-models | 70d00225789153ebcbedcea936019e00705ce2a9 |

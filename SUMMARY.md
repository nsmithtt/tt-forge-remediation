# Remediation Summary: gemma3_4b_it_qat_gguf-causal_lm-pytorch-4B_IT_QAT_Q4_0_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_4b_it_qat_gguf/causal_lm/pytorch-4B_IT_QAT_Q4_0-single_device-inference]

## Result
SILICON_PASS — two-layer fix: 26 loaders **kwargs patch + clamp_out_of_range_slice_starts FX pass

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
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(masking the original)
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

## Root cause
Two bugs in sequence:

**Bug 1 (loader):** 26 other GGUF loaders in `tt_forge_models` (e.g. `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`) monkey-patch `transformers.gguf_utils.load_gguf_checkpoint` at import time with a fixed signature `(gguf_path, return_tensors=False)`. During pytest collection, all loaders are imported. When `gemma3_4b_it_qat_gguf` then calls `AutoModelForCausalLM.from_pretrained`, `transformers 5.2.0` invokes `load_gguf_checkpoint(..., model_to_load=dummy_model)` — the fixed-signature patch drops this kwarg and raises `TypeError`.

**Bug 2 (tt-xla):** `Gemma3ForCausalLM` uses `SlidingWindowCache.update()` which slices key/value tensors as `full_value_states[:, :, -sliding_window+1:, :]`. With `sliding_window=1024` and `seq_len=23`, the start index is `-1023` which is outside `[-23, 22]`. PyTorch clamps such indices to 0, but the XLA/TT backend raises `RuntimeError: Value out of range`.

## Fix
**Bug 1 (loader) — tt_forge_models:**
Updated all 26 affected loaders to use `**kwargs` in `_patched_load_gguf_checkpoint`:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    ...
    return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)
```
Branch: `remediation/gemma3_4b_it_qat_gguf-causal_lm-pytorch-4B_IT_QAT_Q4_0_GGUF-single_device-inference` in tt_forge_models.
Commit: `f6299a0226`

**Bug 2 (tt-xla):**
Added `clamp_out_of_range_slice_starts(gm)` FX pass in `python_package/tt_torch/backend/passes.py` that iterates `aten.slice.Tensor` nodes, reads `dim_size` from `node.args[0].meta["val"].shape`, and clamps negative start indices to `max(-dim_size, start)`. Called from `torch_pass_pipeline` in `backend.py` after `bypass_assert_tensor_metadata`.
Branch: `remediation/gemma3_4b_it_qat_gguf-causal_lm-pytorch-4B_IT_QAT_Q4_0_GGUF-single_device-inference` in tt-xla.
Commit: `80bc3bc27` (includes the submodule update to the fixed tt_forge_models)

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    390.13s (0:06:30)
- Tier A attempts: 1

## Files changed
- `python_package/tt_torch/backend/passes.py` (tt-xla) — added `clamp_out_of_range_slice_starts` FX pass
- `python_package/tt_torch/backend/backend.py` (tt-xla) — call new pass from `torch_pass_pipeline`
- 26 × `*/causal_lm/pytorch/loader.py` (tt_forge_models) — `**kwargs` in `_patched_load_gguf_checkpoint`
- `third_party/tt_forge_models` (tt-xla) — submodule pointer updated to fixed tt_forge_models commit

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 80bc3bc27d18df125eb9ee2b551302feda1f9be9 |
| tt-forge-models | f6299a0226f7247cb5b96b8472598c53b4269e9d |

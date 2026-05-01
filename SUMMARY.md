# Remediation Summary: littlemonster_12b_qvo_heretic_hf_i1_gguf-causal_lm-pytorch-12B_QVO_heretic_HF_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[littlemonster_12b_qvo_heretic_hf_i1_gguf/causal_lm/pytorch-12B_QVO_heretic_HF_i1_GGUF-single_device-inference]

## Result
SILICON_PASS

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
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(The original CI failure was ImportError: Please install torch and gguf>=0.10.0 — absent requirements.txt. The TypeError appeared in the reproduction environment because gguf was already installed from another loader, but 26 other GGUF loaders had installed a broken patch at collection time.)

## Root cause
Two bugs:

1. **Loader — missing requirements.txt**: The littlemonster loader had no `requirements.txt`, so `gguf>=0.10.0` was not declared as a dependency. In CI this surfaced as an ImportError from within `load_gguf_checkpoint`.

2. **Loader — broken `_patched_load_gguf_checkpoint` signature (26 loaders)**: 26 GGUF loaders (all qwen35/gpt-oss variants) patched `load_gguf_checkpoint` at import time with the signature `(gguf_path, return_tensors=False)`. transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` as a keyword argument. When any of these loaders was collected before the littlemonster test, the global `load_gguf_checkpoint` was replaced with this fixed-signature function, causing a TypeError.

3. **tt-xla — aten.slice OOB start (Tier A)**: After the loader bugs were fixed, Gemma3's sliding-window attention cache update (`self.values = full_value_states[:, :, -self.sliding_window + 1:, :]`) produced an `aten.slice.Tensor` with start=-1023 on a dim of 24 elements. PyTorch eager silently clamps this to -24; the XLA/TT backend raised "Value out of range (expected to be in range of [-33, 32], but got -1023)".

## Fix
**Fix 1 — requirements.txt** (`tt-forge-models`):  
Added `littlemonster_12b_qvo_heretic_hf_i1_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`.

**Fix 2 — chat_template guard** (`tt-forge-models`):  
Added `if self.tokenizer.chat_template is not None:` guard around `apply_chat_template` in `littlemonster_12b_qvo_heretic_hf_i1_gguf/causal_lm/pytorch/loader.py`.

**Fix 3 — 26-loader signature fix** (`tt-forge-models`):  
Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):` and updated the inner call to `_orig_load_gguf_checkpoint(..., **kwargs)` in all 26 affected loaders.

**Fix 4 — clamp_out_of_range_slice_starts FX pass** (`tt-xla`):  
Added `clamp_out_of_range_slice_starts()` to `python_package/tt_torch/backend/passes.py` and wired it into the pass pipeline in `python_package/tt_torch/backend/backend.py`. The pass mirrors PyTorch eager semantics by clamping negative slice start indices that are smaller than `-dim_size`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    793.02s (0:13:13)
- Tier A attempts: 1

## Files changed
**tt-forge-models (remediation branch):**
- `littlemonster_12b_qvo_heretic_hf_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- `littlemonster_12b_qvo_heretic_hf_i1_gguf/causal_lm/pytorch/loader.py`
- 26 × `*/causal_lm/pytorch/loader.py` (qwen35/gpt-oss family patchers)

**tt-xla (remediation branch):**
- `python_package/tt_torch/backend/passes.py`
- `python_package/tt_torch/backend/backend.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f6dbb2518869334885f44f15034d492e9635324a |
| tt-forge-models | 124958c5d74e441442d1cca6369c3a1df1b9f495 |

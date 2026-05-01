# Remediation Summary: bartowski_nousresearch_nouscoder_14b_gguf-causal_lm-pytorch-NousCoder_14B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_nousresearch_nouscoder_14b_gguf/causal_lm/pytorch-NousCoder_14B_GGUF-single_device-inference]

## Result
SILICON_PASS — fixed _patched_load_gguf_checkpoint missing model_to_load kwarg in 26 GGUF loaders

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(The originally reported failure message `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")` was from a CI run with a different loader ordering; the root cause is the same: a narrow-signature patch intercepting the transformers 5.x call.)

## Root cause
During pytest collection, all tt_forge_models loaders are imported. 26 GGUF loaders (Qwen3.5-family and related) define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and install it globally on `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` (and three other module bindings) at import time. In transformers 5.x, `load_gguf_checkpoint` now accepts a third positional keyword argument `model_to_load`. When the NousCoder loader calls `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`, transformers passes `model_to_load=dummy_model` to `load_gguf_checkpoint`, which by then has been replaced by the narrow-signature patched version from one of the 26 other loaders. This causes the `TypeError`.

The NousCoder loader itself is correct and does not patch `load_gguf_checkpoint`; it is the victim of cross-loader global state contamination.

## Fix
In `tt-forge-models`, cherry-picked commit `e01fa5e36c` onto a new remediation branch. The fix changes the signature of `_patched_load_gguf_checkpoint` in all 26 affected loaders from:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```

to:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)
```

This forwards `model_to_load` (and any future kwargs) through the patch chain to the original transformers function.

Files changed: 26 loader.py files in `tt-forge-models` (all Qwen3.5-family and related GGUF loaders with the narrow-signature pattern).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    534.80s (0:08:54)
- Tier A attempts: N/A

## Files changed
- tt-forge-models: 26 loader.py files (aaryan_k_qwen_3_5_2b_gguf, daniloreddy_qwen3_5_0_8b_gguf, dmind_3_mini_i1_gguf, gpt_oss_swallow_120b_rl_v0_1_gguf, gpt_oss_swallow_20b_rl_v0_1_gguf, mradermacher_bartleby_qwen3_5_4b_gguf, mradermacher_luna_qwen3_5_27b_v5_i1_gguf, mradermacher_qwen3_5_27b_gguf, mradermacher_qwen3_5_27b_homebrew_gguf, mradermacher_qwen3_5_4b_gabliterated_gguf, mradermacher_qwen3_5_4b_unfiltered_gguf, mradermacher_vilm_0_8b_sft_gguf, qwen_3_5_imatrix_gguf, tvall43_qwen3_5_2b_heretic_v3b_i1_gguf, tvall43_qwen3_5_4b_heretic_v2_i1_gguf, unified_reward_flex_qwen35_27b_gguf, and others)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 414d75b440145ffeb0f5156cc4a1c44ecc2eebc7 |
| tt-forge-models | 59deb9da19395915c9184a897f552ba5dbe63585 |

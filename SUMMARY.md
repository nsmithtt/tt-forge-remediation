# Remediation Summary: cde_small_v2-embedding_generation-pytorch-cde-small-v2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cde_small_v2/embedding_generation/pytorch-cde-small-v2-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
modernbert-rope-inv-freq-uninit-from-meta

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise RuntimeError(
```
(followed by "You are using 'from_pretrained' with a meta device context manager", then "AttributeError: 'ContextualDocumentEmbeddingTransformer' has no attribute 'all_tied_weights_keys'", then "TypeError: forward() missing 2 required positional arguments: 'dataset_input_ids' and 'dataset_attention_mask'", then "RuntimeError: mat1 and mat2 must have the same dtype, Float and BFloat16", then pcc=nan on silicon due to NaN in model output)

## Root cause
Five sequential loader bugs in the CDE-Small-V2 (jxm/cde-small-v2) loader:

1. **Meta device context (transformers 5.x)**: `ContextualDocumentEmbeddingTransformer.__init__` calls `from_pretrained` for the dataset backbone (ModernBERT-base) as a nested call, which fails with RuntimeError under transformers 5.x meta device initialization. Fix: patch `PreTrainedModel.get_init_context` to filter out `torch.device` objects from the context list.

2. **Missing `all_tied_weights_keys` (transformers 5.x)**: The remote model code never calls `self.post_init()`, so `all_tied_weights_keys` is absent when `_finalize_model_loading` accesses it. Fix: patch `_adjust_tied_keys_with_tied_pointers` to call `post_init()` if the attribute is missing.

3. **Missing dataset context tensors in `load_inputs`**: `ContextualDocumentEmbeddingTransformer.forward()` requires `dataset_input_ids` and `dataset_attention_mask` as keyword args, but `load_inputs()` only provided `input_ids`, `attention_mask`, and `token_type_ids`. Fix: add dataset context tensors to `load_inputs()`.

4. **Float32/bfloat16 dtype mismatch**: The nested `from_pretrained` for the dataset backbone loads in float32 (meta device bypassed, no dtype propagation to nested calls), leaving backbone parameters as float32 while the outer model is bfloat16. Additionally, `mean_pool` uses `int64_tensor + 1e-20` Python float which promotes bfloat16 numerators to float32. Fix: explicit `model.to(dtype_override)` cast + patch `mean_pool` to preserve input dtype.

5. **ModernBertRotaryEmbedding inv_freq uninitialized (transformers 5.x)**: `_move_missing_keys_from_meta_to_device()` unconditionally replaces ALL non-persistent buffers with `torch.empty_like()` (uninitialized garbage), even when the model was not loaded on meta device. The outer `ContextualDocumentEmbeddingTransformer.from_pretrained` runs this on the full model tree after the backbone is initialized, trashing `ModernBertRotaryEmbedding.{layer_type}_inv_freq` buffers. The CDE model's `_initialize_weights` is a no-op for `ModernBertRotaryEmbedding`, so the buffers stay garbage, producing NaN in RoPE cos/sin and NaN model output (PCC=nan). Fix: after `from_pretrained` returns, re-run `compute_default_rope_parameters` for each `ModernBertRotaryEmbedding` module.

## Fix
All fixes are in `cde_small_v2/embedding_generation/pytorch/loader.py` in `third_party/tt_forge_models`, on branch `remediation/cde_small_v2-embedding_generation-pytorch-cde-small-v2-single_device-inference`.

Two commits:
- `147f2d0c95`: fixes 1, 2, 3, 4 (meta device patch, post_init patch, load_inputs dataset tensors, dtype cast + mean_pool fix)
- `033d3116c6`: fix 5 (ModernBertRotaryEmbedding inv_freq reinit after from_pretrained)

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    101.49s (0:01:41)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/cde_small_v2/embedding_generation/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 033d3116c6e728bc6bf8128a7fb3c8df60d22360 |

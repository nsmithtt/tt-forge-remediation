# Remediation Summary: gpt_bert-masked_lm-pytorch-BabyLM_Baseline_100M_Masked_Focus-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_bert/masked_lm/pytorch-BabyLM_Baseline_100M_Masked_Focus-single_device-inference]

## Result
SILICON_PASS — four transformers 5.x loader bugs fixed; test passes on n150 in 100.55s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-gptbert-post-init-inplace-set-slice-nonpersistent-buffer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Could not set tensor of type c10::BFloat16 to a tensor of type float
```

## Root cause

Four loader-layer bugs, all caused by transformers 5.x meta-device model construction interacting with a custom `trust_remote_code` model that predates the new API:

1. **`all_tied_weights_keys` missing:** `GPTBERTForMaskedLM.__init__` never calls `self.post_init()`, so the `all_tied_weights_keys` attribute (required by `_finalize_model_loading` in transformers 5.x) is never initialized, causing `AttributeError`.

2. **`LayerNorm(elementwise_affine=False)` crash in `_init_weights`:** `_finalize_model_loading` calls `_initialize_missing_keys` → `initialize_weights()` → `smart_apply(_initialize_weights)`. `GPTBERTPreTrainedModel._init_weights` calls `module.bias.data.zero_()` on `LayerNorm` modules without guarding for `bias=None`, crashing on `elementwise_affine=False` layers.

3. **`InPlaceSetSlice` dtype mismatch (the reported error):** The cached `modeling_gpt_bert.py` (different content from the HuggingFace snapshot due to HF's module caching) uses `torch.Tensor().to(device)` to create an accumulator tensor `ret`, which is always float32 regardless of the model's dtype. The subsequent `ret.set_(bfloat16_slice)` then fails with "Could not set tensor of type c10::BFloat16 to a tensor of type float". The fix avoids `set_` entirely by returning `full_tensor[:x_idx+1]` directly — safe because `x_idx` is constant at each call site (Python for-loops are unrolled by `torch.compile`).

4. **`position_indices` non-persistent buffer uninitialised:** `Attention.__init__` registers `position_indices` with `persistent=False`, so it is not saved in the checkpoint. With meta-device construction, materialised non-persistent buffers contain garbage memory (e.g. −70368744177664001), causing `IndexError: index out of range in self` in `F.embedding`.

## Fix

All four fixes are in `gpt_bert/masked_lm/pytorch/loader.py` in `tt_forge_models`, on branch `remediation/gpt_bert-masked_lm-pytorch-babyl_m-baseline-100m-masked-focus`, commit `05b41ccee7`.

- **Bugs 1+2:** Wrap `PreTrainedModel._finalize_model_loading` with a patch that (a) initialises `all_tied_weights_keys` via `get_expanded_tied_weights_keys()` when absent, and (b) marks all `LayerNorm(elementwise_affine=False)` modules as `_is_hf_initialized = True` before delegating to the original finalizer. The patch is installed around `from_pretrained` and restored in a `finally` block.
- **Bug 3:** In `_patch_inplace_set_slice_dtype()`, find `InPlaceSetSlice` in `sys.modules` (keyed by `"babylm"` and `"modeling_gpt_bert"`) and replace its `forward` with one that returns `full_tensor[:x_idx+1]` directly instead of using `set_`.
- **Bug 4:** In `_reinit_position_indices()`, iterate over `model.model.attention_layers` and re-run the exact `make_log_bucket_position` computation from `Attention.__init__` to restore valid `position_indices` buffers.

Files changed:
- `tt-xla/third_party/tt_forge_models/gpt_bert/masked_lm/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    100.55s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/gpt_bert/masked_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7c0b4c1367a01b0abee8980cab073751161e32ac |
| tt-forge-models | 05b41ccee78a00386b0b7818a9868fa478674ceb |

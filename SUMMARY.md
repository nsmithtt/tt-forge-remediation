# Remediation Summary: babylm_baseline_100m_gpt_bert_causal_focus-pytorch-Default-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[babylm_baseline_100m_gpt_bert_causal_focus/pytorch-Default-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-post-init-all-tied-weights-keys

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'GPTBERTForCausalLM' object has no attribute 'all_tied_weights_keys'. Did you mean: '_tied_weights_keys'?

## Root cause
The custom `GPTBERTForCausalLM` model (loaded via `trust_remote_code=True` from
HuggingFace) was written for an older version of transformers. In transformers 5.x,
`PreTrainedModel.post_init()` now initialises the `all_tied_weights_keys` instance
attribute (a dict used by `_adjust_tied_keys_with_tied_pointers`). Because
`GPTBERTForCausalLM.__init__` never calls `self.post_init()`, the attribute is never
set, and `_finalize_model_loading` raises `AttributeError` before the model is returned.

Three additional loader-layer bugs appear once the first is fixed:
1. `_init_weights` unconditionally accesses `LayerNorm.bias`, but many layers in this
   model use `elementwise_affine=False` and have no bias — causes `AttributeError`
   during `_initialize_missing_keys`.
2. Non-persistent `position_indices` buffers are overwritten with uninitialised
   garbage by `_move_missing_keys_from_meta_to_device` after load.
3. `DWAModules.forward` uses `InPlaceSetSlice` (a custom `torch.autograd.Function`
   that calls `Tensor.set_()` for storage aliasing). `torch.compile`/dynamo cannot
   trace `set_()` — the `FakeTensor` retains its empty pre-`set_()` shape, so
   `tensordot` sees size-0 inputs and produces wrong output.

## Fix
All four bugs are fixed in the loader at
`babylm_baseline_100m_gpt_bert_causal_focus/pytorch/loader.py` in the
`tt-forge-models` repo (branch
`remediation/babylm_baseline_100m_gpt_bert_causal_focus-pytorch-Default-single_device-inference`):

1. Temporarily patch `PreTrainedModel._finalize_model_loading` to seed
   `model.all_tied_weights_keys = {}` when the attribute is absent (before calling
   the original finaliser), so `_adjust_tied_keys_with_tied_pointers` can populate
   it via pointer detection.
2. In the same patch, pre-mark all `elementwise_affine=False` `LayerNorm` modules
   as `_is_hf_initialized = True` so `_initialize_missing_keys` skips them.
3. After `_finalize_model_loading` returns, recompute `position_indices` for every
   `Attention` module using `make_log_bucket_position` and `config.*` attributes.
4. After model load, replace `DWAModules.init_accumulator` and `DWAModules.forward`
   on the concrete instance class with `cat`-based implementations that are fully
   traceable by dynamo, eliminating all references to `InPlaceSetSlice`/`set_()`.

The same four bugs and fixes were previously identified and fixed for the sibling
`babylm_baseline_gpt_bert_mixed/causal_lm` variant (see
`origin/remediation/babylm_baseline_gpt_bert_mixed-causal_lm-pytorch-100M-single_device-inference`
in tt-forge-models); this report applies the identical pattern to the causal-focus
variant independently.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    59.03s
- Tier A attempts: N/A

## Files changed
- `babylm_baseline_100m_gpt_bert_causal_focus/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0c47f1723ec59ce029373b80b88774c701de738d |
| tt-forge-models | dd559a87e9b66a453be036752f47438f19eec8ec |

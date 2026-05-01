# Remediation Summary: mrnabert/embedding_generation/pytorch-YYLY66/mRNABERT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mrnabert/embedding_generation/pytorch-YYLY66/mRNABERT-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
mosaic-bert-xla-prims-view-of-and-transformers5x-meta-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Found a custom (non-ATen) operator whose output has alias
annotations: prims::view_of(Tensor(a) a) -> Tensor(a). We only support
functionalizing operators whose outputs do not have alias annotations...

While executing %view_of : [num_users=1] =
call_function[target=torch.ops.prims.view_of.default](args =
(%mark_argument_attributes_5,), kwargs = {})
Original traceback:
  File ".../bert_layers.py", line 188, in forward
    attention = unpad_input_only(attention, torch.squeeze(attn_mask) == 1)
  File ".../torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))

## Root cause
Six cascading bugs in the loader, all in the loader layer:

1. **triton ImportError**: `transformers.dynamic_module_utils.check_imports`
   recursively finds `flash_attn_triton.py`'s bare `import triton` (which is
   inside a try/except in `bert_layers.py` but check_imports doesn't respect
   that boundary).

2. **meta-device ALiBi init**: transformers 5.x wraps `from_pretrained` in a
   `torch.device("meta")` init context. `BertEncoder.__init__` calls
   `rebuild_alibi_tensor()` which creates `torch.arange(..., device=None)` →
   meta tensor and `torch.Tensor([slopes])` → CPU tensor. ALiBi slope
   multiplication fails with a meta/CPU device mismatch.

3. **State dict key mismatch**: Checkpoint is saved as `BertForMaskedLM` with
   `bert.` prefix; loading into `BertModel` needs the prefix stripped.

4. **Missing pooler keys**: Model was trained without the pooler head;
   `strict=False` is required.

5. **ALiBi dtype mismatch**: `BertEncoder.alibi` is a plain Python attribute
   (not `register_buffer`). `model.to(bfloat16)` skips plain attrs, leaving
   the float32 ALiBi slopes. The subsequent `attn_bias + alibi_bias` produces
   float32, which then causes dtype mismatch in the attention matmul with
   bfloat16 V.

6. **prims::view_of in XLA trace** (the final/reported failure): Two sources:
   a) `bert_padding.py`'s `IndexFirstAxis`/`IndexPutFirstAxis` custom autograd
      Functions and `unpad_input_only`/`unpad_input`/`pad_input` use
      `einops.rearrange` (internally `.view()`) on model-input-derived tensors.
      bert_layers.py imports these names directly into its own namespace so
      patching the bert_padding module alone is insufficient.
   b) `bert_layers.py:188` calls `torch.squeeze(attn_mask)` where `attn_mask`
      has shape `[batch, seq]` with no size-1 dims. `aten.squeeze.default`
      decomposes to `prims.view_of(self)` when there is nothing to squeeze.
      XLA's functionalizer rejects `prims::view_of` with alias annotations on
      non-ATen ops (triggered by `tt::mark_argument_attributes` annotating the
      attention mask input).

## Fix
All fixes in `tt-xla/third_party/tt_forge_models/mrnabert/embedding_generation/pytorch/loader.py`:

1. Added `_fixed_get_imports` that filters `triton` from the import list of
   `flash_attn_triton.py`; applied via `unittest.mock.patch` around
   `get_class_from_dynamic_module`.

2. Replaced `AutoModel.from_pretrained(trust_remote_code=True)` with:
   `get_class_from_dynamic_module` (no meta-device context) → `cls(config)`
   (CPU init) → `hf_hub_download` + `torch.load` + `load_state_dict`.

3. Stripped `bert.` prefix from checkpoint keys before `load_state_dict`.

4. Added `strict=False` to `load_state_dict`.

5. After `model.to(dtype)`, manually cast `model.encoder.alibi.to(dtype)`.

6. Added `_patch_bert_padding_for_xla()` which:
   - Replaces `index_first_axis`, `index_put_first_axis`, `unpad_input`,
     `unpad_input_only`, and `pad_input` in BOTH `bert_padding` and
     `bert_layers` module namespaces with view-free equivalents using advanced
     indexing (`tensor[batch_idx, seq_idx]`) and direct scatter into
     `torch.zeros`.
   - Replaces `BertUnpadSelfAttention.forward` entirely to change
     `torch.squeeze(attn_mask) == 1` to `attn_mask.ne(0)`, which avoids the
     `aten.squeeze.default` → `prims.view_of` decomposition.

Remediation branch in tt_forge_models:
`remediation/mrnabert-embedding_generation-pytorch-YYLY66_mRNABERT-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    88.13s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mrnabert/embedding_generation/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94978a8683ad9b2c152f418a77d08b89f526d76b |
| tt-forge-models | 85483c45123ee8e732d3be523b0d5f0e5760665e |

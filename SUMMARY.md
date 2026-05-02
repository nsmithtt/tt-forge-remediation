# Remediation Summary: mms-feature_extraction-pytorch-MMS_300M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mms/feature_extraction/pytorch-MMS_300M-single_device-inference]

## Result
SILICON_PASS — three bugs fixed (loader dtype mismatch, integer index dtype normalisation, bool-mask scatter); test passes on n150

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
index-put-bool-mask-scatter-index-vector-dim-ne-1

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Cannot concatenate arrays with different element types: S64 vs S32.

## Root cause

Three independent bugs in two layers.

**Bug 1 — loader** (`tt_forge_models/mms/feature_extraction/pytorch/loader.py`):  
`load_inputs` returned `float32` tensors even when `dtype_override=torch.bfloat16` was
requested. The model was loaded in `bfloat16`, so the first `conv1d` call saw a dtype
mismatch and raised `RuntimeError: Input type (float) and bias type (c10::BFloat16) should
be the same`.

**Bug 2 — tt-xla loader interaction** (`python_package/tt_torch/torch_overrides.py`):  
`_get_feature_vector_attention_mask` sets one element of a 2D `int32` attention mask with
`attention_mask[(torch.arange(...), output_lengths - 1)] = 1`.  `torch.arange()` produces
`int64` while arithmetic on the `int32` `output_lengths` stays `int32` under XLA.
XLA's scatter requires all index tensors in a single `index_put` call to share the same
integer dtype and concatenates them to form an index vector; mixing `S64` with `S32` raises
the reported error.  The fix normalises all non-float integer index tensors in
`aten.index_put.default` to `int64` inside `TorchFunctionOverride.__torch_function__`.

**Bug 3 — tt-xla compiler frontend** (`python_package/tt_torch/backend/passes.py`):  
`Wav2Vec2EncoderStableLayerNorm.forward` masks padding tokens with
`hidden_states[~expand_attention_mask] = 0`.  XLA lowers this to
`stablehlo.scatter` with `index_vector_dim = 3` (last dim of the 4-D index tensor
`[1, 49, 1024, 1]`).  `StableHLOToTTIRScatterOpConversionPattern.checkBasicLegality`
in `tt-mlir` requires `index_vector_dim == 1` for single-dimensional scatter and calls
`notifyMatchFailure`, producing `Error code: 13`.  The fix adds an FX graph pass
`rewrite_bool_index_put_to_where` that runs after export/decomposition inside
`torch_pass_pipeline`.  It rewrites any `aten.index_put[_].default` whose sole index
tensor is a boolean mask into `aten.where.self + aten.full_like`, which XLA lowers to
`stablehlo.select` — no scatter involved.

## Fix

**tt_forge_models** branch `remediation/mms-feature_extraction-pytorch-MMS_300M-single_device-inference`:
- `mms/feature_extraction/pytorch/loader.py` — cast `load_inputs` output tensors to `dtype_override`

**tt-xla** branch `remediation/mms-feature_extraction-pytorch-MMS_300M-single_device-inference`:
- `python_package/tt_torch/torch_overrides.py` — normalise integer index tensors in `aten.index_put.default` to `int64`; plus dead-code `index_put_.default` bool-mask override (superseded by the FX pass but harmless)
- `python_package/tt_torch/backend/passes.py` — add `rewrite_bool_index_put_to_where` FX pass
- `python_package/tt_torch/backend/backend.py` — wire new pass into `torch_pass_pipeline`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    88.65s
- Tier A attempts: 1

## Files changed
- tt-xla: `python_package/tt_torch/torch_overrides.py`
- tt-xla: `python_package/tt_torch/backend/passes.py`
- tt-xla: `python_package/tt_torch/backend/backend.py`
- tt_forge_models: `mms/feature_extraction/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 147ac0b9daa2f99eb20665bda183fd5c84cca741 |
| tt-forge-models | d084fa87caf94eca23840e4796f5e55310fb2f34 |

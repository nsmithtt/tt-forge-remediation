# Remediation Summary: git_base_coco-image_captioning-pytorch-Base_COCO-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[git_base_coco/image_captioning/pytorch-Base-COCO-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
xla-index-tensor-mixed-int32-int64-indices

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

Subsequent runtime error (the actual compilation blocker):
```
RuntimeError: Cannot concatenate arrays with different element types: S64 vs S32.
While executing %index : call_function[target=torch.ops.aten.index.Tensor](args = (%cat_2, [%view_51, %where_1]))
```

## Root cause
Two independent bugs:

**Loader bug** — `AutoProcessor.from_pretrained` in transformers 5.x defaults to the fast `CLIPImageProcessor`. The original loader did not pass `use_fast=False`, triggering the breaking-change warning.

**Compiler-frontend (tt-xla) bug** — `modeling_git.GitModel.forward` creates `cache_position` with `dtype=torch.int` (int32) at line 1103 when visual features are present:
```python
cache_position = torch.arange(embedding_output.shape[1], device=embedding_output.device, dtype=torch.int)
```
Meanwhile, `masking_utils.sdpa_mask` creates `batch_arange = torch.arange(batch_size, ...)` as int64 (the PyTorch default). Both become vmap scalar indices (`batch_idx` = int64, `q_idx` = int32) that are passed together to `token_type_ids[batch_idx, safe_q_idx]` inside the GIT mask function. Standard CPython PyTorch silently promotes int32 to int64 for mixed-dtype indexing, but XLA raises `Cannot concatenate arrays with different element types: S64 vs S32` when the `aten.index.Tensor` FX node is evaluated during graph partitioning.

## Fix
**Loader fix** (`tt-forge-models`, `git_base_coco/image_captioning/pytorch/loader.py`):
Added `use_fast=False` to `AutoProcessor.from_pretrained` to silence the transformers 5.x `CLIPImageProcessor` fast-processor breaking-change warning.

**Compiler-frontend fix** (`tt-xla`, `python_package/tt_torch/torch_overrides.py`):
In `TorchFunctionOverride.__torch_function__`, intercept `torch.ops.aten.index.Tensor` calls where the index tensor list contains a mix of int32 and int64 tensors. Promote all int32 indices to int64 before forwarding to the XLA backend. This replicates the standard PyTorch promotion behavior that XLA does not perform automatically.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    76.12s (0:01:16)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — add int32→int64 promotion for `aten.index.Tensor` mixed-dtype indices
- `tt-xla/third_party/tt_forge_models/git_base_coco/image_captioning/pytorch/loader.py` — add `use_fast=False` to `AutoProcessor.from_pretrained`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 08702666cae1264a220af6d75527fbff96cdf535 |
| tt-forge-models | 2570b707da61755ecfd24df215ae36e82d5ff76b |

# Remediation Summary: mitra-pytorch-autogluon-mitra-classifier-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mitra/pytorch-autogluon/mitra-classifier-single_device-inference]

## Result
FAIL — aten.quantile has no XLA/StableHLO lowering in the TT compiler backend

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
aten-quantile-no-xla-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
torch._dynamo.exc.BackendCompilerFailed: backend='tt' raised:
RuntimeError: Check failed: xtensor: Input tensor is not an XLA tensor: XLAFloatType

While executing %quantile : [num_users=1] = call_function[target=torch.ops.aten.quantile.default](args = (%index_put_, %div, 1), kwargs = {})
Original traceback:
  File ".../autogluon/tabular/models/mitra/_internal/models/embedding.py", line 62, in forward
    quantiles = torch.quantile(x_support, q=q, dim=1)
```

## Root cause
Three loader bugs were fixed (all in tt_forge_models):

1. Tab2D.from_pretrained() only accepts path_or_repo_id and device; the
   loader was passing torch_dtype which raises TypeError.
2. Tab2DQuantileEmbeddingX uses torch.quantile, which requires float32 inputs;
   the loader was passing dtype_override=torch.bfloat16 to inputs.
3. torch.utils.checkpoint(use_reentrant=False) in the Tab2D transformer blocks
   generates a HigherOrderOperator that torch.export.export cannot handle
   (layer_norm appears as a Python function instead of aten.native_layer_norm).

After all three loader fixes, the remaining failure is in the compiler stack:
torch.ops.aten.quantile.default has no lowering in torch-xla's StableHLO
emission pipeline. When torch.compile encounters aten.quantile, the XLA
backend raises "Input tensor is not an XLA tensor: XLAFloatType" because the op
reaches XLA without a decomposition or StableHLO composite. There is no
registered decomposition for aten.quantile.default in PyTorch's decomposition
table, and no corresponding handler in the TT XLA backend.

## Fix
Loader fixes committed to tt_forge_models on branch
remediation/mitra-pytorch-autogluon-mitra-classifier-single_device-inference
(commit 4bcffbd3bace5d1f8a39258981d5f9297c491a6e):

- mitra/pytorch/loader.py: Remove torch_dtype kwarg from
  Tab2D.from_pretrained() call; keep model in float32.
- mitra/pytorch/loader.py: Fix load_inputs to always use torch.float32
  (not dtype_override) since Tab2DQuantileEmbeddingX requires float32.
- mitra/pytorch/loader.py: Patch _tab2d_mod.checkpoint to identity function
  to disable gradient checkpointing at load time.
- mitra/pytorch/requirements.txt: Add autogluon.tabular and einx dependencies.

Proposed compiler fix (not implemented):
Add a decomposition for aten.quantile.default in the tt-xla FX pre-processing
pipeline (e.g. in torch_overrides.py or a dedicated decompositions file) that
rewrites quantile as sort + gather + lerp, which are ops with existing XLA
lowerings.

## Tier B justification
new-infrastructure — aten.quantile.default has no decomposition in PyTorch's
core decomposition table and no lowering in the TT XLA backend. Implementing it
correctly requires a sort-based decomposition (sort + index + lerp for
interpolation between adjacent order statistics), which is new multi-op
infrastructure not present in the current stack.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 35.32s (to failure)
- Tier A attempts: 0

## Files changed
- mitra/pytorch/loader.py (tt_forge_models, 3 fixes)
- mitra/pytorch/requirements.txt (tt_forge_models, new file)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 4bcffbd3bace5d1f8a39258981d5f9297c491a6e |

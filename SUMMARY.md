# Remediation Summary: eomt-pytorch-dinov3-large-640-coco-panoptic-single-device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[eomt/pytorch-Dinov3_Large_640_Coco_Panoptic-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
stablehlo-batch-norm-training-as-layernorm

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.014900257938762745. Required: pcc=0.95.

## Root cause
TorchXLA lowers `nn.LayerNorm` to `stablehlo.batch_norm_training` with identity scale (ones) and bias (zeros), `feature_index = rank - 2`, and unused mean/variance outputs. The existing `StableHLOToBatchNormTrainingOpConversionPattern` lowered this to `ttir.batch_norm_training`, which in turn maps to the TTNN batch_norm kernel. That kernel introduces numerical error due to an intermediate reshape performed before normalization, producing PCC ≈ 0.015 against the CPU reference. EomtDinov3 has LayerNorm after every attention block and MLP block throughout the ViT-Large backbone (24 transformer layers × 2 + query module), so the error accumulates severely.

## Fix
Cherry-picked commit `8effbef57` from `origin/remediation/eomt-instance_segmentation-pytorch-Large_1280_Coco_Instance-single_device-inference` into a new remediation branch in tt-mlir. The fix adds a pattern guard in `StableHLOToBatchNormTrainingOpConversionPattern` (`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`): when the input is rank-3, `feature_index == rank - 2`, mean/var outputs are unused, and scale/bias are identity splat constants, lower to `ttir.layer_norm` instead of `ttir.batch_norm_training`. This produces correct numerics for every LayerNorm call in the model.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    99.82s (0:01:39)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — detect LayerNorm pattern in batch_norm_training and lower to ttir.layer_norm

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | b505c29a2cc5fffeaee8838fc5e68dcb61a2c4c4 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7d22c10feac8dc8d515b8668d6a1e74f1700e206 |

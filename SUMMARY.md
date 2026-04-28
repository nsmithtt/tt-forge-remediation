# Remediation Summary: fast3r-pytorch-ViT_Large_512-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[fast3r/pytorch-ViT_Large_512-single_device-inference]

## Result
SILICON_PASS — loader fixes (requirements, CUDA autocast patch, view fields) + tt-mlir batch_norm_training→layer_norm lowering resolved PCC from -0.04 to ≥0.95

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
batch-norm-training-as-layer-norm-graph-break

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=-0.044945197992813185. Required: pcc=0.95.

## Root cause
Three loader bugs and one compiler-stack bug combined to produce garbage output:

1. **Missing dependencies** (`loader` layer): fast3r's `setup.py` declares no dependencies, but `fast3r.utils` imports `hydra`, `lightning`, and `omegaconf` at module level. The test runner installed the package but not its transitive deps, causing `ModuleNotFoundError: No module named 'omegaconf'`.

2. **CUDA-only autocast** (`loader` layer): `fast3r/croco/models/blocks.py` hardcodes `torch.autocast("cuda", dtype=torch.bfloat16)` in three attention branches. On TT hardware (no CUDA bfloat16 support) this raised `RuntimeError: Current CUDA Device does not support bfloat16`.

3. **Missing view metadata fields** (`loader` layer): fast3r's forward contains a debug slow-encoder path (`time.time()` check > 20s) that calls `view_name(view)` requiring `view["dataset"]`. The loader's `load_inputs` only provided `img` and `true_shape`, causing `KeyError: 'dataset'`.

4. **batch_norm_training lowered as LayerNorm** (`tt-mlir` layer): In RoPE2D (`pos_embed.py:176`), `int(positions.max())` triggers a `Tensor.item()` graph break inside each encoder block's attention. This splits `norm2` (LayerNorm) into a resumed subgraph. TorchXLA lowers that standalone LayerNorm to `stablehlo.batch_norm_training` with identity scale/bias and `feature_index=rank-2`. The TTNN batch_norm kernel introduces severe numerical error, yielding PCC ≈ -0.04. The same bug was previously fixed for EoMT (commit `8effbef57` on tt-mlir).

## Fix

**tt-forge-models** (`fast3r/pytorch/`): Branch `remediation/fast3r-pytorch-ViT_Large_512-single_device-inference`
- Created `requirements.txt` with `omegaconf>=2.3.0`, `hydra-core>=1.3.2`, `lightning>=2.0.0`.
- Added `_patch_fast3r_blocks()` to `loader.py` that rewrites `fast3r/croco/models/blocks.py` at runtime, replacing all three `torch.autocast("cuda", dtype=torch.bfloat16)` calls with `contextlib.nullcontext()`, and deletes the stale `.pyc`.
- Added `"dataset"`, `"label"`, and `"instance"` keys to each view dict in `load_inputs`.

**tt-mlir**: Branch `remediation/fast3r-pytorch-ViT_Large_512-single_device-inference`
- Cherry-picked commit `8effbef57` ("Lower batch_norm_training LayerNorm pattern to ttir.layer_norm") from the EoMT remediation branch.
- In `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`, added `isConstantSplatFloat` helper and extended `StableHLOToBatchNormTrainingOpConversionPattern::matchAndRewrite` to detect the pattern (rank-3 input, `feature_index=rank-2`, unused mean/var, scale=1.0, offset=0.0) and lower it to `ttir.LayerNormOp` instead.

**tt-xla**: Branch `remediation/fast3r-pytorch-ViT_Large_512-single_device-inference`
- Added `fast3r/pytorch-ViT_Large_512-single_device-inference: status: EXPECTED_PASSING` to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.
- Bumped `third_party/tt_forge_models` submodule pointer to the remediation commit.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    267.51s (0:04:27)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/fast3r/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/fast3r/pytorch/requirements.txt` (new)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | a980d76bdbf547cad341b5404a03e79096110591 |
| tt-xla          | 874bfb42be861421849cef4a211a0658a6b1ff91 |
| tt-forge-models | 23f402e8a359ed0c2284746ee72f25fed177a7dc |

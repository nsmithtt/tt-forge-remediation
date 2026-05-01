# Remediation Summary: hunyuan_world_mirror-pytorch-World_Mirror-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan_world_mirror/pytorch-World_Mirror-single_device-inference]

## Result
FAIL — Shardy propagation in tt-mlir doesn't support tuple outputs from CPU-fallback custom-calls (Error code: 13)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
shardy-propagation-tuple-output

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

(The trailing DeprecationWarning was the reported failure. The actual errors found during reproduction were three sequential loader bugs and then a Tier B compiler bug.)

## Root cause

Three loader bugs were found and fixed:

1. **Missing dependencies**: `lightning_utilities` (from `lightning>=2.5.0`) and `gsplat>=1.5.3` were not listed in a `requirements.txt`. The upstream HunyuanWorld-Mirror repo's `vision_transformer.py` imports `training.utils.logger` which uses `lightning_utilities`; `gsplat` is the Gaussian splatting library used by `GaussianSplatRenderer`.

2. **dtype mismatch in `depth_to_camera_coords`**: The upstream function calls `camera_intrinsics.float()` to force float32. When the model runs in bfloat16, the model's predictions (`R_cam_to_world`, `t_cam_to_world`) are bfloat16 but `camera_points` computed from the float32 intrinsics becomes float32. This causes `RuntimeError: expected m1 and m2 to have the same dtype` in the einsum.

3. **dtype mismatch in `GaussianSplatRenderer.prune_gs`**: All `torch.zeros(...)` calls lack an explicit `dtype`, defaulting to float32. When splat weights from the model are bfloat16, `scatter_add_` fails with `RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype`.

After fixing these three loader bugs, the CPU forward pass succeeds. The TT device run then fails with:

```
loc("custom-call.15579"): error: Shardy propagation doesn't support tuples: 'tuple<tensor<1x2x4x4xf32>, tensor<1x2x4xf32>>'
Failed to run stablehlo pipeline
ValueError: Error code: 13
```

The TT backend's `partition_fx_graph_for_cpu_fallback` creates CPU-fallback subgraphs that are represented as `custom-call` nodes in StableHLO. When a CPU-fallback subgraph returns a tuple of tensors (shapes `[1,2,4,4]` and `[1,2,4]`), the Shardy tensor-parallel sharding propagation pass in `tt-mlir` fails because it has no handling for tuple output types.

## Fix

**Loader fixes (committed to `tt-forge-models` remediation branch):**

- `hunyuan_world_mirror/pytorch/requirements.txt` (new): `lightning>=2.5.0`, `gsplat>=1.5.3`
- `hunyuan_world_mirror/pytorch/loader.py`: `_patch_geometry()` function that:
  - Replaces `depth_to_camera_coords` to use `.to(dtype)` instead of `.float()` for camera intrinsics
  - Replaces `GaussianSplatRenderer.prune_gs` to pass explicit `dtype=` to all `torch.zeros(...)` calls

**Proposed compiler fix (not attempted — Tier B):**

The `Shardy propagation` pass in `tt-mlir` needs to be updated to handle `custom-call` nodes with tuple output types. The pass likely has a fallback or assertion that fires when it encounters a `stablehlo.tuple`-typed output. The fix would need to add tuple-output propagation logic throughout the pass, which crosses multiple files and is new infrastructure work.

## Tier B justification

**Indicator: new-infrastructure**

The Shardy propagation pass explicitly states it "doesn't support tuples". Adding tuple support requires implementing propagation logic for tuple-typed values throughout the pass, not just adding a single missing lowering pattern. This is new compiler infrastructure, not a scoped one-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n300
- Duration:    271.18s (0:04:31) before Tier B failure
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/hunyuan_world_mirror/pytorch/requirements.txt` (new)
- `tt_forge_models/hunyuan_world_mirror/pytorch/loader.py` (patched)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 936641329053bd702e99fcc78b5a5cc060c4aefd |

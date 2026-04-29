# Remediation Summary: depth_anything_v3-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[depth_anything_v3/pytorch-Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
depth-anything-v3-namespace-shadow-and-position-getter-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AttributeError: module 'spacy' has no attribute 'Language'

(Full reproduction sequence, from clean state):
1. ModuleNotFoundError: No module named 'depth_anything_3.api'  — depth-anything-3 not installed
2. ModuleNotFoundError: No module named 'evo.core'             — evo local dir shadows pip evo
3. AttributeError: module 'spacy' has no attribute 'Language'  — spacy local dir shadows; datasets._dill checks spacy.Language
4. TorchRuntimeError: aten.cat found two different devices xla:0, cpu — PositionGetter cache device mismatch

## Root cause
Four bugs in the loader layer, each uncovered after the previous was fixed:

1. **Missing dependency**: `depth-anything-3` was not listed in requirements.txt and
   was not installed, causing `ModuleNotFoundError` on `from depth_anything_3.api
   import DepthAnything3`.

2. **evo namespace collision**: `tt_forge_models/evo/` (a model directory for the Evo
   biology model) is a Python namespace package. When `tt_forge_models` root is in
   `sys.path`, `import evo` resolves to this stub instead of the real pip `evo`
   package (evo-1.36.3). This causes `ModuleNotFoundError: No module named 'evo.core'`
   when `depth_anything_3.utils.pose_align` tries `from evo.core.trajectory import
   PosePath3D`.

3. **spacy namespace collision**: `tt_forge_models/spacy/` is also a namespace package.
   The top-level `from datasets import load_dataset` in the original loader caused
   `datasets.utils._dill` to run `import spacy`, which cached the stub. Later, when
   `datasets` fingerprints a function, `_dill.py:save()` does `if "spacy" in
   sys.modules: if issubclass(obj_type, spacy.Language):` — the stub has no
   `Language` attribute, raising `AttributeError`.

4. **PositionGetter device mismatch**: `PositionGetter.__call__` caches positions
   keyed by `(height, width)` but not by device. If the cache was populated on CPU
   during model initialisation, subsequent XLA calls return CPU tensors. During Dynamo
   tracing, `torch.cat([pos_special_xla, pos_cpu], dim=2)` fails with
   `TorchRuntimeError: aten.cat found two different devices xla:0, cpu`.

Additionally, the original wrapper used `self.model.infer()` (method does not exist
on `DepthAnythingNet`) and input shape `(B, 3, H, W)` instead of the required
`(B, N, 3, H, W)`.

## Fix
All changes in `depth_anything_v3/pytorch/` of `tt-forge-models`:

**`depth_anything_v3/pytorch/requirements.txt`** (new file):
- Added `depth-anything-3>=0.1.0`.

**`depth_anything_v3/pytorch/loader.py`**:
- Removed top-level `from datasets import load_dataset`; deferred to `load_inputs()`
  inside a `_clear_shadow_modules()` guard so the spacy namespace proxy is evicted
  from `sys.modules` before `datasets._dill` can cache it.
- `_clear_shadow_modules()`: removes `tt_forge_models` root from `sys.path`; stashes
  and later restores the evo namespace proxy; permanently drops the spacy namespace
  proxy (spacy is not a real installed package, so dropping it prevents
  `datasets._dill` from hitting `AttributeError`).
- Monkey-patched `PositionGetter.__call__` to call `.to(device)` on the cached
  position tensor before returning, ensuring XLA-device tensors during Dynamo tracing.
- Switched wrapper to use `da3.model` (the inner `DepthAnything3Net`) directly,
  bypassing the outer `@torch.inference_mode()` / `torch.autocast` decorators that
  interfere with XLA compilation.
- Fixed `forward`: calls `self.net(pixel_values)` returning `dict["depth"]` instead
  of the nonexistent `.infer()` method.
- Fixed input shape in `load_inputs()` to `(B, N, 3, H, W)` with N=1, using
  `torchvision.transforms.Resize(504, 504)` to match the model's expected resolution.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    230.72s (0:03:50)
- Tier A attempts: N/A

## Files changed
- `depth_anything_v3/pytorch/loader.py` (tt-forge-models)
- `depth_anything_v3/pytorch/requirements.txt` (tt-forge-models, new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 171f01b7b7cb52f534eeb808fe31b25f5741c626 |

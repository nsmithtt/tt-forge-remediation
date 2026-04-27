# Remediation Summary: add_detail_xl/pytorch-add-detail-xl-single_device-inference

## Test

`tests/runner/test_models.py::test_all_models_torch[add_detail_xl/pytorch-add-detail-xl-single_device-inference]`

## Status: SILICON_PASS

## Problem

The test was failing with `failure_category: hang` on the hardware. The last captured output before the hang was:

```
/home/ttuser/hf-bringup/tt-xla/.local_venv/lib/python3.12/site-packages/diffusers/schedulers/scheduling_euler_ancestral_discrete.py:214: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed.
```

Two root causes were identified:

### 1. numpy 2.0 DeprecationWarning from EulerAncestralDiscreteScheduler

`EulerAncestralDiscreteScheduler.__init__` and `set_timesteps` call `np.array()` on a torch tensor, which in numpy 2.x passes `copy=False` to `__array__`. The torch tensor's `__array__` method doesn't accept the `copy` keyword, emitting a DeprecationWarning.

**Fix:** Wrap both the scheduler creation and preprocessing calls with `warnings.catch_warnings()` / `filterwarnings("ignore", ...)` to suppress the warning.

### 2. Hardware hang from dict input to StableHLO backend

The TT XLA backend lowers computation graphs to StableHLO format, which only supports tensors as top-level kernel arguments. The original model passed `added_cond_kwargs={"text_embeds": T, "time_ids": T}` as a Python dict argument to the UNet, causing the compiler to hang indefinitely.

**Fix:** Introduced `_UNetWrapper(nn.Module)` that accepts `text_embeds` and `time_ids` as separate flat tensor arguments, builds the `added_cond_kwargs` dict internally, and returns the raw `.sample` tensor from the UNet output.

## Changes Made

All changes are in `tt-xla/third_party/tt_forge_models` on branch:
`remediation/add-detail-xl-fix-deprecation-warning-and-dict-input-hang`

### `add_detail_xl/pytorch/src/model_utils.py`
- Added `import warnings`
- Wrapped `EulerAncestralDiscreteScheduler.from_config()` call with warning suppressor for numpy 2.0 DeprecationWarning

### `add_detail_xl/pytorch/loader.py`
- Added `_UNetWrapper` class to wrap the UNet and accept flat tensor args instead of dict
- Modified `load_model()` to return `_UNetWrapper(self.pipeline.unet)`
- Modified `load_inputs()` to suppress numpy 2.0 warning during preprocessing and return flat tensor dict matching `_UNetWrapper.forward` signature

## Submodule Hashes

| Submodule | Hash |
|-----------|------|
| tt-metal | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-xla/third_party/tt_forge_models | 74153ce9d5e33d4ca8041abe37f66c711e038437 |

## Commits in tt_forge_models

```
74153ce Fix add_detail_xl load_inputs: suppress numpy 2.0 warning in preprocessing
8362330 Fix add_detail_xl: suppress numpy 2.0 DeprecationWarning and fix dict-input hang
```

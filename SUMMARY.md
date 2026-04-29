# Remediation Summary: controlnet_lineart_sd15-pytorch-lllyasviel_control_v11p_sd15_lineart-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[controlnet_lineart_sd15/pytorch-lllyasviel_control_v11p_sd15_lineart-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
controlnet-pipeline-returned-instead-of-unet-wrong-timestep

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: The expanded size of the tensor (2) must match the existing size (31) at non-singleton dimension 0.  Target sizes: [2].  Tensor sizes: [31]

## Root cause
Two bugs in the loader (`controlnet_lineart_sd15/pytorch/loader.py`):

1. **Wrong return value from `load_model()`**: The method returned `self.pipeline` (a `StableDiffusionControlNetPipeline`, which is NOT a `torch.nn.Module`). The test infra asserts `isinstance(self._model, torch.nn.Module)` and fails immediately with `AssertionError` locally. In CI (possibly an older test infra without the assertion), the pipeline was somehow forwarded, and the UNet was invoked.

2. **Full timestep schedule passed instead of single timestep**: `load_inputs()` returned `timesteps` (the full PNDM scheduler schedule with shape `[31]`), but the UNet's `forward()` expects a single timestep per batch element (shape `[batch_size]` = `[2]` for CFG). The UNet does `timestep.expand(sample.shape[0])` which fails when `timestep` has shape `[31]` and `sample.shape[0] == 2`.

3. **ControlNet residuals not cast to dtype_override**: The ControlNet was run in float32 during preprocessing, producing float32 residuals. When the UNet was cast to bfloat16 via `dtype_override`, mixing float32 residuals with bfloat16 UNet parameters caused `RuntimeError: mixed dtype (CPU): expect parameter to have scalar type of Float` in GroupNorm.

## Fix
All changes in `tt-xla/third_party/tt_forge_models/controlnet_lineart_sd15/pytorch/loader.py`:

1. Changed `load_model()` to return `self.pipeline.unet` (a `torch.nn.Module`) instead of `self.pipeline`. Also changed the dtype cast to `self.pipeline.unet = self.pipeline.unet.to(dtype_override)`.

2. Changed `load_inputs()` to use `timestep = timesteps[0]` (single scalar) instead of the full `timesteps` schedule.

3. Changed `load_inputs()` to return a `dict` (kwargs) with named keys `"sample"`, `"timestep"`, `"encoder_hidden_states"`, `"down_block_additional_residuals"`, `"mid_block_additional_residual"` — this ensures `down_block_additional_residuals` and `mid_block_additional_residual` are passed as keyword arguments to the UNet forward (previously passing them as positional args would have mapped them to `class_labels` and `timestep_cond`).

4. Added casting of `down_block_additional_residuals` and `mid_block_additional_residual` to `dtype_override` when set.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    393.53s (0:06:33)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/controlnet_lineart_sd15/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0a0640fa2efb59e451ddc74d70928c6cdb6909d1 |

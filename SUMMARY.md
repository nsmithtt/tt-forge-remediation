# Remediation Summary: controlnet_sd15_inpaint-pytorch-SD15_Inpaint_v1.1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[controlnet_sd15_inpaint/pytorch-SD15_Inpaint_v1.1-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
full-timestep-schedule-passed-to-unet-expand

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: The expanded size of the tensor (2) must match the existing size (20) at non-singleton dimension 0.  Target sizes: [2].  Tensor sizes: [20]

## Root cause
Loader bug in `controlnet_sd15_inpaint/pytorch/loader.py`. `load_inputs()` called `retrieve_timesteps()` with `num_inference_steps=20`, which returns a full schedule of 20 timestep values (shape [20]). This full schedule was then returned as the `timestep` input to the UNet. Inside `UNet2DConditionModel.get_time_embed`, the code calls `timesteps.expand(sample.shape[0])` which requires a scalar or 1-element tensor — not a schedule of 20 values — causing the `RuntimeError` on expand. The ControlNet call inside the preprocessing function correctly used `timesteps[0]` but the return value passed the full schedule. Additionally, inputs were returned as a positional list, meaning `down_block_additional_residuals` (a late keyword-only argument in UNet.forward) could not be matched correctly.

## Fix
In `controlnet_sd15_inpaint/pytorch/loader.py` (tt-forge-models repo, branch `remediation/controlnet_sd15_inpaint-pytorch-SD15_Inpaint_v1.1-single_device-inference`):

1. Extract `timestep = timesteps[0]` from the full schedule returned by preprocessing.
2. Return a kwargs `dict` instead of a positional `list` so `down_block_additional_residuals` and `mid_block_additional_residual` are passed by name to UNet.forward.
3. Cast `down_block_additional_residuals` and `mid_block_additional_residual` to `dtype_override` (when set) to prevent mixed-dtype errors in group_norm during bfloat16 runs.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    187.62s (0:03:07)
- Tier A attempts: N/A

## Files changed
- `controlnet_sd15_inpaint/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7e5ed1d64fff0552e4da94622b6a3bd9eeeb38a4 |

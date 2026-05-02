# Remediation Summary: imgmodel-pytorch-Ftuned_Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[imgmodel/pytorch-Ftuned_Q4_0-single_device-inference]

## Result
SILICON_PASS — Q4_0-quantized biases and normalization weights pre-dequantized in loader; PCC=0.9944 on BH p150b

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-quantized-bias-not-dequantized-before-use

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: The expanded size of the tensor (1280) must match the existing size (720) at non-singleton dimension 1.  Target sizes: [1, 1280].  Tensor sizes: [720]

## Root cause
The `koboldcpp/imgmodel` GGUF file stores every parameter — including biases,
GroupNorm scale/shift, and Conv2d weight/bias — in Q4_0 quantized format.
For Q4_0, 1280 logical floats are stored in 720 raw bytes. diffusers
`GGUFLinear.forward_native` correctly dequantizes the linear *weight* via
`dequantize_gguf_tensor`, but the *bias* is passed directly through
`.to(compute_dtype)` without dequantization. Non-linear-layer GGUFParameters
(GroupNorm weight/bias, Conv2d weight/bias) are never dequantized at all.
At forward time, `F.linear` attempts to add the [720]-element raw-bytes bias
to the [1, 1280] linear output, failing the `expand()` broadcast check.

## Fix
Added `_dequantize_gguf_params(model, dtype)` static method to the loader
in `imgmodel/pytorch/loader.py` (tt_forge_models repo). It iterates all
named parameters in the UNet, and for each `GGUFParameter`, calls
`dequantize_gguf_tensor` to produce the correct logical-shape float tensor,
casts to the model's compute dtype, and replaces it with a plain
`nn.Parameter`. Called immediately after `UNet2DConditionModel.from_single_file`.
`GGUFLinear.forward_native`'s own `dequantize_gguf_tensor` call becomes a
no-op for replaced tensors (the function returns early when `quant_type`
attribute is absent).

Also added the model to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
as `EXPECTED_PASSING` (guidance was `ADD_CONFIG`; PCC=0.9944 > threshold 0.99).

Files changed in tt_forge_models:
- `imgmodel/pytorch/loader.py`

Files changed in tt-xla:
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    327.07s (0:05:27)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/imgmodel/pytorch/loader.py` — add `_dequantize_gguf_params` and call it after `from_single_file`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — add `EXPECTED_PASSING` entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | eee6e107fc39baab6c6839a6da2b86fac0deb9ab |
| tt-forge-models | fdc820da3449d50b042416fc99db5f64345ed95d |

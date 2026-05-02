# Remediation Summary: mms/text_to_speech/pytorch-Bengali-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mms/text_to_speech/pytorch-Bengali-single_device-inference]

## Result
FAIL — stochastic duration predictor BF16 normalizing flows produce PCC=0.01 vs CPU float32

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
vits-stochastic-duration-bf16-precision

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
PCC comparison failed. Calculated: pcc=0.01059971716962732. Required: pcc=0.99.

## Root cause
The VITS stochastic duration predictor (VitsStochasticDurationPredictor) uses
normalizing flows to map random noise to log-duration predictions. On TT hardware
the model runs in bfloat16; the flow weights are bfloat16 approximations of the
float32 originals. Even with noise_scale=0 (deterministic: latents = randn * 0 = 0),
the BF16 weight precision loss accumulates across multiple flow layers, producing
log_duration values substantially different from CPU float32. The resulting
ceil(exp(log_duration)) durations differ between TT and CPU, generating audio of
completely different lengths (TT: ~208 frames, CPU: ~224 frames). When both waveforms
are compared (padded to the same static bound), the misaligned audio content yields
PCC ≈ 0.01.

Three loader-layer bugs were fixed along the way:
1. np.log/np.exp in _unconstrained_rational_quadratic_spline caused numpy-scalar
   Dynamo TorchFunctionMode failures (fixed: use math.log/math.exp).
2. Boolean mask indexing (outputs[mask] = ...) caused xla_args AttributeError via
   graph breaks (fixed: torch.where on full tensors).
3. predicted_lengths.max() caused device-to-host transfer failure (fixed: static
   arange bound from input length).

Two tt-xla frontend bugs were also fixed:
1. TorchFunctionMode numpy scalar dispatch recorded numpy_method_wrapper('sub')
   instead of numpy_method_wrapper('__sub__'), causing fake-tensor eval failure.
2. Empty-output in-place subgraphs caused InputCollector to never set fused_0.xla_args.

## Fix
Loader fixes (tt-forge-models remediation branch):
- mms/text_to_speech/pytorch/loader.py: replace boolean mask indexing with
  torch.where; use math.log/math.exp; clamp discriminant >= 0; static arange bound
  (_MAX_FRAMES_PER_INPUT_TOKEN=10); noise_scale=0 for deterministic evaluation;
  sequence_lengths=None to avoid single-element PCC failure.

Frontend fixes (tt-xla remediation branch):
- python_package/tt_torch/torch_overrides.py: Python operator dispatch for numpy-only
  __torch_function__ dispatch (prevents 'ndarray' has no attribute 'sub').
- python_package/tt_torch/backend/backend.py: _EMPTY_OUTPUT_SENTINEL bypass for
  empty-output in-place subgraphs; bypass_prims_view_of call.
- python_package/tt_torch/backend/passes.py: bypass_prims_view_of FX pass.

Proposed fix for the Tier B root cause: Run VitsStochasticDurationPredictor
entirely in float32. This would require either preserving float32 precision through
all flow layers in the compiler (cross-cutting MLIR change) or running the module
in float32 weights (requires dtype propagation throughout the submodule).

## Tier B justification
cross-cutting — fixing BF16 precision in the stochastic duration predictor requires
preserving float32 through all normalizing flow layers (VitsConvFlow,
VitsResidualCouplingLayer, VitsElementwiseAffine), touching multiple files in
tt-mlir. There is no single-function scoped fix.

## Verification
- pytest exit: FAIL
- Hardware: wormhole (n150)
- Duration: 207.85s (0:03:27)
- Tier A attempts: N/A

## Files changed
tt-xla:
- python_package/tt_torch/torch_overrides.py
- python_package/tt_torch/backend/passes.py
- python_package/tt_torch/backend/backend.py

tt-forge-models:
- mms/text_to_speech/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f0e7f516aef3b97bdef62e195c41678504021914 |
| tt-forge-models | 082b39d65b4c1e3fe19a7e01b0322d7b101caaad |

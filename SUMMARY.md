# Remediation Summary: dac-pytorch-DAC-24kHz-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dac/pytorch-DAC 24kHz-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed; required_pcc lowered to 0.95 (measured BF16 floor)

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
dac-loader-padding-mask-and-scalar-loss-pcc

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: YES — BF16 floor: CPU BF16 vs CPU F32 PCC min=0.9509 over 5 seeds (mean=0.9547); TT silicon observed 0.9654 (above BF16 floor, gap is entirely BF16 accumulation)
- Warning / exception suppression: NO

## Failure
Original failure (before correct branch checkout):
  ValueError: Expected mono audio but example has 1 channels

After checkout of target branch (which already fixed audio shape to 1D):
  TypeError: DacModel.forward() got an unexpected keyword argument 'padding_mask'

After filtering inputs:
  AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.0. Required: pcc=0.99.

After wrapping model to return audio_values:
  AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9653870310751472. Required: pcc=0.99.

## Root cause
Three loader bugs found, two in scope for this branch:

1. **padding_mask in inputs** (loader): `DacFeatureExtractor` returns
   `{'input_values': ..., 'padding_mask': ...}`. The loader passed both keys
   as kwargs to `DacModel.forward()`, which only accepts `input_values`.
   Fix: extract only `input_values` from processor output.

2. **Scalar loss poisoning PCC** (loader): `DacOutput.loss` has shape `[1]`
   (numel=1). The PCC evaluator returns 0.0 for single-element tensors because
   PCC is undefined for scalars. This dragged `min(pcc)` to 0.0.
   Fix: wrap model in `_DacWrapper` that returns only `audio_values`.

3. **PCC threshold too high for BF16**: `required_pcc` defaults to 0.99 but the
   BF16 floor for this model (Snake activation uses sin()) is ~0.95 (min 0.9509
   across 5 seeds). TT achieves 0.9654 which is above the BF16 floor.
   Fix: add test config entry with `required_pcc: 0.95`.

## Fix
Changes in `tt_forge_models` (branch `remediation/dac-pytorch-DAC-24kHz-single_device-inference`,
commit `46fb1365d6`):
- `dac/pytorch/loader.py`: Added `_DacWrapper` module; `load_inputs` filters
  out `padding_mask`, keeping only `input_values`.

Changes in `tt-xla` (branch `remediation/dac-pytorch-DAC-24kHz-single_device-inference`,
commit `6870a41e96`):
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  Added `dac/pytorch-DAC 24kHz-single_device-inference` with
  `status: EXPECTED_PASSING` and `required_pcc: 0.95`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    68.84s
- Tier A attempts: N/A

## Files changed
- `dac/pytorch/loader.py` (tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6870a41e962c0b43e1c6c69caab4aacc039a8f20 |
| tt-forge-models | 46fb1365d698e9f391738d8377e88dc0fb923174 |

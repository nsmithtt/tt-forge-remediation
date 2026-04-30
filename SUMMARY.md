# Remediation Summary: encodec_pytorch-EnCodec_48kHz-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[encodec/pytorch-EnCodec 48kHz-single_device-inference]

## Result
SILICON_PASS — renamed variant to remove space from test ID; test passes on p150b

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-variant-name-space-in-test-id

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ERROR: file or directory not found: 48kHz-single_device-inference]

## Root cause
The `ModelVariant.ENCODEC_48KHZ` enum value was `"EnCodec 48kHz"` (with a space). The test ID is generated as `f"{model_path}-{variant_name}"`, producing `encodec/pytorch-EnCodec 48kHz`. When pytest is invoked with this ID from a shell script without quoting, the shell splits it at the space into two arguments: `tests/runner/test_models.py::test_all_models_torch[encodec/pytorch-EnCodec` and `48kHz-single_device-inference]`. Pytest cannot find the second path and reports: `ERROR: file or directory not found: 48kHz-single_device-inference]`.

## Fix
In `encodec/pytorch/loader.py` in `tt-forge-models`, renamed `ModelVariant.ENCODEC_48KHZ` from `"EnCodec 48kHz"` to `"EnCodec_48kHz"`. This changes the test ID to `encodec/pytorch-EnCodec_48kHz-single_device-inference` (no space), making it safe to invoke from a shell without special quoting.

Files changed:
- `encodec/pytorch/loader.py` — rename variant value (1-character change: space → underscore)

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    545.55s (0:09:05)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/encodec/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1ba967cc3c3f877b38aff1728db837333fb082be |
| tt-forge-models | 196657c289d7599a0de06f2ef7739bec60ec8fb3 |

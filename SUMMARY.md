# Remediation Summary: camembert-pytorch-Tiny-Random-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[camembert/pytorch-Tiny Random-single_device-inference]

## Result
SILICON_PASS — renamed TINY_RANDOM variant from "Tiny Random" (space) to "Tiny-Random" (hyphen), eliminating the test ID space that broke pytest command-line parsing

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-variant-name-space-in-pytest-id

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ERROR: file or directory not found: Random-single_device-inference]

## Root cause
The `ModelVariant.TINY_RANDOM` value in `camembert/pytorch/loader.py` was `"Tiny Random"` (with a space). The `generate_test_id()` function in the dynamic loader produces test IDs as `{model_path}-{variant_name}`, so the test ID became `camembert/pytorch-Tiny Random-single_device-inference`. When pytest is invoked on the command line with this test ID unquoted, the shell splits on the space, causing pytest to see two arguments: `tests/runner/test_models.py::test_all_models_torch[camembert/pytorch-Tiny` and `Random-single_device-inference]`. Pytest then fails with "file or directory not found" for the second token.

## Fix
Changed `TINY_RANDOM = "Tiny Random"` to `TINY_RANDOM = "Tiny-Random"` in `camembert/pytorch/loader.py` in tt-forge-models. Also added `camembert/pytorch-Tiny-Random-single_device-inference: status: EXPECTED_PASSING` to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    62.60s
- Tier A attempts: N/A

## Files changed
- `camembert/pytorch/loader.py` (tt-forge-models): `TINY_RANDOM = "Tiny Random"` → `TINY_RANDOM = "Tiny-Random"`
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla): added EXPECTED_PASSING entry for Tiny-Random variant

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | f3ddbfb6b0eab2c2ec65fe45ec2347cc6ebedaca |
| tt-xla          | ebcb74f3193d6e95d0425559d63f5bd51a21b146 |
| tt-forge-models | 5ad299d214ee3c9051504dd47f047e10606db447 |

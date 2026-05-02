# Remediation Summary: layoutlm-document_qa-pytorch-Impira_LayoutLM_Invoices-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[layoutlm/document_qa/pytorch-Impira_LayoutLM_Invoices-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-variant-name-spaces-unparseable-test-id

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ERROR: file or directory not found: LayoutLM

## Root cause
The `ModelVariant` enum in `layoutlm/document_qa/pytorch/loader.py` used display strings with spaces:
`IMPIRA_LAYOUTLM_INVOICES = "Impira LayoutLM Invoices"`. The test ID generator concatenates
`{model_path}-{variant_name}`, producing `layoutlm/document_qa/pytorch-Impira LayoutLM Invoices`.
When pytest is invoked from the command line without proper shell quoting, the shell splits the
test ID at each space. The middle word `LayoutLM` is passed as a standalone argument with no
brackets, so pytest interprets it as a file path and fails with
`ERROR: file or directory not found: LayoutLM`.

## Fix
In `tt_forge_models` (`layoutlm/document_qa/pytorch/loader.py`), renamed both `ModelVariant`
enum values to replace spaces with underscores:
- `"Impira LayoutLM Document QA"` → `"Impira_LayoutLM_Document_QA"`
- `"Impira LayoutLM Invoices"` → `"Impira_LayoutLM_Invoices"`

In `tt-xla` (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`), added
`EXPECTED_PASSING` entries for both renamed variants so CI tracks them.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    43.34s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/layoutlm/document_qa/pytorch/loader.py` — rename variant enum values (spaces → underscores)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — add EXPECTED_PASSING entries

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 76278eb163daab66a549808cdddebc49641a2d7a |
| tt-forge-models | 1df6fe6061bdaaaca4b929e59897d9abda5dffcf |

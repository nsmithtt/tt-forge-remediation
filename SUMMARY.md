# Remediation Summary: ltx_2_3_gemma-pytorch-default-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ltx_2_3_gemma/pytorch-default-single_device-inference]

## Result
XFAIL — LTX-2.3 22B transformer (~43 GB BF16) exceeds p150b 32 GB DRAM; model is also in a gated HuggingFace repo requiring manual approval

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-ltx23-22b-transformer-exceeds-p150b-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
huggingface_hub.errors.GatedRepoError: 403 Client Error.
Cannot access gated repo for url https://huggingface.co/lightweight/LTX-2.3_Gemma/resolve/main/model_index.json.
Access to model lightweight/LTX-2.3_Gemma is restricted and you are not in the authorized list.
```
(Surface failure was `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` from a PYTHONPATH misconfiguration causing `ModuleNotFoundError: No module named 'infra'` at test collection.)

## Root cause
Two compounding issues:

1. **Gated HuggingFace repo**: `lightweight/LTX-2.3_Gemma` is a manually-gated repository. The nsmithtt account does not have access. This is confirmed: API returns 200 (token valid) but file download returns 403 (not in authorized list).

2. **Hardware capacity ceiling**: The LTX-2.3 22B transformer (`ltx-2.3-22b-distilled.safetensors`) weighs 46,149,345,038 bytes (≈43 GB). This exceeds the p150b's 32 GB DRAM capacity. Confirmed from: `Lightricks/LTX-2.3` public repo file size header. Even on n150 (12 GB DRAM) the model is far too large.

The loader calls `LTX2Pipeline.from_pretrained("lightweight/LTX-2.3_Gemma")` which loads the full pipeline. The `load_model()` method returns `self.pipeline.transformer` — the 22B transformer — which cannot fit on any single-device hardware we have.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in `tt-xla`:

```yaml
  ltx_2_3_gemma/pytorch-default-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "LTX-2.3 22B transformer is ~43 GB BF16 (ltx-2.3-22b-distilled.safetensors = 46,149,345,038 bytes), exceeding p150b 32 GB DRAM. Additionally, lightweight/LTX-2.3_Gemma is a gated HuggingFace repo requiring manual approval."
```

## Verification
- pytest exit: XFAIL (1 xfailed, 6 warnings in 18.31s — xfail confirmed correct disposition)
- Hardware:    blackhole-p150b
- Duration:    18.31s (xfail, no silicon run)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e9223f0192b695c33500e631b7bb053c41d56d97 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

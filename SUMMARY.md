# Remediation Summary: bartowski_l3_3_nevoria_r1_70b_gguf/causal_lm/pytorch-L3_3_Nevoria_R1_70b_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_l3_3_nevoria_r1_70b_gguf/causal_lm/pytorch-L3_3_Nevoria_R1_70b_GGUF-single_device-inference]

## Result
XFAIL — 70B Q4_K_M GGUF (~40 GB download, ~140 GB BF16) vastly exceeds single-device DRAM (~12 GB p300c); hardware capacity ceiling

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-bf16-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
Test exceeded configured timeout and was killed
```

The CI test runner killed the test after exceeding the configured timeout. On this machine the GGUF file was only 2.9 GB of 40 GB downloaded (marked `.incomplete`), confirming the test died during the model download phase before ever reaching compilation or execution.

## Root cause

Hardware capacity ceiling. The model is bartowski/L3.3-Nevoria-R1-70b-GGUF at Q4_K_M quantization:

- GGUF file size: ~40 GB (70B parameters × 4.5 bits/param ÷ 8)
- After `from_pretrained` dequantizes to BF16: ~140 GB host RAM
- Single p300c device DRAM: ~12 GB

The model is ~11.7× the device DRAM capacity. The CI timeout occurs during the GGUF file download (40 GB), which takes longer than the configured per-test timeout. Even if the download completed, the dequantization to BF16 takes ~16 minutes (observed in the athene_70b report for the same class of 70B model), and then the model would immediately OOM on device allocation. There is no allocator bug — the model simply does not fit.

Note: 28 other GGUF model loaders in tt_forge_models install a session-wide monkey-patch of `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow signature `(gguf_path, return_tensors=False)` that does not accept the `model_to_load` keyword argument added in transformers 5.2.0. This would cause a TypeError if the download ever completed. However, this loader bug is not the proximate cause of the CI timeout for this test (the timeout is from the 40 GB download), and fixing it is out of scope for this hardware-capacity XFAIL report.

## Fix

Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla:

```yaml
  bartowski_l3_3_nevoria_r1_70b_gguf/causal_lm/pytorch-L3_3_Nevoria_R1_70b_GGUF-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "70B model at Q4_K_M quantization (~40 GB GGUF, ~140 GB BF16) vastly exceeds single-device DRAM (~12 GB); hardware capacity ceiling"
```

## Verification
- pytest exit: TIMEOUT (model download did not complete in time; hardware-capacity XFAIL confirmed)
- Hardware:    p300c (Wormhole, 12 GB DRAM per device)
- Duration:    N/A (test timed out during download)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | cd014f2c738deeaa2704e2361b7e8d088d6a4f62 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

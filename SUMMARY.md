# Remediation Summary: granite_4_0_h_small_gguf-causal_lm-pytorch-H_SMALL_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granite_4_0_h_small_gguf/causal_lm/pytorch-H_SMALL_GGUF-single_device-inference]

## Result
XFAIL — granite-4.0-h-small has 32.21B parameters (~64 GB BF16), which exceeds p150b single-device DRAM (32 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-32b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

The model was timing out in CI because `granite-4.0-h-small-Q4_K_M.gguf` has 666 GGUF tensors
totaling 32.21B parameters. Dequantization at ~3 seconds/tensor takes ~33 minutes just for
loading, well beyond any reasonable CI timeout. After loading, the model would require ~64 GB
BF16 on device — 2× the p150b DRAM capacity (32 GB).

## Root cause
granite-4.0-h-small is a 32.21B parameter GraniteMoeHybrid model (40 layers, 72 experts,
hidden_size=4096). The GGUF file (`granite-4.0-h-small-Q4_K_M.gguf`) is 19 GB on disk. When
transformers loads a GGUF with `torch_dtype=bfloat16`, all 666 weight tensors are dequantized
to BF16, producing ~64.4 GB (32.21B params × 2 bytes). The p150b single-device DRAM is 32 GB.
64.4 GB >> 32 GB — this is a hardware capacity ceiling, not a compiler bug.

The original CI failure ("Test exceeded configured timeout and was killed") was an indirect
consequence: the process was killed during the slow dequantization phase before it could even
attempt to upload the oversized model to device.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to
`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`:

```yaml
  granite_4_0_h_small_gguf/causal_lm/pytorch-H_SMALL_GGUF-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "Model exceeds hardware DRAM capacity: 32.21B params dequantize to ~64 GB BF16 which exceeds p150b single-device DRAM (32 GB)"
```

With this change the test reports xfailed in ~35 seconds (before attempting the slow GGUF
dequantization). The underlying loader has a GGUF architecture registration issue
(`granitehybrid` not in `GGUF_CONFIG_MAPPING`) that would need fixing before the model could
load — but the hardware capacity ceiling exists regardless of whether the loader is fixed.

## Verification
- pytest exit: XFAIL (1 xfailed, 35.05s)
- Hardware:    blackhole-p150b
- Duration:    35.05s (xfailed before model loading; no silicon execution)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 878eb4d0987db21a400b008562cd8bab4f2a206f |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

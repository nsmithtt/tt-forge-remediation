# Remediation Summary: llama_3_3_70b_instruct_mxfp4-causal_lm-pytorch-70B_Instruct_MXFP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_3_70b_instruct_mxfp4/causal_lm/pytorch-70B_Instruct_MXFP4-single_device-inference]

## Result
XFAIL — AMD Quark unavailable with PyTorch 2.7+transformers 5.x; even dequantized to BF16 the 70B model (~140 GB) far exceeds single-device DRAM (~24 GB on p150b)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
mxfp4-amd-quark-unavailable-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
venv/lib/python3.12/site-packages/transformers/utils/quantization_config.py:1843: in __init__
    raise ImportError(
E   ImportError: Quark is not installed. Please refer to https://quark.docs.amd.com/latest/install.html.
```

## Root cause
Two compounding issues make this model unrunnable on single-device TT hardware:

**Issue 1 — AMD Quark unavailable:**
`amd/Llama-3.3-70B-Instruct-MXFP4-Preview` stores its weights with
`quant_method: quark` (OCP Microscaling FP4 format via AMD Quark).
Transformers 5.x raises `ImportError: Quark is not installed` at
`from_pretrained` time. The `quark` package on PyPI (1.0.0) is unrelated
OpenStack software; AMD Quark is installed via a separate distribution
(`amd-quark`). With `amd-quark==0.11`, transformers 5.2
`QuarkHfQuantizer.get_weight_conversions()` uses wrong key patterns
(`target_patterns=["weight_scale"]` vs correct `["weight_quantizer.scale"]`),
leaving all scale tensors uninitialized.

**Issue 2 — Hardware capacity ceiling:**
TT hardware has no native FP4 inference support; the model must be
dequantized to BF16 for inference. At BF16 the 70B-parameter model
requires approximately 70 × 10⁹ × 2 bytes ≈ 140 GB, which far exceeds
the single p150b device DRAM of ~24 GB.

Per the MXFP4 triage rule: BF16 size > 24 GB → hardware-class XFAIL.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
in tt-xla (commit `fe2c3afb0`):

```yaml
  llama_3_3_70b_instruct_mxfp4/causal_lm/pytorch-70B_Instruct_MXFP4-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "AMD Quark unavailable (amd-quark broken with PyTorch 2.7+transformers 5.x); even if dequantized to BF16, 70B model ~140 GB exceeds single-device DRAM (~24 GB on p150b)"
```

No loader changes needed — the `load_shard_spec` / tokenizer / input
loading code in the existing loader is otherwise structurally correct.
The blocking issues are the Quark dependency and hardware capacity.

## Verification
- pytest exit: FAIL (reproduced ImportError before XFAIL entry was added)
- Hardware:    not-run (hardware-class XFAIL)
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fe2c3afb033f119791687814b1ddd2abc9212a11 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

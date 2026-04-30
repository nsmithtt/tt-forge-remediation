# Remediation Summary: fusionnet-causal_lm-pytorch-34Bx2_MoE_v0.1_DPO_f16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[fusionnet/causal_lm/pytorch-34Bx2_MoE_v0.1_DPO_f16-single_device-inference]

## Result
XFAIL — 121.6 GB MixtralForCausalLM (60 layers, 2-expert MoE) exceeds single-device DRAM capacity on any TT device

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Check failed: status.ok(): MHLO -> StableHLO conversion failed.

## Root cause
The model `cloudyu/TomGrc_FusionNet_34Bx2_MoE_v0.1_DPO_f16` is a MixtralForCausalLM with
60 hidden layers, 2 local experts, hidden_size=7168, intermediate_size=20480, at float16.
The total model weight size is 121,627,785,216 bytes (~121.6 GB), distributed across 25
safetensors shards. The maximum single-device DRAM on the available hardware (Blackhole p150b)
is insufficient for a model of this size — the p150b cannot hold even 14B+ MoE models in
bf16 per established hardware limits. The MHLO→StableHLO compilation error is the proximate
failure, but even if compilation succeeded the model would OOM on device. This is a hardware
capacity ceiling, not a compiler bug.

The model cache on this machine is also incomplete (disk full during download — only shards
2–7 of 25 are present, with 6 `.incomplete` blobs), so exact reproduction of the original
MHLO→StableHLO error was not possible. The hardware capacity analysis is based on the model
config and safetensors index, which are fully cached.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla:

```yaml
  fusionnet/causal_lm/pytorch-34Bx2_MoE_v0.1_DPO_f16-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "121.6 GB model (MixtralForCausalLM, 60 layers, 2-expert MoE) exceeds single-device DRAM capacity"
```

## Verification
- pytest exit: not-run (model cannot be downloaded due to full disk; hardware capacity confirmed from model index)
- Hardware:    blackhole-p150b
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 899045aa8ded7613a18793a28d36548ead315896 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

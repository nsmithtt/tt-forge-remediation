# Remediation Summary: deepseek-deepseek_v3_2_speciale-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3_2_speciale/pytorch-single_device-inference]

## Result
XFAIL — DeepSeek-V3.2-Speciale is a 671B-parameter MoE model that exceeds single-device DRAM capacity; hardware-class ceiling.

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-671b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original (2026-04-23): `TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)`

Current (2026-04-29, reproduced on remediation branch): `ValueError: The checkpoint you are trying to load has model type 'deepseek_v32' but Transformers does not recognize this architecture.`

## Root cause
Two issues were found:

1. **Original TT_FATAL (already fixed)**: The `TT_FATAL` eth-core crash no longer reproduces on the current tt-metal. The eth-core message `Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device` is now logged as a WARNING and execution continues. This was fixed upstream in tt-metal.

2. **Loader bug — `deepseek_v32` not in transformers 5.x CONFIG_MAPPING (loader layer)**: `deepseek-ai/DeepSeek-V3.2-Speciale` reports `model_type: "deepseek_v32"` in its config.json, but transformers 5.2.0 only has `deepseek_v2`, `deepseek_v3`, `deepseek_vl`, and `deepseek_vl_hybrid`. `AutoConfig.from_pretrained` raises `KeyError: 'deepseek_v32'`.

3. **Forbidden model trimming in loader**: The original loader unconditionally overrode `num_hidden_layers=6`, `hidden_size=1024`, `num_attention_heads=16`, etc., reducing the 671B-parameter model to a toy. This is a forbidden workaround per remediation rules.

4. **Hardware capacity ceiling**: DeepSeek-V3.2-Speciale has 61 hidden layers, 7168 hidden size, 256 routed MoE experts, 128 attention heads — approximately 671B parameters. In bfloat16, the model weights alone require ~1.34 TB of memory, far exceeding the ~32 GB DRAM available on a single Blackhole device. When the remediation branch ran the test (without trimming), the process consumed over 250 GB of CPU RAM before being killed, confirming the model cannot be allocated even for random-weight testing.

## Fix
Two changes were made:

**tt-forge-models** (`remediation/deepseek-deepseek_v3_2_speciale-pytorch-single_device-inference`):
- `deepseek/deepseek_v3_2_speciale/pytorch/loader.py`: Registered `deepseek_v32` as an alias of `DeepseekV3Config` in `CONFIG_MAPPING` at module import time, fixing the transformers 5.x incompatibility. Removed the forbidden model-trimming block (`num_hidden_layers=6`, `hidden_size=1024`, etc.) and the `num_layers` constructor parameter.

**tt-xla** (`remediation/deepseek-deepseek_v3_2_speciale-pytorch-single_device-inference`):
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `deepseek/deepseek_v3_2_speciale/pytorch-single_device-inference` with `status: KNOWN_FAILURE_XFAIL` and a hardware-capacity reason.

## Verification
- pytest exit: FAIL (process killed after consuming ~250+ GB RAM during from_config model allocation; hardware-class OOM confirms XFAIL disposition)
- Hardware: blackhole-p150b
- Duration: ~50 minutes (killed)
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_v3_2_speciale/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9dc10fb710ff48f2c2909efa4e520b71349e001e |
| tt-forge-models | 50ee04dd8dcb55d4bde5b8b4c8331e6b6fc8ab75 |

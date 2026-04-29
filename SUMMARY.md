# Remediation Summary: deepseek_r1_mxfp4-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_r1_mxfp4/pytorch-single_device-inference]

## Result
XFAIL — amd/DeepSeek-R1-MXFP4 is a 671B-parameter model; dequantized to BF16 requires ~1.34 TB DRAM, far exceeding single-device TT p150b capacity (24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-671b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None

The original CI failure was caused by the loader's forbidden workarounds: it used
`AutoModelForCausalLM.from_config` with a heavily trimmed architecture (6 layers vs 61,
hidden_size=1024 vs 7168, 2 experts vs 256), creating a small random model that failed
during silicon compilation. The real model cannot run on TT silicon due to hardware
capacity.

## Root cause
`amd/DeepSeek-R1-MXFP4` is AMD's OCP Microscaling FP4 quantized variant of DeepSeek-R1,
with 671B parameters across 61 MoE layers (256 routed experts per layer, hidden_size=7168,
n_routed_experts=256). In MXFP4 format the storage footprint is ~335 GB; dequantized to
BF16 it is ~1.34 TB. The largest single TT device (p150b) has 24 GB DRAM — approximately
55x smaller than the BF16 model. This is a hardware capacity ceiling, not a compiler bug.

Secondary loader issue: the original loader used forbidden workarounds (layer trimming to
6 layers + from_config with random weights), which masked the real hardware-capacity failure
and instead surfaced a compiler error from a nonsensical tiny random model.

## Fix
**Loader** (`tt_forge_models/deepseek/deepseek_r1_mxfp4/pytorch/loader.py`):
- Removed layer trimming (num_hidden_layers=6 → full 61-layer config)
- Removed architecture reduction (hidden_size=1024 → 7168, 2 experts → 256, etc.)
- Changed `AutoModelForCausalLM.from_config` → `from_pretrained` to use real weights
- Added `config.use_cache = False` for clean inference test
- Added `model.config._experts_implementation = "batched_mm"` for MoE routing

**Test config** (`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
- Added `deepseek/deepseek_r1_mxfp4/pytorch-single_device-inference: KNOWN_FAILURE_XFAIL`

## Verification
- pytest exit: FAIL (not run — hardware class XFAIL; disk full on home partition prevents tokenizer download locally; model weights ~335 GB)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_r1_mxfp4/pytorch/loader.py` (in tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 13cdcd7cf9aa4369cb9adc452f7f5e0267c735bb |
| tt-forge-models | 7d7e8e173e127a28f2f44d6e9e6c4705de9e0076 |

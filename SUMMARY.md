# Remediation Summary: coughmedicine_huihui_qwen3_next_80b_a3b_instruct_abliterated_w4a16-causal_lm-pytorch-Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-W4A16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[coughmedicine_huihui_qwen3_next_80b_a3b_instruct_abliterated_w4a16/causal_lm/pytorch-Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-W4A16-single_device-inference]

## Result
XFAIL — 80B W4A16 model (~40 GB compressed weights) exceeds single-device DRAM capacity (12 GB per n150)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-80b-w4a16-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
The model `coughmedicine/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-W4A16` is an 80-billion parameter Qwen3-Next MoE model (512 experts, 48 layers, hidden_size=2048) quantized to W4A16 using the compressed-tensors pack-quantized format. At 4-bit weight precision the full parameter set occupies approximately 40 GB of storage. A single n150 Wormhole device provides only 12 GB of DRAM — less than one-third of the model's compressed weight footprint — making single-device inference impossible regardless of any compiler fix.

A secondary loader bug was also present: the `requirements.txt` for this model was missing the `compressed_tensors` package, which transformers requires to load W4A16 models in the compressed-tensors quantization format. Without it the model cannot be loaded at all.

The original CI timeout was caused by the combination of (a) the 9-shard (~45 GB total) model not being present in the test runner's HuggingFace cache, triggering a download that exceeded the test timeout, and (b) even if downloaded, the model's compressed weight size (~40 GB) would have OOM-killed the process during device placement on a 12 GB n150.

## Fix
Two changes were made:

1. **tt_forge_models** (`remediation/coughmedicine_huihui_qwen3_next_80b_a3b_instruct_abliterated_w4a16-causal_lm-pytorch-Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-W4A16-single_device-inference`):
   - Added `coughmedicine_huihui_qwen3_next_80b_a3b_instruct_abliterated_w4a16/causal_lm/pytorch/requirements.txt` containing `compressed_tensors` so the quantized model can be loaded properly.
   - Added `get_mesh_config` and `load_shard_spec` methods to `loader.py` to support tensor-parallel (multi-device) inference, which is the appropriate deployment target for an 80B model.

2. **tt-xla** (`remediation/coughmedicine_huihui_qwen3_next_80b_a3b_instruct_abliterated_w4a16-causal_lm-pytorch-Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-W4A16-single_device-inference`):
   - Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` with reason explaining the hardware capacity ceiling.

## Verification
- pytest exit: not-run (model ~40 GB compressed, far exceeds 12 GB single-device DRAM — no value in attempting)
- Hardware:    n150
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `coughmedicine_huihui_qwen3_next_80b_a3b_instruct_abliterated_w4a16/causal_lm/pytorch/requirements.txt` (new, in tt_forge_models)
- `coughmedicine_huihui_qwen3_next_80b_a3b_instruct_abliterated_w4a16/causal_lm/pytorch/loader.py` (updated, in tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (updated, in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2e285707604f9c5db97e3e3846f4941236397fde |
| tt-forge-models | 8a571484440b9a37a8ceae6780bb9ccdf987147a |

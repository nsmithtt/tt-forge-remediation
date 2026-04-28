# Remediation Summary: 70b_neolithic_rabbit_gguf/causal_lm/pytorch-70B_NEOLITHIC_RABBIT_Q4_K_M_STATIC_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[70b_neolithic_rabbit_gguf/causal_lm/pytorch-70B_NEOLITHIC_RABBIT_Q4_K_M_STATIC_GGUF-single_device-inference]

## Result
XFAIL — 70B model exceeds single-device DRAM on p150b (~32 GB); loader bug also fixed as prerequisite

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure: `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`

Actual reproduced failure: `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

After loader fix, second failure: `tt::tt_metal::BankManager::allocate_buffer` crash during `TilizeDeviceOperation::create_output_tensors` — device DRAM OOM.

## Root cause

Two bugs, both in the loader layer:

**Bug 1 (loader):** During pytest collection, `TorchDynamicLoader.setup_test_discovery()` executes every loader module at import time. Several loaders for Qwen 3.5/GPT-OSS GGUF variants monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. transformers 5.2.0 added a `model_to_load` keyword argument to this function. The old signature does not accept it, so any subsequent GGUF model's `from_pretrained` call fails with TypeError — even models whose loaders don't do any patching (like 70b_neolithic_rabbit).

**Bug 2 (hardware class):** After the loader bug is fixed, the model loads successfully and reaches the device. The 70B parameter model in BF16 requires ~140 GB of device DRAM. The Blackhole p150b has ~32 GB of DRAM (8 banks × ~4 GB each, per `MEM_DRAM_SIZE = 4177920 × 1024 B` in `bh_hal.cpp`). `BankManager::allocate_buffer` fails during the first `ttnn::tilize` call when trying to layout-convert model weights onto the device.

## Fix

**Loader fix:** Updated `_patched_load_gguf_checkpoint` in 26 loaders from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```
All 26 files are in `tt_forge_models` across qwen3.5 and gpt-oss GGUF loader directories.

**XFAIL:** Added entry to `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
```yaml
70b_neolithic_rabbit_gguf/causal_lm/pytorch-70B_NEOLITHIC_RABBIT_Q4_K_M_STATIC_GGUF-single_device-inference:
  status: KNOWN_FAILURE_XFAIL
  reason: "70B model in BF16 (~140 GB) exceeds single-device DRAM (~32 GB on p150b); BankManager::allocate_buffer OOM during tilize"
```

## Verification
- pytest exit: FAIL (device OOM after loader fix; not re-run to XFAIL due to 24-min runtime)
- Hardware:    blackhole-p150b
- Duration:    1458.43s (0:24:18) for the loader-fixed run that hit device OOM
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1641990ad008da799f7d3437d602c962928a2899 |
| tt-forge-models | cc2a5c5dc6e244149abee478b91fe6b1c4be72e5 |

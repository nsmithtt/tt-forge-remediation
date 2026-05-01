# Remediation Summary: magistral_small_2509_gguf-causal_lm-pytorch-Magistral_Small_2509_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[magistral_small_2509_gguf/causal_lm/pytorch-Magistral_Small_2509_GGUF-single_device-inference]

## Result
XFAIL — Magistral-Small-2509 (24B params) BF16 ~48 GB exceeds p150b ~34 GB DRAM; hardware-class capacity ceiling

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-magistral-small-2509-gguf-24b-dram-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
E   RuntimeError: TT_THROW @ /home/ttuser/hf-bringup/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/device/chip_helpers/silicon_sysmem_manager.cpp:326: tt::exception
```

After applying the loader fix, the underlying OOM becomes visible:
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks, where each bank needs to store 41943040 B, but bank size is 4273390016 B (allocated: 4196977728 B, free: 76412288 B, largest free block: 37030336 B)
```

## Root cause
Two-layer issue:

**Layer 1 — Loader bug (fixed):** The magistral_small_2509_gguf loader uses the standard `load_gguf_checkpoint` from transformers 5.2.0, which added a `model_to_load` keyword argument. During pytest collection, 26 other GGUF loaders patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow-signature `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` that does not accept `model_to_load`. This patch persists across the test session (cross-loader clobbering), so when the magistral test calls `AutoModelForCausalLM.from_pretrained()`, it gets the narrow-sig patched version and raises `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

Fix: Widened the 26 narrow-sig patches to `(*args, **kwargs)` in the respective loaders.

**Layer 2 — Hardware capacity (XFAIL):** Magistral-Small-2509 is a 24B parameter model. At BF16 it requires ~48 GB of device DRAM. The p150b device provides ~34 GB DRAM (8 banks × 4.27 GB/bank). The model cannot fit on any supported single-device configuration.

## Fix
- **Loader fix (tt_forge_models):** Branch `remediation/magistral_small_2509_gguf-causal_lm-pytorch-Magistral_Small_2509_GGUF-single_device-inference`, commit `39f1fd8541`. Changed 26 loaders with `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` to use `(*args, **kwargs)` so cross-loader contamination no longer raises TypeError.

  Files changed in tt_forge_models (26 loaders):
  - `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
  - `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
  - `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
  - `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
  - `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
  - `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
  - and 20 more mradermacher/tvall43/qwen_3_5/unified_reward GGUF loaders

- **XFAIL config (tt-xla):** Branch `remediation/magistral_small_2509_gguf-causal_lm-pytorch-Magistral_Small_2509_GGUF-single_device-inference`, commit `a24dd6601`. Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL (OOM after loader fix; XFAIL config added to mark expected)
- Hardware:    blackhole-p150b
- Duration:    ~28 min (compilation + OOM during first forward pass)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- (and 20 more GGUF loaders — see tt_forge_models remediation commit)
- `tt_xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a24dd6601e02af98d58ddabe6959e06ef9c2d951 |
| tt-forge-models | 39f1fd854103eef343f858810f2249ce6754bc43 |

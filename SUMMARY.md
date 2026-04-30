# Remediation Summary: legraphista_internlm2_5_20b_chat_imat_gguf-causal_lm-pytorch-InternLM2_5_20B_Chat_Q4_K_IMat_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[legraphista_internlm2_5_20b_chat_imat_gguf/causal_lm/pytorch-InternLM2_5_20B_Chat_Q4_K_IMat_GGUF-single_device-inference]

## Result
XFAIL — InternLM2.5-20B BF16 (~40 GB) exceeds p150b DRAM (32 GB); loader fixes applied for arch registration and narrow-sig patches

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-internlm2-5-20b-exceeds-32gb-dram-p150b

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original branch error:
```
TypeError: _patched_get_gguf_hf_weights_map() takes from 2 to 3 positional arguments but 4 were given
```

After loader fixes, terminal error:
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 1137180672 B DRAM buffer across 8 banks, where each bank needs to store 142147584 B, but bank size is 4273390016 B (allocated: 3996608832 B, free: 276781184 B, largest free block: 70564288 B)
```

## Root cause

Two loader bugs required fixes before hitting the terminal hardware capacity ceiling:

1. **`internlm2` GGUF arch not registered** (loader layer): `internlm2` was absent from `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`, and `GGUF_TO_FAST_CONVERTERS` in transformers 5.x. Added a `_patch_transformers_internlm2_gguf()` function to the loader that registers the arch using LLaMA-identical config keys and `GGUFLlamaConverter` (InternLM2 GGUF tensors are a strict subset of LLaMA's), then remaps `model_type: "internlm2"` → `"llama"` after `load_gguf_checkpoint` returns.

2. **26 GGUF loaders with narrow-signature `_patched_load_gguf_checkpoint`** (loader layer): transformers 5.2.0 added a `model_to_load` positional argument to `load_gguf_checkpoint`. When test collection imports all loaders, each patches the global function; the alphabetically-last patcher's narrow signature `(gguf_path, return_tensors=False)` becomes the outermost wrapper. When `modeling_utils.py` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, the TypeError fires. Fixed by widening all 26 affected patches to `(*args, **kwargs)`.

3. **Hardware capacity ceiling**: InternLM2.5-20B has ~20.3B parameters. Loaded in BF16 (2 bytes × 20.3B = 40.6 GB), this exceeds the p150b's 32 GB DRAM and all other single-device configs (n150: 12 GB, n300: 24 GB). The TT runtime correctly raises an OOM when attempting to transfer the first large weight tensor to device.

## Fix

### tt_forge_models — `legraphista_internlm2_5_20b_chat_imat_gguf/causal_lm/pytorch/loader.py`
- Added `_patch_transformers_internlm2_gguf()` with:
  - Registration of `"internlm2"` in `GGUF_SUPPORTED_ARCHITECTURES`
  - `GGUF_TO_TRANSFORMERS_MAPPING["config"]["internlm2"]` with LLaMA-equivalent key names
  - `GGUF_TO_FAST_CONVERTERS["internlm2"] = GGUFLlamaConverter`
  - Wrapping of `load_gguf_checkpoint` to remap `model_type: "internlm2"` → `"llama"` (patches 4 module references)
- Commit: `e7677065df`

### tt_forge_models — 26 loaders widened narrow-sig `_patched_load_gguf_checkpoint`
- All loaders with `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` widened to `(*args, **kwargs)` with matching call-through change
- Commit: `9ceedab323`

### tt-xla — `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- Added `KNOWN_FAILURE_XFAIL` entry for the test with the OOM reason string
- Commits: `d23041bc9`, `a1fe7e191`

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — result is XFAIL (hardware-class), not FAIL.

## Verification
- pytest exit: FAIL (OOM on device, as expected for hardware-class XFAIL)
- Hardware:    blackhole-p150b
- Duration:    680.53s (0:11:20)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/legraphista_internlm2_5_20b_chat_imat_gguf/causal_lm/pytorch/loader.py`
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
| tt-xla          | a1fe7e1918ca71a02ada382d8a6339699ca542a5 |
| tt-forge-models | 9ceedab323635130f65b8a3d4ad7e9b36157382b |

# Remediation Summary: lmstudio_deepseek_r1_distill_llama_70b_gguf-causal_lm-pytorch-DeepSeek_R1_Distill_Llama_70B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lmstudio_deepseek_r1_distill_llama_70b_gguf/causal_lm/pytorch-DeepSeek_R1_Distill_Llama_70B_GGUF-single_device-inference]

## Result
XFAIL — 70B BF16 model (~140 GB) exceeds single p150b device DRAM (~32 GB); loader TypeError also fixed

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
Original failure message:
```
Converting and de-quantizing GGUF tensors...:  88%|████████▊ | 638/724 [03:53<00:51,  1.68it/s]
```

Actual error (reproduced):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After fixing the loader bug, the underlying hardware capacity failure:
```
TT_FATAL @ bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 469762048 B DRAM buffer across 8 banks, where each bank needs to store 58720256 B, but bank size is 4273390016 B (allocated: 4113318208 B, free: 160071808 B, largest free block: 45351360 B)
```

## Root cause
Two issues were present:

**1. Loader bug (primary):** 26 GGUF model loaders in tt_forge_models defined `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` at module level, replacing `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` globally at import time. In a full pytest session, these loaders are imported before the lmstudio test runs, leaving the narrowly-typed wrapper as the active function. Transformers 5.x now calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` from `modeling_utils.py:4016`, which hits the wrapper and raises `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. The tqdm progress bar at 88% was the last output before the error.

**2. Hardware capacity (terminal):** After fixing the loader, the model loads fully (~140 GB BF16 for 70B parameters). When the test framework attempts to tilize and place tensors on the p150b device (~32 GB DRAM), the allocator fails: 96% of all 8 DRAM banks (4113 MB / 4273 MB each) were consumed and the remaining allocation of 448 MB (5× ~89 MB per bank) could not fit. This is a genuine hardware capacity ceiling, not a compiler bug.

## Fix
**Loader fix** (`tt_forge_models` remediation branch `remediation/lmstudio_deepseek_r1_distill_llama_70b_gguf-causal_lm-pytorch-DeepSeek_R1_Distill_Llama_70B_GGUF-single_device-inference`):

Changed signature from `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(*args, **kwargs):` and updated the internal call from `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `_orig_load_gguf_checkpoint(*args, **kwargs)` in all 26 affected loaders:
- `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `dmind_3_mini_i1_gguf`, `gpt_oss_swallow_120b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`, and 20 `mradermacher_*`/`tvall43_*`/`qwen_3_5_imatrix_gguf`/`unified_reward_flex_qwen35_27b_gguf` loaders.

**XFAIL config** (`tt-xla` remediation branch):

Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` for `lmstudio_deepseek_r1_distill_llama_70b_gguf/causal_lm/pytorch-DeepSeek_R1_Distill_Llama_70B_GGUF-single_device-inference` with the OOM message and hardware explanation.

## Verification
- pytest exit: FAIL (OOM on device after loader fix)
- Hardware:    blackhole-p150b
- Duration:    957.16s (0:15:57) for the run that exposed the OOM
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
| tt-xla          | 2ba4a7c2bc5fada33c0f7b813a51e45fc75c06d6 |
| tt-forge-models | 41b2e755f85e9e3ffb0bf5eb63eb3eca68eaf2ad |

# Remediation Summary: edensfall_l3_3_70b_0_1a_i1_gguf-causal_lm-pytorch-EDENSFALL_L3_3_70B_0_1A_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[edensfall_l3_3_70b_0_1a_i1_gguf/causal_lm/pytorch-EDENSFALL_L3_3_70B_0_1A_I1_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — 70B model at bfloat16 (~140 GB weights) exceeds single-device DRAM capacity (~32 GB); loader-layer TypeError fixed as prerequisite

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
Original stated failure: `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`

Reproduced as: `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

After loader fix, secondary failure at inference:
`TT_FATAL @ bank_manager.cpp:439: Out of Memory: Not enough space to allocate 469762048 B DRAM buffer across 8 banks, where each bank needs to store 58720256 B, but bank size is 4273390016 B (allocated: 4113318208 B, free: 160071808 B, largest free block: 45351360 B)`

## Root cause
Two-layer failure:

**Layer 1 (loader):** At pytest collection time, `TorchDynamicLoader.setup_test_discovery()` imports all loader.py files. Twenty-six GGUF loaders (qwen35, gpt-oss variants) patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` globally at import time with a function whose signature is `(gguf_path, return_tensors=False)`. Transformers 5.x added `model_to_load` as a third parameter, so when the edensfall test subsequently calls `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`, transformers calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` — which fails with `TypeError` because the globally-installed patched function doesn't accept that keyword.

**Layer 2 (hardware-class):** After fixing the loader, the model loads and compiles, then fails at inference with an OOM. EdensFall-L3.3-70B loaded at bfloat16 requires approximately 140 GB of DRAM for weights alone (70B parameters × 2 bytes), while a single p150b device has ~32 GB of DRAM. The device is 95%+ allocated when inference begins, and the attempt to allocate one more 450 MB tensor for tilize fails.

## Fix
**Loader fix** (`third_party/tt_forge_models` — 26 files): Updated `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None)` and forwarded `model_to_load` to the wrapped `_orig_load_gguf_checkpoint` call in all 26 loaders that had the limited signature.

**XFAIL config** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`): Added `KNOWN_FAILURE_XFAIL` entry for this test explaining the hardware capacity limitation.

Files changed:
- `third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry)

## Verification
- pytest exit: XFAIL
- Hardware:    blackhole-p150b
- Duration:    921.53s (0:15:21)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/` — 26 loader.py files (model_to_load kwarg fix)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 145ecd860bedddb627e5981d355aa82e09d1daee |
| tt-forge-models | 6349d2bcf92b94c75b94e6e6a0e4780193d563c0 |

# Remediation Summary: bartowski_skywork_skywork_swe_32b_gguf-causal_lm-pytorch-Skywork_SWE_32B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_skywork_skywork_swe_32b_gguf/causal_lm/pytorch-Skywork_SWE_32B_GGUF-single_device-inference]

## Result
XFAIL — 32B Skywork GGUF model (~31 GB quantized weights) exhausts single-device DRAM on p150b (34 GB); OOM during inference buffer allocation

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-dram-capacity-32b-model-oom

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported error:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Reproduced error (after gguf is installed):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix, final error:
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 283115520 B DRAM buffer across 8 banks,
where each bank needs to store 35389440 B, but bank size is 4273390016 B
(allocated: 4209222080 B, free: 64167936 B, largest free block: 26546624 B)
```

## Root cause
Two layered loader bugs, both fixed, followed by a genuine hardware capacity ceiling:

1. **Missing requirements.txt** (loader): The `bartowski_skywork_skywork_swe_32b_gguf` loader had no `requirements.txt` listing `gguf>=0.10.0`. In a fresh environment without gguf installed, transformers raises `ImportError` before reaching the model load.

2. **Narrow `_patched_load_gguf_checkpoint` signature** (loader): 26 other GGUF loaders monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with signature `(gguf_path, return_tensors=False)`. Transformers 5.2 added a `model_to_load=` keyword argument to this call site. When any of those 26 loaders is imported during pytest collection (before the Skywork test runs), the global function is replaced with the narrow-signature patch, causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` for every subsequent GGUF test.

3. **Hardware capacity ceiling**: After fixing both loader bugs, the model loads successfully but OOMs during inference. The Skywork-SWE-32B-Q4_K_M GGUF is ~18 GB on disk but expands to ~31 GB of quantized weight buffers on device (as seen by 3.92 GB × 8 banks = ~31.4 GB allocated). The p150b device has only ~34 GB total DRAM (4.27 GB × 8 banks). When the runtime tries to allocate the first 270 MB inference buffer, only 61 MB per bank (488 MB total) remains, which is insufficient.

## Fix
**Loader fixes (tt-forge-models, remediation branch):**
- `bartowski_skywork_skywork_swe_32b_gguf/causal_lm/pytorch/requirements.txt` — created with `gguf>=0.10.0`
- 26 GGUF loader files — changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(*args, **kwargs):` and forwarded args through to `_orig_load_gguf_checkpoint(*args, **kwargs)`

**Test config (tt-xla, remediation branch):**
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `KNOWN_FAILURE_XFAIL` entry for this test

## Verification
- pytest exit: FAIL (OOM — expected hardware-class outcome after loader fixes)
- Hardware:    blackhole-p150b
- Duration:    853.42s (0:14:13)
- Tier A attempts: N/A

## Files changed
**tt-forge-models (remediation branch `remediation/bartowski_skywork_skywork_swe_32b_gguf-causal_lm-pytorch-Skywork_SWE_32B_GGUF-single_device-inference`):**
- `bartowski_skywork_skywork_swe_32b_gguf/causal_lm/pytorch/requirements.txt` (created)
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

**tt-xla (remediation branch `remediation/bartowski_skywork_skywork_swe_32b_gguf-causal_lm-pytorch-Skywork_SWE_32B_GGUF-single_device-inference`):**
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 950caeab5efce67bd8b464d62d12326674a939ac |
| tt-forge-models | f7de68b4c605d7859365b90abedc0574ac67dc8d |

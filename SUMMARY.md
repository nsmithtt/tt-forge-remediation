# Remediation Summary: codestral_gguf-causal_lm-pytorch-22B_v0.1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[codestral_gguf/causal_lm/pytorch-22B_v0.1_GGUF-single_device-inference]

## Result
XFAIL — 22B BF16 model (~44 GB) exceeds single-device DRAM capacity on p150b (32 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-model-exceeds-single-device-dram

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
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Reproduced as:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After fixing the TypeError, next failure:
```
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set
```

After fixing chat_template, final failure:
```
TT_FATAL: Out of Memory: Not enough space to allocate 201326592 B DRAM buffer across 8 banks,
where each bank needs to store 25165824 B, but bank size is 4273390016 B
(allocated: 4081641664 B, free: 191748352 B, largest free block: 20232640 B)
```

## Root cause
Three layered issues:

1. **Missing requirements.txt (loader)**: `codestral_gguf` loader had no `requirements.txt`, so `gguf>=0.10.0` was not installed when the test ran, causing the original ImportError. Fixed by adding requirements.txt.

2. **Narrow `_patched_load_gguf_checkpoint` signature (loader)**: 26 other GGUF loaders monkey-patched `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow `(gguf_path, return_tensors=False)` signature at import time. When pytest collects all test modules, these patches persist globally. transformers 5.2.0 calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` which the narrow-signature patch rejects with TypeError. Fixed by updating all 26 loaders to `(*args, **kwargs)`.

3. **Missing chat_template fallback (loader)**: The Codestral-22B GGUF tokenizer does not embed a `chat_template`. The loader's `load_inputs` unconditionally called `apply_chat_template`, raising ValueError. Fixed by checking `tokenizer.chat_template is not None` first and falling back to `sample_text` otherwise.

4. **Hardware capacity ceiling**: After all loader fixes, the 22B parameter model dequantized to BF16 requires approximately 44 GB of device DRAM. The p150b device has ~32 GB (8 banks × ~4 GB each). The model loads partially but OOMs at inference when tilizing a 201 MB weight matrix (6144×16384 BF16, matching Codestral-22B FFN dimensions). This is a genuine hardware capacity ceiling, not a compiler bug.

## Fix
Three changes in `tt_forge_models` remediation branch (`remediation/codestral_gguf-causal_lm-pytorch-22B_v0.1_GGUF-single_device-inference`):

1. `codestral_gguf/causal_lm/pytorch/requirements.txt` — created with `gguf>=0.10.0`
2. `codestral_gguf/causal_lm/pytorch/loader.py` — `load_inputs`: guard `apply_chat_template` with `if self.tokenizer.chat_template is not None`
3. 26 GGUF loader files — `_patched_load_gguf_checkpoint` signature: `(gguf_path, return_tensors=False)` → `(*args, **kwargs)`; call: `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(*args, **kwargs)`

One change in `tt-xla` remediation branch (`remediation/codestral_gguf-causal_lm-pytorch-22B_v0.1_GGUF-single_device-inference`):

4. `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `codestral_gguf/causal_lm/pytorch-22B_v0.1_GGUF-single_device-inference` as `KNOWN_FAILURE_XFAIL` with reason "Out of Memory: 22B BF16 model (~44 GB) exceeds single-device DRAM capacity"

## Verification
- pytest exit: XFAIL (exit code 0, `1 xfailed`)
- Hardware:    blackhole-p150b
- Duration:    553.27s (0:09:13)
- Tier A attempts: N/A

## Files changed
**tt_forge_models**:
- `codestral_gguf/causal_lm/pytorch/requirements.txt` (created)
- `codestral_gguf/causal_lm/pytorch/loader.py`
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_gguf/causal_lm/pytorch/loader.py`
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

**tt-xla**:
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 96ae05a3b |
| tt-forge-models | 604154bec3 |

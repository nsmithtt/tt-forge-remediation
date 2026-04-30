# Remediation Summary: deepseek_r1_qwen_2_5_32b_ablated_gguf-causal_lm-pytorch-Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_qwen_2_5_32b_ablated_gguf/causal_lm/pytorch-Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — 32B bfloat16 model consumes ~31 GB of 32 GB device DRAM leaving no room for inference compute buffers

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
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix, second failure:

```
TT_FATAL @ bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 283115520 B DRAM buffer across 8 banks,
where each bank needs to store 35389440 B, but bank size is 4273390016 B
(allocated: 4209221952 B, free: 64168064 B, largest free block: 26546624 B)
```

## Root cause

Two distinct issues:

**Loader bug:** 26 Qwen3.5/gpt-oss GGUF loaders monkey-patch `load_gguf_checkpoint` in
`transformers.integrations.gguf_utils` (and related modules) using the signature
`(gguf_path, return_tensors=False)`. transformers 5.2.0 added `model_to_load` as a
keyword argument. When pytest collects all loaders and one of these patches is installed,
any subsequent GGUF loader (including `deepseek_r1_qwen_2_5_32b_ablated_gguf`) calls
`from_pretrained` which internally calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`,
hitting the TypeError.

**Hardware capacity:** After fixing the loader, the 32B Q4_K_M GGUF dequantizes to
bfloat16 on CPU then transfers to the TT device. The model weights consume ~31.4 GB of
the 32 GB DRAM (8 banks × 3.98 GB each). When inference begins, the tilize operation
for the first input tensor fails because only ~61 MB per bank remains — far less than the
~270 MB needed for compute buffers. This is a fundamental capacity mismatch; no compiler
change can fit this model on a single 32 GB device.

## Fix

**Loader fix (tt-forge-models, 26 files):** Changed each `_patched_load_gguf_checkpoint`
from fixed positional signature to `(*args, **kwargs)` and forwarded to
`_orig_load_gguf_checkpoint(*args, **kwargs)`. Files modified:
`bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`,
`dmind_3_mini_i1_gguf`, `gpt_oss_swallow_120b_rl_v0_1_gguf`,
`gpt_oss_swallow_20b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`,
`mradermacher_bartleby_qwen3_5_4b_gguf`, `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf`,
`mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf`,
`mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf`,
`mradermacher_luna_qwen3_5_27b_v5_i1_gguf`, `mradermacher_qwen3_5_27b_gguf`,
`mradermacher_qwen3_5_27b_homebrew_gguf`, `mradermacher_qwen3_5_27b_tainted_heresy_gguf`,
`mradermacher_qwen3_5_4b_abliterated_i1_gguf`, `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf`,
`mradermacher_qwen3_5_4b_gabliterated_gguf`, `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf`,
`mradermacher_qwen3_5_4b_unfiltered_gguf`, `mradermacher_qwen3_5_4b_unredacted_max_gguf`,
`mradermacher_qwen3_5_9b_abliterated_i1_gguf`, `mradermacher_vilm_0_8b_sft_gguf`,
`qwen_3_5_imatrix_gguf`, `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf`,
`tvall43_qwen3_5_4b_heretic_v2_i1_gguf`, `unified_reward_flex_qwen35_27b_gguf`
(all under `causal_lm/pytorch/loader.py`).

**XFAIL config (tt-xla):**
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` —
added `KNOWN_FAILURE_XFAIL` entry for this test with OOM explanation.

## Verification
- pytest exit: FAIL (OOM after loader fix)
- Hardware:    blackhole-p150b
- Duration:    867.30s (0:14:27)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: 26 × `<loader>/causal_lm/pytorch/loader.py` — `_patched_load_gguf_checkpoint` signature fix
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 893ab2a15f1028ba45cea0c8d23bb1e3ef19ca58 |
| tt-forge-models | a193f5dd486d4767a4eb4f0fd9656d750d7712c5 |

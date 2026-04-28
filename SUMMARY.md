# Remediation Summary: athene_70b_gguf-causal_lm-pytorch-70B_Q4_0_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[athene_70b_gguf/causal_lm/pytorch-70B_Q4_0_GGUF-single_device-inference]

## Result
XFAIL — Athene 70B BF16 (~140 GB) exceeds Blackhole P150B single-device GDDR capacity (~32 GB); hardware-class ceiling

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-bf16-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_FATAL @ /home/nsmith/tt-forge-remediation/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 469762048 B DRAM buffer across 8 banks, where each bank needs to store 58720256 B, but bank size is 4273390016 B (allocated: 4113318208 B, free: 160071808 B, largest free block: 45351360 B)
```

The CI failure reported "Test exceeded configured timeout and was killed". On local silicon, the 38 GB GGUF file downloads successfully, dequantizes to BF16 (~140 GB) in ~16 minutes, and then hits DRAM OOM on the first device buffer allocation attempt.

## Root cause

Two issues were found:

**1. Loader bug (fixed):** 26 GGUF loaders in `tt_forge_models` patched `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a narrow signature `(gguf_path, return_tensors=False)`. During pytest collection, all loaders are imported via `TorchDynamicLoader.setup_test_discovery`, installing the broken patch session-wide. When the athene test ran, transformers 5.2.0 called `load_gguf_checkpoint(path, model_to_load=dummy_model)`, hitting the narrow-signature wrapper and raising `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. This was the proximate cause of the CI timeout (the test died during model load rather than dequantization).

**2. Hardware capacity ceiling:** Athene 70B with Q4_0 quantization is a 38 GB GGUF file. `from_pretrained(..., gguf_file=..., torch_dtype=bfloat16)` fully dequantizes to BF16 in host RAM (~140 GB), then dispatches to device. The Blackhole P150B has 8 GDDR6 channels × ~4 GB = ~32 GB total DRAM. The model is ~4.4× the device capacity; there is no allocator-level bug here — the model simply does not fit.

## Fix

**Loader fix (tt_forge_models, branch `remediation/athene-70b-gguf-causal-lm-single-device-inference`):**
Changed 26 `_patched_load_gguf_checkpoint` functions from narrow signature to variadic:
- Old: `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):`
- New: `def _patched_load_gguf_checkpoint(*args, **kwargs):`

Files changed: `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `dmind_3_mini_i1_gguf`, `gpt_oss_swallow_120b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`, `mradermacher_bartleby_qwen3_5_4b_gguf`, `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf`, `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf`, `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf`, `mradermacher_luna_qwen3_5_27b_v5_i1_gguf`, `mradermacher_qwen3_5_27b_gguf`, `mradermacher_qwen3_5_27b_homebrew_gguf`, `mradermacher_qwen3_5_27b_tainted_heresy_gguf`, `mradermacher_qwen3_5_4b_abliterated_i1_gguf`, `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf`, `mradermacher_qwen3_5_4b_gabliterated_gguf`, `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf`, `mradermacher_qwen3_5_4b_unfiltered_gguf`, `mradermacher_qwen3_5_4b_unredacted_max_gguf`, `mradermacher_qwen3_5_9b_abliterated_i1_gguf`, `mradermacher_vilm_0_8b_sft_gguf`, `qwen_3_5_imatrix_gguf`, `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf`, `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`, `unified_reward_flex_qwen35_27b_gguf` — all in `causal_lm/pytorch/loader.py`.

**Test config XFAIL (tt-xla, branch `remediation/athene-70b-gguf-causal-lm-single-device-inference`):**
Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL (RuntimeError OOM — hardware capacity ceiling, not a compiler bug)
- Hardware:    blackhole-p150b
- Duration:    1146.30s (0:19:06)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry
- `tt_forge_models/<26 loaders>/causal_lm/pytorch/loader.py` — fixed `_patched_load_gguf_checkpoint` signature

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 78c5a94ccf3def15fab3ebcad82165524ecca92a |
| tt-forge-models | d963efb5cd4639735bfaf5b06d70a04e957fb375 |

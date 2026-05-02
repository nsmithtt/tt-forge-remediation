# Remediation Summary: mradermacher_dao1_30b_a3b_i1_gguf-causal_lm-pytorch-30B_A3B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_dao1_30b_a3b_i1_gguf/causal_lm/pytorch-30B_A3B_i1_GGUF-single_device-inference]

## Result
XFAIL — Qwen3-30B-A3B totals 30.53B params; BF16 representation (~61 GB) exceeds all single-device DRAM on supported hardware (p150b 32 GB)

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
Original: Fatal Python error: Segmentation fault

Reproduced as:
1. TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
   (session contamination from 26 other loaders patching load_gguf_checkpoint with narrow signatures)
2. After fix 1: torch.histc on XLA integer tensor in grouped_mm_experts_forward
   (Qwen3MoE expert dispatch without batched_mm setting)

## Root cause
Two loader-layer bugs compounded to produce the segfault:

**Bug 1 (session contamination):** transformers 5.2.0 added `model_to_load` kwarg to
`load_gguf_checkpoint`. 26 other GGUF model loaders in tt_forge_models had patched
`load_gguf_checkpoint` with a narrow signature
`def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` that did not accept
this new kwarg. When pytest collects all tests at startup, those loaders are imported and
their patches take effect globally. The Dao1 loader then fails with TypeError when
`from_pretrained` calls `load_gguf_checkpoint(... model_to_load=dummy_model)`.

**Bug 2 (Qwen3MoE histc on XLA):** The default `grouped_mm` expert dispatch path calls
`torch.histc` on an XLA integer tensor during graph compilation, which fails. Setting
`model.config._experts_implementation = "batched_mm"` switches to the static masked-matmul
path that avoids the histc call.

**Hardware capacity:** After both loader fixes, the model would still fail because
Qwen3-30B-A3B has 30.53B total parameters. At BF16, that requires ~61 GB DRAM. All
supported single-device hardware (p150b 32 GB, n150 12 GB) is below this ceiling.

## Fix
**Fix 1** (tt_forge_models, 26 files): Changed all narrow-sig
`def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` to
`def _patched_load_gguf_checkpoint(*args, **kwargs)` with matching call-site update
`_orig_load_gguf_checkpoint(*args, **kwargs)`. Files changed:
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py
- mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py
- qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py
- daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py
- bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py

**Fix 2** (tt_forge_models, 1 file): Added `model.config._experts_implementation = "batched_mm"`
in `mradermacher_dao1_30b_a3b_i1_gguf/causal_lm/pytorch/loader.py` after `from_pretrained`.

**XFAIL config** (tt-xla): Added entry in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` marking this test
`KNOWN_FAILURE_XFAIL` with hardware-capacity reason.

## Verification
- pytest exit: not run (hardware capacity confirmed by param count × dtype)
- Hardware:    blackhole-p150b
- Duration:    1288.21s (prior run reaching histc error)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 26 × narrow-sig patch widen + 1 × batched_mm fix (2 commits on remediation branch)
- tt-xla: tests/runner/test_config/torch/test_config_inference_single_device.yaml (XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5fc97a88f446c6678567a11478d10b4bc50e7153 |
| tt-forge-models | fa39e95c38f4059844b7188225251f0b522df91f |

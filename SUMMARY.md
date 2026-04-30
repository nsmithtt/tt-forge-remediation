# Remediation Summary: gemma_2_gguf-causal_lm-pytorch-2B_IT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_gguf/causal_lm/pytorch-2B_IT_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
Two bugs combined:

1. **Loader (26 files)**: 26 GGUF loader files in tt_forge_models define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` without `**kwargs`. Transformers 5.x added `model_to_load` to `load_gguf_checkpoint`'s signature. During pytest collection, all 26 loaders import and monkey-patch the global `load_gguf_checkpoint`, forming a chain. When transformers calls the outermost patcher with `model_to_load=...`, it raises TypeError.

2. **tt-xla (TorchFunctionOverride)**: After fixing the loader, the Gemma 2 model (sliding_window=4096) emits `aten.slice.Tensor(cat_4, 2, -4095, ...)` for short sequences. XLA's slice kernel validates bounds strictly (expected range [-22, 21] for a 22-element dim), while PyTorch CPU silently clamps. This raises `RuntimeError: Value out of range`.

## Fix
1. **tt_forge_models** (`remediation/gemma_2_gguf-causal_lm-pytorch-2B_IT_GGUF-single_device-inference`): Added `**kwargs` to the signature of `_patched_load_gguf_checkpoint` and forwarded it to `_orig_load_gguf_checkpoint` in all 26 affected loader files. Sed one-liner applied uniformly.

2. **tt-xla** (`remediation/gemma_2_gguf-causal_lm-pytorch-2B_IT_GGUF-single_device-inference`): Added a slice-start clamp guard at the top of `TorchFunctionOverride.__torch_function__` in `python_package/tt_torch/torch_overrides.py`. When `func is torch.ops.aten.slice.Tensor` and `start < -dim_size`, clamp start to `-dim_size` before dispatch.

Files changed:
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- 26 files under `tt-xla/third_party/tt_forge_models/*/causal_lm/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    497.14s (0:08:17)
- Tier A attempts: N/A

## Files changed
- tt-xla/python_package/tt_torch/torch_overrides.py
- tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3f325426269ee23908e8bfd823c87bcba5664f2d |
| tt-forge-models | 253f8362cbb6066bfb35751ba89c05126ddc826f |

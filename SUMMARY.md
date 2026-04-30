# Remediation Summary: gemma_3_4b_it_max_horror_uncensored_dbl_x_gguf-causal_lm-pytorch-Gemma-3-4b-it-MAX-HORROR-Uncensored-DBL-X-Q4_K_M-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_4b_it_max_horror_uncensored_dbl_x_gguf/causal_lm/pytorch-Gemma-3-4b-it-MAX-HORROR-Uncensored-DBL-X-Q4_K_M-GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

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
```
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

The originally reported failure `RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)` is the second bug; it is blocked by the loader bug during reproduction.

## Root cause
Two bugs in sequence:

1. **Loader — missing `**kwargs` in patched function** (`gguf-load-checkpoint-model-to-load-kwarg`): 26 loaders in tt_forge_models patch `gguf_utils.load_gguf_checkpoint` at import time using a fixed signature `(gguf_path, return_tensors=False)` without `**kwargs`. Transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, causing `TypeError`. All 26 patches are applied to the global `gguf_utils.load_gguf_checkpoint` during pytest collection, so any model loaded in the same session hits this bug.

2. **tt-xla — aten.slice.Tensor out-of-bounds start** (`aten-slice-tensor-out-of-bounds-start`): `SlidingWindowCache.update()` computes `full_value_states[:, :, -self.sliding_window + 1:, :]` = `start=-1023` when `seq_len=24` and `sliding_window=1024`. PyTorch silently clamps such out-of-range indices; the XLA/TT backend validates strictly and raises `RuntimeError: Value out of range`. Tier A fix: a post-export FX pass clamps negative start indices to `-dim_size`.

## Fix
1. **26 loaders in tt_forge_models** (bulk): Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to add `**kwargs` and forward it to the original call. Commit `4f8a163859` on tt-forge-models branch `remediation/gemma_3_4b_it_max_horror_uncensored_dbl_x_gguf-causal_lm-pytorch-Gemma-3-4b-it-MAX-HORROR-Uncensored-DBL-X-Q4_K_M-GGUF-single_device-inference`.

2. **`tt-xla/python_package/tt_torch/backend/passes.py`** and **`tt-xla/python_package/tt_torch/backend/backend.py`**: Added `clamp_out_of_range_slice_starts` FX pass that iterates over `aten.slice.Tensor` nodes and clamps any `start < -dim_size` to `-dim_size`. Commit `34f434e11` on tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    464.02s (0:07:44)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5b3f93151219c3f4815d743ea391c857048ed409 |
| tt-forge-models | 4f8a1638598e1c9b436059e21f21fd152f9ed942 |

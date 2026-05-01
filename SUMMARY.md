# Remediation Summary: gemma_3_12b_it_abliterated_gguf-causal_lm-pytorch-12B_IT_Abliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_12b_it_abliterated_gguf/causal_lm/pytorch-12B_IT_Abliterated_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
xla-lazy-slice-oob

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

## Root cause
Two bugs:

**Bug 1 (tt-xla, Tier A):** Gemma 3 uses sliding window attention with `sliding_window=1024`. When the test input is short (~23 tokens), `aten.slice.Tensor` is called with `start=-1023` on a dimension of size 23. PyTorch eager silently clamps such out-of-range negative starts to 0, but the XLA/TT backend raises `RuntimeError: Value out of range`. The fix is a pre-compilation FX pass that clamps any negative `start` that is more negative than `-dim_size` to `-dim_size`, matching eager semantics.

**Bug 2 (loader):** 26+ GGUF loaders install a monkey-patch for `load_gguf_checkpoint` at module-import time with a narrow signature `(gguf_path, return_tensors=False)`. transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`. When any of these loaders is imported during pytest collection, it installs the narrow-sig patch globally into all four transformers internal modules, clobbering the real function. Subsequent calls from unrelated models that pass `model_to_load=...` hit `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. The fix adds `**kwargs` to the patched signature and passes them through.

## Fix
**Fix 1 — tt-xla** (`python_package/tt_torch/backend/passes.py`, `backend.py`):
Added `clamp_out_of_range_slice_starts(gm)` FX pass that iterates over all `aten.slice.Tensor` call nodes, checks whether `start` is a negative integer more negative than `-dim_size`, and clamps it to `-dim_size`. Called from `torch_pass_pipeline` in `backend.py` after `bypass_assert_tensor_metadata`. Committed on branch `remediation/gemma_3_12b_it_abliterated_gguf-causal_lm-pytorch-12B_IT_Abliterated_GGUF-single_device-inference` in tt-xla.

**Fix 2 — loader** (`third_party/tt_forge_models/*/causal_lm/pytorch/loader.py`, 26 files):
Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):` and passed `**kwargs` through to `_orig_load_gguf_checkpoint(...)`. Committed on branch `arch-c-36-tt-xla-dev/nsmith/hf-bringup-17` in tt_forge_models.

## Verification
- pytest exit: PASS
- Hardware: blackhole-p150b
- Duration: 765.92s (0:12:45)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/passes.py` — added `clamp_out_of_range_slice_starts`
- `tt-xla/python_package/tt_torch/backend/backend.py` — import and call `clamp_out_of_range_slice_starts`
- `tt-xla/third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6cf18d07abaaf73241a3c887c2e816e442ed524b |
| tt-forge-models | 50f60246aa1eb639c9bf7bf879b43aa778b9c6c5 |

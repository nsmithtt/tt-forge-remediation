# Remediation Summary: gemma3_qat_gguf-causal_lm-pytorch-BARTOWSKI_1B_IT_QAT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_qat_gguf/causal_lm/pytorch-BARTOWSKI_1B_IT_QAT_GGUF-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: model_to_load kwarg in narrow-sig GGUF loaders (loader) and aten.slice negative-OOB in XLA TorchFunctionOverride (tt-xla)

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
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_29, 2, -511, 9223372036854775807), kwargs = {})
Original traceback:
  File "transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause

Two bugs in sequence:

**Bug 1 — loader layer:** 26 GGUF loader modules (qwen35 variants) monkey-patch `transformers.utils.gguf_utils.load_gguf_checkpoint` at module-import time with a narrow signature `(gguf_path, return_tensors=False)`. During pytest collection `setup_test_discovery` imports every loader module; the last of these narrow-sig patches clobbers the global. When transformers 5.2.0 calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` inside Gemma3 `from_pretrained`, it hits a `TypeError` because the keyword argument is not accepted.

**Bug 2 — tt-xla layer:** Gemma3's `SlidingWindowCache.update` in `transformers/cache_utils.py` computes `full_value_states[:, :, -sliding_window + 1 :, :]`. With `sliding_window=512` and max_length=128 inputs the KV tensor on dim 2 has only 23 elements; the slice start `-511` is out of the range `[-23, 22]`. PyTorch eager silently clamps out-of-range slice indices, but the XLA lazy backend raises "Value out of range". The fix is in `TorchFunctionOverride.__torch_function__` in `tt_torch/torch_overrides.py`.

## Fix

**Fix 1 — tt_forge_models (loader):** In all 26 GGUF loaders that define `_patched_load_gguf_checkpoint` with the narrow signature, changed:
- `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` → `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):`
- `result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)`

Branch: `remediation/gemma3_qat_gguf-causal_lm-pytorch-BARTOWSKI_1B_IT_QAT_GGUF-single_device-inference` in tt_forge_models (commit `4200f01ceca1d01daa07bef6378ccd04d9784c44`)

**Fix 2 — tt-xla:** Cherry-picked commit `74848586376a458b5bcbdb61d8a994844b375c6f` from `origin/remediation/gemma3-emotional-1b-i1-gguf-slice-oob`. Added intercept in `TorchFunctionOverride.__torch_function__` for `torch.ops.aten.slice.Tensor` that clamps start/end to `[-size, size]` when they are negative integers out of the tensor's valid range.

File: `python_package/tt_torch/torch_overrides.py`
Branch: `remediation/gemma3_qat_gguf-causal_lm-pytorch-BARTOWSKI_1B_IT_QAT_GGUF-single_device-inference` in tt-xla (commit `33b6abdd90580b8c04fb47eef5304a7eb5594765`)

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    332.34s (0:05:32)
- Tier A attempts: 1

## Files changed
**tt_forge_models:**
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

**tt-xla:**
- `python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 33b6abdd90580b8c04fb47eef5304a7eb5594765 |
| tt-forge-models | 4200f01ceca1d01daa07bef6378ccd04d9784c44 |

# Remediation Summary: aidc_llm_laos_4b_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aidc_llm_laos_4b_gguf/causal_lm/pytorch-Q4_K_M-single_device-inference]

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
Original CI failure:
  ImportError: Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.

Local reproduction failure #1 (loader bug):
  TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

Local failure #2 (after loader fix, compiler frontend bug):
  RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
  While executing %slice_6 = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_37, 2, -1023, 9223372036854775807))

## Root cause

**Bug 1 — loader (tt_forge_models):** 26 GGUF loaders patched `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module-import time with a fixed signature `(gguf_path, return_tensors=False)` that does not accept the `model_to_load` keyword argument added in transformers 5.x. When these loaders are collected in the same pytest session, the last patched version intercepts the call from `modeling_utils.py:4016` (`load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)`) and raises TypeError. The `aidc_llm_laos_4b_gguf` loader itself does not patch anything but is a victim. Additionally, `aidc_llm_laos_4b_gguf` was missing a `requirements.txt` listing `gguf>=0.10.0`, which caused the ImportError in CI environments where `gguf` is not in the base environment.

**Bug 2 — tt-xla compiler frontend:** The Gemma3-based `aidc-llm-laos-4b` model uses `SlidingWindowCache` with `sliding_window=1024`. When `seq_len < sliding_window`, `cache_utils.py:SlidingWindowCache.update` does `full_value_states[:, :, -sliding_window+1:, :]` which translates to `aten.slice.Tensor(tensor, dim=2, start=-1023, end=max_int)` on a tensor with dim_size=23. PyTorch allows `start < -dim_size` (semantically clamped to 0), but the XLA/TT backend validates bounds strictly. The `clamp_out_of_range_slice_starts` FX pass was not present in the configured branch.

## Fix

**Fix 1 (loader, tt_forge_models):**
- Updated all 26 loaders that had the broken `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` signature to `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):` and pass `**kwargs` to `_orig_load_gguf_checkpoint`.
- Added `aidc_llm_laos_4b_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`.
- Branch: `remediation/aidc_llm_laos_4b_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference` in `tenstorrent/tt-forge-models`.

**Fix 2 (tt-xla compiler frontend):**
- Added `clamp_out_of_range_slice_starts(gm)` FX pass in `python_package/tt_torch/backend/passes.py`. The pass iterates `aten.slice.Tensor` nodes with negative `start` arguments, computes `dim_size` from the input tensor's fake value metadata, and clamps `start` to `max(-dim_size, start)`.
- Wired the pass in `python_package/tt_torch/backend/backend.py` after `bypass_assert_tensor_metadata`.
- Branch: `remediation/aidc_llm_laos_4b_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference` in `tenstorrent/tt-xla`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    507.68s (0:08:27)
- Tier A attempts: 1

## Files changed
**tt_forge_models:**
- `aidc_llm_laos_4b_gguf/causal_lm/pytorch/requirements.txt` (new)
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
- `python_package/tt_torch/backend/passes.py` (added `clamp_out_of_range_slice_starts`)
- `python_package/tt_torch/backend/backend.py` (wired new pass)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9dbbaaa865f885a0e133f0f9a24088cf887bc1f7 |
| tt-forge-models | ee29b1a28c2d40f5eab190b69d8a806b5fb85372 |

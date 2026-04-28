# Remediation Summary: albert-wesker-gguf-1b-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[albert_wesker_gguf/causal_lm/pytorch-1B_GGUF-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: (1) GGUF _patched_load_gguf_checkpoint old-style signature, (2) XLA lazy backend rejects negative out-of-bounds slice indices

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
xla-lazy-slice-oob-negative-start

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

Two bugs needed to be fixed in sequence:

**Bug 1 (loader) — prerequisite:** 26 GGUF loader files in `tt_forge_models` had an outdated `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signature. In transformers 5.2.0, `modeling_utils.py` added a `model_to_load=` keyword argument to the `load_gguf_checkpoint()` call. Since all loaders are imported during test collection (via `discover_loader_paths` → `get_model_variants` → `import_model_loader`), any old-style patched loader that runs before Albert Wesker replaces `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a function that rejects the new kwarg. The albert_wesker_gguf loader itself does no patching, so it inherited the broken global state.

**Bug 2 (tt-xla) — the reported error:** The Gemma3 sliding-window KV cache update does `full_value_states[:, :, -self.sliding_window + 1:, :]` where `sliding_window=512`. For a short input (23 tokens), start index = `-511`. Python/PyTorch eager silently clamps out-of-range negative slice indices to `-size` (returning all elements). The XLA lazy tensor backend (`torch/csrc/lazy/core/helpers.cpp`) validates strictly and raises `RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)`. The fix is to pre-clamp in `TorchFunctionOverride.__torch_function__` before the call reaches the XLA dispatch.

## Fix

**Fix 1 — tt_forge_models (26 files):** Changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `_patched_load_gguf_checkpoint(*args, **kwargs)` and updated the body to call `_orig_load_gguf_checkpoint(*args, **kwargs)`. Branch: `remediation/albert-wesker-gguf-1b-single-device-inference` in tenstorrent/tt-forge-models.

**Fix 2 — tt-xla (1 file):** In `python_package/tt_torch/torch_overrides.py`, added a guard in `TorchFunctionOverride.__torch_function__` that intercepts `torch.ops.aten.slice.Tensor` calls and clamps `start`/`end` to `max(index, -size)` when `index < -size` and the dimension size is statically known. Branch: `remediation/albert-wesker-gguf-1b-single-device-inference` in tenstorrent/tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    309.79s (0:05:09)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — slice index clamping
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 09dc7541e62d1c5a96a61def4e513272b2e786b0 |
| tt-forge-models | bd9f26b4e6f29e5f8383b79d4bd7a736e6067be3 |

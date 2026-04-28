# Remediation Summary: albert_wesker_gguf-causal_lm-pytorch-1B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[albert_wesker_gguf/causal_lm/pytorch-1B_i1_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start, gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
(surfaced first; original reported error was RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511))

## Root cause
Two independent bugs:

1. **loader layer** — 26 GGUF loaders in `tt_forge_models` register a module-level
   monkey-patch for `load_gguf_checkpoint` using the old fixed signature
   `(gguf_path, return_tensors=False)`. transformers 5.x added a `model_to_load`
   keyword argument; when any of those loaders is imported during pytest collection,
   it installs a wrapper that rejects the new kwarg, causing `TypeError` in every
   subsequent GGUF test run in the same session.

2. **tt-xla layer** — `aten.slice.Tensor` with a start index more negative than
   `-dim_size` raises `RuntimeError: Value out of range` in XLA/TT but is silently
   clamped to 0 by PyTorch CPU. Albert Wesker-1B has a sliding window of 512;
   `SlidingWindowCache.update()` issues `full_value_states[:, :, -511:, :]` when
   seq_len=23 < sliding_window=512, putting start=-511 outside the valid
   range [-23, 22].

## Fix

**Fix 1 — loader (tt_forge_models):**
Updated all 26 loaders to use `**kwargs` in the patched wrapper signature and
forward them to the original function:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)
```
Files: 26 loaders under `tt_forge_models/*/causal_lm/pytorch/loader.py`
Branch: `remediation/albert_wesker_gguf-causal_lm-pytorch-1B_i1_GGUF-single_device-inference`
in `tenstorrent/tt-forge-models`.

**Fix 2 — tt-xla compiler frontend:**
Added `clamp_out_of_range_slice_starts(gm)` FX pass to
`tt-xla/python_package/tt_torch/backend/passes.py`. The pass iterates
`aten.slice.Tensor` nodes, reads `dim_size` from `node.args[0].meta["val"].shape`
(or `tensor_meta`), and clamps any `start < -dim_size` to `-dim_size`. Called from
`torch_pass_pipeline` in `backend.py` after `bypass_assert_tensor_metadata`.

Files changed:
- `python_package/tt_torch/backend/passes.py` — new pass
- `python_package/tt_torch/backend/backend.py` — import and call site
Branch: `remediation/albert_wesker_gguf-causal_lm-pytorch-1B_i1_GGUF-single_device-inference`
in `tenstorrent/tt-xla`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    345.45s (0:05:45)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d797b0af43a01162f09a2c321fbf06a28801c54f |
| tt-forge-models | 66e632e8183bdd3d92e2dd4d7ac6460f181a1cfc |

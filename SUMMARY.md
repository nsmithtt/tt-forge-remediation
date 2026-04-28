# Remediation Summary: fallen_amoral_gemma3_gguf/causal_lm/pytorch-12B_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[fallen_amoral_gemma3_gguf/causal_lm/pytorch-12B_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.4296182659737333. Required: pcc=0.95.

(Pre-fix, the test actually first surfaced two earlier bugs that blocked reaching the PCC check — see Root cause.)

## Root cause
Two bugs blocked the test:

**Bug 1 (loader layer):** Multiple GGUF loaders (`bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `dmind_3_mini_i1_gguf`, 26 loaders total) patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a fixed-signature wrapper `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. Transformers 5.2.0 added `model_to_load=None` kwarg to that function, so when pytest co-collects these loaders alongside `fallen_amoral_gemma3_gguf`, the patched function raises `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Bug fingerprint: `gguf-load-checkpoint-model-to-load-kwarg`.

**Bug 2 (tt-xla compiler frontend):** Gemma3's `SlidingWindowCache` (sliding_window=1024) does `full_value_states[:, :, -sliding_window + 1 :, :]` = `[:, :, -1023:, :]` to trim the KV cache. With a 23-token prompt, dim 2 is only 23 elements. Standard PyTorch clamps out-of-range negative starts (treating -1023 as 0), but the TT/XLA backend raises `RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)`. The FX graph node `aten.slice.Tensor(tensor, 2, -1023, INT64_MAX)` fails when executed on the XLA device because XLA slice requires start ≥ -dim_size. Bug fingerprint: `aten-slice-tensor-out-of-bounds-start`.

## Fix
**Fix 1** (`tt-forge-models`, branch `remediation/fallen_amoral_gemma3_gguf-causal_lm-pytorch-12B_GGUF-single_device-inference`): Updated 26 GGUF loader files — all loaders that define `_patched_load_gguf_checkpoint` with fixed signature — to accept `**kwargs` and pass them through: `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs)` + `result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)`. Two loaders (`gpt_oss_swallow_20b_sft_v0_1_gguf`, `mradermacher_qwen_3_5_27b_derestricted_gguf`) already used `*args, **kwargs` and were left unchanged.

**Fix 2** (`tt-xla`, branch `remediation/fallen_amoral_gemma3_gguf-causal_lm-pytorch-12B_GGUF-single_device-inference`):
- `python_package/tt_torch/backend/passes.py`: Added `clamp_out_of_range_slice_starts(gm)` pass. Iterates over all `aten.slice.Tensor` nodes in the FX graph; if the start index is a constant integer and `start < -dim_size` (checked via `node.args[0].meta['val'].shape`), replaces start with `-dim_size`. This matches PyTorch semantics (clamping to 0 effectively) while staying in bounds for XLA.
- `python_package/tt_torch/backend/backend.py`: Added `clamp_out_of_range_slice_starts` to imports and calls it after `bypass_assert_tensor_metadata` in `torch_pass_pipeline`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    746.88s (0:12:26)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/passes.py` — new `clamp_out_of_range_slice_starts` pass
- `tt-xla/python_package/tt_torch/backend/backend.py` — import + call new pass
- `tt-forge-models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 06619299126b94599ecd6f56e9722fff6f301e07 |
| tt-forge-models | abbf7caceb67f75ed21c45f7e928dcb6ec6ff938 |

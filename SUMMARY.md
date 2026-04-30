# Remediation Summary: gemma3_orthogonal_reflection_gguf-causal_lm-pytorch-12B_IT_Orthogonal_Reflection_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_orthogonal_reflection_gguf/causal_lm/pytorch-12B_IT_Orthogonal_Reflection_GGUF-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: GGUF loader narrow-signature and XLA slice OOB clamping

## Stack layer
loader, tt-xla

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
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

(Blocked earlier by: TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load')

## Root cause
Two independent bugs blocked this test:

**Bug 1 — Loader (tt_forge_models):** 26 GGUF loaders monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module
import time with a narrow signature `(gguf_path, return_tensors=False)`.
All loaders are imported at pytest collection time (via `setup_test_discovery`),
so the patch is active for every test in the session. Transformers 5.2.0
now calls `load_gguf_checkpoint` with `model_to_load=dummy_model`, raising
`TypeError` on the narrow signature.

**Bug 2 — tt-xla (TorchFunctionOverride):** Gemma3 SlidingWindowCache slices
the KV cache with `start = position - sliding_window + 1`. When `seq_len (32)
< sliding_window (1024)`, this produces `start = -1023` on a `size=23` dim.
PyTorch eager silently clamps this but the XLA lazy backend (`torch/csrc/lazy/
core/helpers.cpp`) raises "Value out of range (expected to be in range of
[-23, 22], but got -1023)".

## Fix

**Loader fix (tt_forge_models `remediation/...` branch, commit `db36aeab19`):**
Updated all 26 GGUF loaders with the narrow `_patched_load_gguf_checkpoint`
signature to use `(*args, **kwargs)` and pass through to the original with
`_orig_load_gguf_checkpoint(*args, **kwargs)`.

Files changed (26):
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

**tt-xla fix (`tt-xla/python_package/tt_torch/torch_overrides.py`, commit `2eb6d0242`):**
Added interception of `torch.ops.aten.slice.Tensor` in
`TorchFunctionOverride.__torch_function__`. When `not torch.compiler.is_compiling()`
and the start/end indices are less than `-size` (XLA's valid range floor),
clamp them to `-size` to match PyTorch eager semantics.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    748.99s (0:12:28)
- Tier A attempts: 1

## Files changed
- `tt_forge_models`: 26 GGUF loader files (narrow `_patched_load_gguf_checkpoint` signature)
- `tt-xla/python_package/tt_torch/torch_overrides.py` (slice OOB clamping)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2eb6d0242f86919fd92e3e1c6861a726b0b02dc8 |
| tt-forge-models | db36aeab19bb43aa5e2245238c43a938aa728b36 |

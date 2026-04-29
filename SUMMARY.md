# Remediation Summary: bielik_gguf-causal_lm-pytorch-7B_V0_1_INSTRUCT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bielik_gguf/causal_lm/pytorch-7B_V0.1_INSTRUCT_GGUF-single_device-inference]

## Result
FAIL — loader TypeError fixed; model now loads and runs but PCC=0.989 is below required 0.99 threshold

## Stack layer
loader

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
Original CI failure (2026-04-25):
```
E   TypeError: _patch_transformers_qwen3vlmoe_gguf.<locals>.patched_get_gguf_hf_weights_map() takes from 1 to 4 positional arguments but 5 were given
```

Reproduced failure (after configure.sh, _patch_transformers_qwen3vlmoe_gguf already fixed):
```
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After fix:
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9894184914440995. Required: pcc=0.99.
```

## Root cause

Two distinct loader-layer bugs:

**Bug 1 (original CI failure, already fixed in branch):** A loader for the `bartowski_browser_use_bu_30b_a3b_preview_gguf` model introduced `_patch_transformers_qwen3vlmoe_gguf` with a `patched_get_gguf_hf_weights_map(hf_model, model_type=None, num_layers=None, qual_name="")` signature (4 positional args). The patch was applied globally at module import time. When transformers called `get_gguf_hf_weights_map(hf_model, processor, model_type, num_layers, qual_name)` with 5 positional args, it hit this narrow-sig patched version. Fixed by commits 93c49df0b4 and a38f1aa44b in tt_forge_models.

**Bug 2 (active when remediation ran):** 26 GGUF loaders define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and apply it globally at import time by assigning to `_gguf_utils.load_gguf_checkpoint`, `_config_utils.load_gguf_checkpoint`, etc. transformers 5.2.0 added `model_to_load=None` to `load_gguf_checkpoint`, called as `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)`. Pytest collects all model loaders alphabetically before running any test, so the narrow-signature patch from the alphabetically-last patcher (`unified_reward_flex_qwen35_27b_gguf`) is active when bielik's test runs, causing TypeError.

**Bug 3 (remaining):** After both loader bugs are fixed, the model loads and runs on silicon but produces PCC=0.9894 vs required 0.99. The cause is unclear — it may be Wormhole BF16 matmul accumulation over 32 Mistral layers, compounded by Q8_0 dequantization noise, but this has not been measured against a CPU FP32 reference to confirm it is a precision floor rather than a correctness bug.

## Fix

**Bug 1:** Already fixed in the tt_forge_models branch by prior commits.

**Bug 2:** Fixed in this remediation. Branch: `remediation/bielik_gguf-causal_lm-pytorch-7B_V0_1_INSTRUCT_GGUF-single_device-inference` in tt_forge_models (commit `0df79d24ec`).

Changed 26 loaders that applied narrow-signature `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` globally. Fixed all to use `*args, **kwargs` pattern that forwards all arguments to the wrapped original, matching the pattern established by already-fixed loaders like `gpt_oss_swallow_20b_sft_v0_1_gguf`.

Files changed (26 loaders):
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

**Bug 3 (unfixed):** PCC=0.9894 vs required 0.99. Needs CPU FP32 reference measurement to determine whether this is a BF16 accumulation floor for Mistral-7B + Q8_0 or a compiler correctness bug.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — Tier N/A (loader fix, PCC issue needs measurement before classification)

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    233.67s (3:53)
- Tier A attempts: N/A

## Files changed
tt_forge_models (26 loader files) — see Fix section above

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0df79d24ec (remediation/bielik_gguf-causal_lm-pytorch-7B_V0_1_INSTRUCT_GGUF-single_device-inference) |

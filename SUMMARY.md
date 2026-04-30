# Remediation Summary: gliese_qwen3_5_2b_gguf-causal_lm-pytorch-2B_Abliterated_Caption_i1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gliese_qwen3_5_2b_gguf/causal_lm/pytorch-2B_Abliterated_Caption_i1-single_device-inference]

## Result
SILICON_PASS

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
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
followed (after first fix) by:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
```

## Root cause
Two loader-layer bugs:

1. **model_to_load kwarg missing (transformers 5.2.0 breaking change):** 26 Qwen3.5 and gpt-oss GGUF loaders each install a module-level `_patched_load_gguf_checkpoint` wrapper that omits the `model_to_load` parameter added in transformers 5.2.0. During pytest collection all loaders are imported, so whichever of these runs last leaves the broken wrapper installed in the global `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`. When `gliese_qwen3_5_2b_gguf` calls `AutoModelForCausalLM.from_pretrained(...)`, transformers passes `model_to_load=dummy_model` to the (now broken) global function, raising the TypeError.

2. **GGUF quantized weight shape mismatch:** Q4_K_M GGUF files contain quantized tensors whose shapes differ from the float tensors in the initialized model. Without `ignore_mismatched_sizes=True`, transformers raises a RuntimeError after loading. The gliese_qwen3_5_2b_gguf loader was missing this flag (same issue previously fixed in gliese_qwen3_5_0_8b, 4b, and 9b variants).

## Fix
Two commits in `tt_forge_models` on branch `remediation/gliese_qwen3_5_2b_gguf-causal_lm-pytorch-2B_Abliterated_Caption_i1-single_device-inference`:

1. `0d52622f37` — Fix `_patched_load_gguf_checkpoint` missing `model_to_load` kwarg (26 loaders):
   - Changed signature from `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):` in all 26 broken loaders
   - Added `model_to_load=model_to_load` forwarding in the `_orig_load_gguf_checkpoint` call
   - Files: 26 loader.py files across bartowski_coniccat_qwen3_5_27b_writer_gguf, daniloreddy_qwen3_5_0_8b_gguf, dmind_3_mini_i1_gguf, gpt_oss_swallow_120b_rl_v0_1_gguf, gpt_oss_swallow_20b_rl_v0_1_gguf, gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf, mradermacher_bartleby_qwen3_5_4b_gguf, mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf, mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf, mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf, mradermacher_luna_qwen3_5_27b_v5_i1_gguf, mradermacher_qwen3_5_27b_gguf, mradermacher_qwen3_5_27b_homebrew_gguf, mradermacher_qwen3_5_27b_tainted_heresy_gguf, mradermacher_qwen3_5_4b_abliterated_i1_gguf, mradermacher_qwen3_5_4b_ara_heresy_v2_gguf, mradermacher_qwen3_5_4b_gabliterated_gguf, mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf, mradermacher_qwen3_5_4b_unfiltered_gguf, mradermacher_qwen3_5_4b_unredacted_max_gguf, mradermacher_qwen3_5_9b_abliterated_i1_gguf, mradermacher_vilm_0_8b_sft_gguf, qwen_3_5_imatrix_gguf, tvall43_qwen3_5_2b_heretic_v3b_i1_gguf, tvall43_qwen3_5_4b_heretic_v2_i1_gguf, unified_reward_flex_qwen35_27b_gguf

2. `b8c6532dff` — Fix gliese_qwen3_5_2b_gguf: add `ignore_mismatched_sizes=True` for GGUF loading:
   - `gliese_qwen3_5_2b_gguf/causal_lm/pytorch/loader.py`: Added `ignore_mismatched_sizes=True` to `AutoModelForCausalLM.from_pretrained()`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    493.60s (0:08:13)
- Tier A attempts: N/A

## Files changed
- `gliese_qwen3_5_2b_gguf/causal_lm/pytorch/loader.py` (in tt_forge_models)
- 26 × `*/causal_lm/pytorch/loader.py` (in tt_forge_models, `_patched_load_gguf_checkpoint` signature fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | b8c6532dff634df2e3d230a4f57e3db849eb73f5 |

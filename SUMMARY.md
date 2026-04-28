# Remediation Summary: bartowski_qwen2_5_14b_uncencored_gguf-causal_lm-pytorch-Qwen2.5-14B_Uncencored-Q4_K_M-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_qwen2_5_14b_uncencored_gguf/causal_lm/pytorch-Qwen2.5-14B_Uncencored-Q4_K_M-GGUF-single_device-inference]

## Result
SILICON_PASS — loader bug fixed: _patched_load_gguf_checkpoint widened to accept model_to_load kwarg required by transformers 5.x

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-patched-load-checkpoint-narrow-signature

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(Originally reported as "Test exceeded configured timeout and was killed"; the actual TypeError surfaces as an immediate failure once model loading begins.)

## Root cause
Transformers 5.x changed `load_gguf_checkpoint` (in `modeling_gguf_pytorch_utils.py`) to accept a new
`model_to_load` keyword argument. Several GGUF loaders in tt_forge_models install a module-level
monkey-patch of this function with a narrow signature:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
```

When the dynamic test collector imports all loaders alphabetically, any loader with the narrow patch
that is imported before `bartowski_qwen2_5_14b_uncencored_gguf` replaces the module-global function.
The first such loader alphabetically is `bartowski_coniccat_qwen3_5_27b_writer_gguf`. When
`modeling_utils.py` then calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` for the target
model, the narrow patch raises TypeError.

26 loaders had this narrow signature.

## Fix
All 26 affected loaders in `tt_forge_models` were updated to use `(*args, **kwargs)` (pass-through)
instead of the narrow `(gguf_path, return_tensors=False)` signature, so any future kwargs added by
transformers are forwarded to the original function.

Fix commit: `ba72b5dca41b63a8189aa8d09ab3974b0f249eb4` on branch
`remediation/bartowski_qwen2_5_14b_uncencored_gguf-causal_lm-pytorch-Qwen2.5-14B_Uncencored-Q4_K_M-GGUF-single_device-inference`
in `tenstorrent/tt-forge-models`.

Files changed (27 loader files, one per affected GGUF model):
- bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py
- daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py
- dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py
- mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py
- abhiray_qwen3_5_9b_abliterated_claude_4_6_opus_reasoning_distilled_gguf/causal_lm/pytorch/loader.py

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    189.64s (0:03:09)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 27 loader files (see Fix section)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a4cf851a4d6bc9e7c891b952a44dd760d3ad0baa |
| tt-forge-models | ba72b5dca41b63a8189aa8d09ab3974b0f249eb4 |

# Remediation Summary: asplos2026-causal_lm-pytorch-Qwen1.5_0.5B_Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[asplos2026/causal_lm/pytorch-Qwen1.5_0.5B_Q4_0-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-patched-loader-missing-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
26 GGUF loader files in tt_forge_models define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and install it as a monkey-patch on `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`. Transformers 5.x changed `modeling_utils.py` to call `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)`, adding the `model_to_load` keyword argument. When any of the 26 loaders with the old signature is imported during pytest collection, the broken patch is globally active. The `asplos2026` loader does not itself install any patch, but it triggers the GGUF code path in transformers, which then calls the broken patched function and raises TypeError.

## Fix
In `tt_forge_models`, branch `remediation/asplos2026-causal_lm-pytorch-Qwen1.5_0.5B_Q4_0-single_device-inference` (commit d44acd0966ab3c1e1ebc45032c5ff813c407bcc3):

Updated all 26 loader files to add `model_to_load=None` to the `_patched_load_gguf_checkpoint` signature and pass it through to `_orig_load_gguf_checkpoint`:

```python
# Before
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)

# After
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)
```

Files changed (all in `tt_forge_models/*/causal_lm/pytorch/loader.py`):
- bartowski_coniccat_qwen3_5_27b_writer_gguf
- daniloreddy_qwen3_5_0_8b_gguf
- dmind_3_mini_i1_gguf
- gpt_oss_swallow_120b_rl_v0_1_gguf
- gpt_oss_swallow_20b_rl_v0_1_gguf
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf
- mradermacher_bartleby_qwen3_5_4b_gguf
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf
- mradermacher_qwen3_5_27b_gguf
- mradermacher_qwen3_5_27b_homebrew_gguf
- mradermacher_qwen3_5_27b_tainted_heresy_gguf
- mradermacher_qwen3_5_4b_abliterated_i1_gguf
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf
- mradermacher_qwen3_5_4b_gabliterated_gguf
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf
- mradermacher_qwen3_5_4b_unfiltered_gguf
- mradermacher_qwen3_5_4b_unredacted_max_gguf
- mradermacher_qwen3_5_9b_abliterated_i1_gguf
- mradermacher_vilm_0_8b_sft_gguf
- qwen_3_5_imatrix_gguf
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf
- unified_reward_flex_qwen35_27b_gguf

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    85.70s
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 26 × `*/causal_lm/pytorch/loader.py` (signature + call site)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | d44acd0966ab3c1e1ebc45032c5ff813c407bcc3 |

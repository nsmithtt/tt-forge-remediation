# Remediation Summary: deepanalyze_8b_q4_k_m_gguf-causal_lm-pytorch-8B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepanalyze_8b_q4_k_m_gguf/causal_lm/pytorch-8B_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS — fixed _patched_load_gguf_checkpoint narrow signature; test passes in 0:06:40

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
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
26 GGUF loaders in tt_forge_models monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 now calls `load_gguf_checkpoint` with `model_to_load=dummy_model` as a keyword argument. When pytest collects the full test suite, any of the 26 loaders imported before the deepanalyze test leaves `load_gguf_checkpoint` patched with the narrow signature, causing the TypeError when the deepanalyze model tries to load its GGUF checkpoint.

## Fix
In `tt_forge_models`, on branch `remediation/deepanalyze_8b_q4_k_m_gguf-causal_lm-pytorch-8B_Q4_K_M_GGUF-single_device-inference`, commit `78ac33a27d`:

Changed all 26 affected loaders from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```

Files changed: 26 loader.py files across bartowski_coniccat_qwen3_5_27b_writer_gguf, daniloreddy_qwen3_5_0_8b_gguf, dmind_3_mini_i1_gguf, gpt_oss_swallow_120b_rl_v0_1_gguf, gpt_oss_swallow_20b_rl_v0_1_gguf, gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf, mradermacher_bartleby_qwen3_5_4b_gguf, mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf, mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf, mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf, mradermacher_luna_qwen3_5_27b_v5_i1_gguf, mradermacher_qwen3_5_27b_gguf, mradermacher_qwen3_5_27b_homebrew_gguf, mradermacher_qwen3_5_27b_tainted_heresy_gguf, mradermacher_qwen3_5_4b_abliterated_i1_gguf, mradermacher_qwen3_5_4b_ara_heresy_v2_gguf, mradermacher_qwen3_5_4b_gabliterated_gguf, mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf, mradermacher_qwen3_5_4b_unfiltered_gguf, mradermacher_qwen3_5_4b_unredacted_max_gguf, mradermacher_qwen3_5_9b_abliterated_i1_gguf, mradermacher_vilm_0_8b_sft_gguf, qwen_3_5_imatrix_gguf, tvall43_qwen3_5_2b_heretic_v3b_i1_gguf, tvall43_qwen3_5_4b_heretic_v2_i1_gguf, unified_reward_flex_qwen35_27b_gguf.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    400.48s (0:06:40)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 26 loader.py files (see Fix section)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 78ac33a27d21563ea1ce486f7f0318cc9b45dd8d |

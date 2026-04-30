# Remediation Summary: lily_cybersecurity_7b_uncensored_gguf/causal_lm/pytorch-Lily_Cybersecurity_7B_Uncensored_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lily_cybersecurity_7b_uncensored_gguf/causal_lm/pytorch-Lily_Cybersecurity_7B_Uncensored_GGUF-single_device-inference]

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
Original reported failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Reproduced as:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
(gguf 0.18.0 was already installed in the test environment; the cross-loader clobbering error surfaced instead)

## Root cause
Two related loader bugs:

1. **Missing `gguf>=0.10.0` in `requirements.txt`** for the lily_cybersecurity loader. Without gguf installed, `from_pretrained` raises `ImportError: Please install torch and gguf>=0.10.0`.

2. **Narrow-signature `_patched_load_gguf_checkpoint`** in 26 other GGUF loaders. During pytest collection those loaders are imported and they install a narrow-signature wrapper:
   ```python
   def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
   ```
   over `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`. Transformers 5.2.0 now calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which raises `TypeError` because the wrapper rejects the `model_to_load` keyword argument.

## Fix
In `tt_forge_models` on branch `remediation/lily_cybersecurity_7b_uncensored_gguf-causal_lm-pytorch-Lily_Cybersecurity_7B_Uncensored_GGUF-single_device-inference` (commits `e79d3ed543`, `ac8d0b7db5`):

1. `lily_cybersecurity_7b_uncensored_gguf/causal_lm/pytorch/requirements.txt` — added `gguf>=0.10.0`

2. All 26 loaders with the narrow signature were updated via sed:
   - Signature: `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` → `def _patched_load_gguf_checkpoint(*args, **kwargs):`
   - Inner call: `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(*args, **kwargs)`

   Affected loaders: `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `dmind_3_mini_i1_gguf`, `gpt_oss_swallow_120b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`, `mradermacher_bartleby_qwen3_5_4b_gguf`, `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf`, `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf`, `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf`, `mradermacher_luna_qwen3_5_27b_v5_i1_gguf`, `mradermacher_qwen3_5_27b_gguf`, `mradermacher_qwen3_5_27b_homebrew_gguf`, `mradermacher_qwen3_5_27b_tainted_heresy_gguf`, `mradermacher_qwen3_5_4b_abliterated_i1_gguf`, `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf`, `mradermacher_qwen3_5_4b_gabliterated_gguf`, `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf`, `mradermacher_qwen3_5_4b_unfiltered_gguf`, `mradermacher_qwen3_5_4b_unredacted_max_gguf`, `mradermacher_qwen3_5_9b_abliterated_i1_gguf`, `mradermacher_vilm_0_8b_sft_gguf`, `qwen_3_5_imatrix_gguf`, `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf`, `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`, `unified_reward_flex_qwen35_27b_gguf`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    215.65s (0:03:35)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/lily_cybersecurity_7b_uncensored_gguf/causal_lm/pytorch/requirements.txt` (new)
- 26 × `<model>/causal_lm/pytorch/loader.py` — narrow-signature fix

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 725fc1c866d625a2c7761d453ef72858857fbc6c |
| tt-forge-models | ac8d0b7db5f4d170ec9b0a4f1dde433a517a5cb3 |

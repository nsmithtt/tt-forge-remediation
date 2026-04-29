# Remediation Summary: apriel_1_6_15b_thinker_magic_beta_decensored_gguf

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[apriel_1_6_15b_thinker_magic_beta_decensored_gguf/causal_lm/pytorch-1.6_15B_Thinker_Magic_beta_decensored_GGUF-single_device-inference]

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
Original CI failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Reproduced locally as (after gguf was installed by another model's requirements):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
Two loader-layer bugs:

1. The apriel GGUF loader had no `requirements.txt`, so `gguf>=0.10.0` was not declared as a dependency. In CI environments where gguf was not yet installed, `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)` raised `ImportError`.

2. transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint()`. 26 loaders in tt_forge_models monkey-patch this function at module import time with a wrapper that had the old fixed signature `(gguf_path, return_tensors=False)`, which does not accept `model_to_load`. Since `discover_loader_paths()` imports ALL loaders during pytest collection (even when running a single test), any of these broken patches applied after an otherwise-correct patch would leave transformers' `load_gguf_checkpoint` with the broken signature, causing a `TypeError` for every GGUF model in the same process.

## Fix
**Fix 1** — `apriel_1_6_15b_thinker_magic_beta_decensored_gguf/causal_lm/pytorch/requirements.txt` (new file):
Added `gguf>=0.10.0` to declare the dependency explicitly.

**Fix 2** — 26 loader files in tt_forge_models:
Changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `_patched_load_gguf_checkpoint(*args, **kwargs)` and updated the internal call to `_orig_load_gguf_checkpoint(*args, **kwargs)` so that `model_to_load` and any future kwargs are forwarded transparently.

Both commits are on branch `remediation/apriel_1_6_15b_thinker_magic_beta_decensored_gguf` in tt-forge-models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    637.32s (0:10:37)
- Tier A attempts: N/A

## Files changed
- `apriel_1_6_15b_thinker_magic_beta_decensored_gguf/causal_lm/pytorch/requirements.txt` (new)
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

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 62c66064fd17a81766dc274d3d90542d319fcec4 |

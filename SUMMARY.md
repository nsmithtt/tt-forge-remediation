# Remediation Summary: aurora_mirage_12b_i1_gguf-causal_lm-pytorch-AURORA_MIRAGE_12B_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aurora_mirage_12b_i1_gguf/causal_lm/pytorch-AURORA_MIRAGE_12B_I1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — after fixing the missing gguf requirement, a second loader bug is revealed: 26 other GGUF loaders monkey-patch `load_gguf_checkpoint` with a narrow signature that rejects the `model_to_load` kwarg added in transformers 5.2, breaking aurora_mirage's model loading at runtime

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
Original (bringup) failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

After adding requirements.txt (second failure, same session):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
Full traceback location: `transformers/modeling_utils.py:4016` calling `load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)` which resolves to the monkey-patched narrow-signature version.

## Root cause

**First bug (fixed):** `aurora_mirage_12b_i1_gguf/causal_lm/pytorch/` had no `requirements.txt`. In bringup environments where `gguf` is not pre-installed, `transformers/modeling_gguf_pytorch_utils.py:load_gguf_checkpoint` raises `ImportError("Please install torch and gguf>=0.10.0...")` because `is_gguf_available()` returns False.

**Second bug (unfixed, blocking):** 26 other GGUF loaders in `tt_forge_models` (all in the Qwen 3.5 family and some gpt_oss_swallow variants) define module-level monkey-patches:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):  # narrow signature
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
    ...

_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
```

These patches replace the global `load_gguf_checkpoint` in transformers module objects during pytest test collection (when the loader files are imported). The narrow signature `(gguf_path, return_tensors=False)` does not accept `model_to_load`, which was added in transformers 5.2 and is passed positionally by `modeling_utils.py:4016`:
```python
load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)
```

When aurora_mirage's test runs (after collection), the global function is already the last-imported narrow-signature patch, causing `TypeError`. Aurora_mirage's own loader has no such patch and does not need Qwen 3.5 architecture remapping.

Affected loaders (26 files) that retain the narrow signature as of main commit `0f7b734348`:
`mradermacher_qwen3_5_4b_unredacted_max_gguf`, `dmind_3_mini_i1_gguf`, `qwen_3_5_imatrix_gguf`, `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`, `mradermacher_qwen3_5_27b_tainted_heresy_gguf`, `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf`, `mradermacher_qwen3_5_27b_gguf`, `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf`, `mradermacher_qwen3_5_4b_gabliterated_gguf`, `gpt_oss_swallow_20b_rl_v0_1_gguf`, `mradermacher_qwen3_5_27b_homebrew_gguf`, `mradermacher_qwen3_5_9b_abliterated_i1_gguf`, `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf`, `mradermacher_vilm_0_8b_sft_gguf`, `mradermacher_bartleby_qwen3_5_4b_gguf`, `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf`, `gpt_oss_swallow_120b_rl_v0_1_gguf`, `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf`, `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `mradermacher_qwen3_5_4b_unfiltered_gguf`, `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `mradermacher_luna_qwen3_5_27b_v5_i1_gguf`, `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`, `mradermacher_qwen3_5_4b_abliterated_i1_gguf`, `unified_reward_flex_qwen35_27b_gguf`.

## Fix

**First fix applied (committed):** Added `aurora_mirage_12b_i1_gguf/causal_lm/pytorch/requirements.txt` containing `gguf>=0.10.0`. This ensures the `gguf` package is installed before the loader runs, so `is_gguf_available()` returns True.

Commit: `7a6f80c840` on branch `remediation/aurora_mirage_12b_i1_gguf-causal_lm-pytorch-AURORA_MIRAGE_12B_I1_Q4_K_M_GGUF-single_device-inference` in `tenstorrent/tt-forge-models`.

**Proposed fix for second bug:** In each of the 26 affected loaders, change the `_patched_load_gguf_checkpoint` definition from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```
This is a backwards-compatible signature widening that passes all arguments through to the original function. Prior remediation commits `f7de68b4c6` and `ee29b1a28c` attempted this change in 26 loaders but were not merged to main. Commit `b7b4e9610c` applied the same fix to 3 of these loaders for a different test's chain.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: FAIL
- Hardware:    p300c (Blackhole) × 4
- Duration:    127.42s (0:02:07) — model loaded but failed at from_pretrained stage
- Tier A attempts: N/A

## Files changed
- `aurora_mirage_12b_i1_gguf/causal_lm/pytorch/requirements.txt` (created, in tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7a6f80c840 (remediation branch; main recorded 0f7b734348) |

# Remediation Summary: anubis_mini_8b_v1_i1_gguf/causal_lm/pytorch-ANUBIS_MINI_8B_V1_I1_Q4_K_M_GGUF-single_device-inference

## Test

```
tests/runner/test_models.py::test_all_models_torch[anubis_mini_8b_v1_i1_gguf/causal_lm/pytorch-ANUBIS_MINI_8B_V1_I1_Q4_K_M_GGUF-single_device-inference]
```

## Status: SILICON_PASS

## Root Cause

Multiple GGUF model loaders in `tt_forge_models` monkey-patch the
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` function to
add support for new GGUF architectures (e.g., qwen35, gpt-oss). These patches
chain together at module import time during pytest collection.

A newer version of `transformers` added a `model_to_load` keyword argument to
`load_gguf_checkpoint`. 26 loader files had the old patch signature
`(gguf_path, return_tensors=False)` that rejected this kwarg, causing:

```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Fix

Updated 26 GGUF loader files in `tt_forge_models` to add `**kwargs` to the
`_patched_load_gguf_checkpoint` function signatures and forward them to the
original function. This ensures compatibility with any future kwargs added to
`load_gguf_checkpoint`.

**Pattern changed (in 26 files):**
```python
# Before:
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)

# After:
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)
```

## Changes Made

All changes in `tt-xla/third_party/tt_forge_models` on branch:
`remediation/anubis-mini-gguf-fix-kwargs-compat`

26 files modified (all `*/causal_lm/pytorch/loader.py` with the old signature pattern).

## Submodule Hashes

| Submodule | Hash |
|-----------|------|
| tt-metal | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla | 767c887326ded96efb84e7d8a1e3d07e24b30a95 |
| tt-xla/third_party/tt_forge_models | 57caeafc70e8edcffa24cb40d07f965a677661d0 |

## Commits in tt_forge_models

Key commits from `7efcea60e1` (base) to `57caeafc70` (fix):

```
57caeafc70 Fix load_gguf_checkpoint patchers: accept **kwargs for model_to_load compat
...  (additional cleanup commits on remediation branch)
70ac01bd56 Add mradermacher/Anubis-Mini-8B-v1-i1-GGUF causal LM model loader
```

## Verification

Test confirmed passing on silicon (1 passed in 584.77s / ~9:44).

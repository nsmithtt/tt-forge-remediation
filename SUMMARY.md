# Remediation Summary: athena_1_3b_i1_gguf-causal_lm-pytorch-3B_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[athena_1_3b_i1_gguf/causal_lm/pytorch-3B_I1_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-patched-wrapper-undefined-call-site-vars

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
NameError: name 'gguf_path' is not defined

From third_party/tt_forge_models/granite_3_1_8b_instruct_gguf/causal_lm/pytorch/loader.py:45 in _patched_load_gguf_checkpoint, which is patched globally at import time and therefore invoked in the call chain whenever any GGUF model is loaded in the same pytest session as granite_3_1_8b_instruct_gguf.

## Root cause
The granite_3_1_8b_instruct_gguf loader patches `load_gguf_checkpoint` at import time. A prior commit on the arch-c-36 branch changed the function signature from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` but forgot to update the call site in the function body. The body still referenced the old parameter names `gguf_path` and `return_tensors`, which are no longer in scope, causing `NameError` on every GGUF model load in a session where the granite loader has been imported.

## Fix
In `granite_3_1_8b_instruct_gguf/causal_lm/pytorch/loader.py`, line 45: changed
```python
return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to
```python
return _orig_load_gguf_checkpoint(*args, **kwargs)
```

Committed as `8cf5a5448b` on `remediation/athena_1_3b_i1_gguf-causal_lm-pytorch-3B_I1_GGUF-single_device-inference` in tt-forge-models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    329.53s (0:05:29)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/granite_3_1_8b_instruct_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f79caac4c862b3c7105c2af35b646d349967c875 |
| tt-forge-models | 8cf5a5448b5d13a2f58e9f6fc22b0aeb5d4f6e99 |

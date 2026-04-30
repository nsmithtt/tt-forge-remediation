# Remediation Summary: gemma_2_9b_it_gguf-causal_lm-pytorch-9B_IT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_9b_it_gguf/causal_lm/pytorch-9B_IT_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
granite-gguf-patch-args-not-forwarded

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
NameError: name 'gguf_path' is not defined

(Reported as `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` — the DeprecationWarning appears as the last line of pytest output; the real error is the NameError in the patched load_gguf_checkpoint chain.)

## Root cause
`granite_3_1_2b_instruct_gguf/causal_lm/pytorch/loader.py` monkey-patches
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module
import time. The function signature was partially updated to `(*args, **kwargs)`
but the body was not updated and still referenced bare `gguf_path` and
`return_tensors` variable names:

```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_granite_gguf_support()
    return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```

Because the patch is applied at import time and persists across the entire
pytest session, any subsequent test that triggers `load_gguf_checkpoint`
(including `gemma_2_9b_it_gguf`) runs through this broken wrapper and hits
the `NameError`.

## Fix
One-line fix in `granite_3_1_2b_instruct_gguf/causal_lm/pytorch/loader.py`:

```python
-    return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
+    return _orig_load_gguf_checkpoint(*args, **kwargs)
```

Committed on branch
`remediation/gemma_2_9b_it_gguf-causal_lm-pytorch-9B_IT_GGUF-single_device-inference`
in tt-forge-models (commit edf544056a).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    591.32s (0:09:51)
- Tier A attempts: N/A

## Files changed
- `granite_3_1_2b_instruct_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8fb56e2b05f5b64dd71112a95b75111fab596dd9 |
| tt-forge-models | edf544056a3c0f3b94548401657fc2ea5c1545bc |

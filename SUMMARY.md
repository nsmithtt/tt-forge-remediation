# Remediation Summary: llama_3_2_3b_instruct_uncensored_gguf/causal_lm/pytorch-hungng_Llama_3_2_uncensored_erotica-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_2_3b_instruct_uncensored_gguf/causal_lm/pytorch-hungng_Llama_3_2_uncensored_erotica-single_device-inference]

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
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
During pytest collection, all loader files are imported to discover test variants. 26 GGUF loaders (primarily qwen35 variants) globally replace `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` wrapper that omits the `model_to_load` parameter. When `transformers 5.x` subsequently calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` from `modeling_utils.py:4016` inside `from_pretrained`, it hits the broken patched version and raises `TypeError`.

The `llama_3_2_3b_instruct_uncensored_gguf` loader itself does no patching — it fails purely due to cross-loader global state pollution.

## Fix
Cherry-picked commit `073cb3abb8` from `remediation/darkc0de-xortron-gguf-model_to_load-kwarg` branch into a new `remediation/llama_3_2_3b_instruct_uncensored_gguf-causal_lm-pytorch-hungng_Llama_3_2_uncensored_erotica-single_device-inference` branch in `tt_forge_models`.

The fix adds `model_to_load=None, **kwargs` to the `_patched_load_gguf_checkpoint` signature in all 26 affected loaders and forwards `model_to_load` to `_orig_load_gguf_checkpoint`.

Changed file (26 loaders, same pattern each):
```
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)
```

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    331.07s (0:05:31)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 26 loader files (qwen35 and related GGUF loaders) — `_patched_load_gguf_checkpoint` signature fix

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0fc75d02bf6e8c5f3e7501947b2ea03dca3c4787 |

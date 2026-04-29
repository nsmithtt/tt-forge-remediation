# Remediation Summary: cutelemonlili_qwen2_5_0_5b_instruct_math_training_response_qwen2_5_32b_gguf

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cutelemonlili_qwen2_5_0_5b_instruct_math_training_response_qwen2_5_32b_gguf/causal_lm/pytorch-cutelemonlili_Qwen2.5-0.5B-Instruct_MATH_training_response_Qwen2.5_32B_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS — fixed cross-contamination of _patched_load_gguf_checkpoint with old signature breaking transformers 5.2.0 model_to_load kwarg

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
transformers 5.2.0 added a model_to_load keyword argument to load_gguf_checkpoint(). During pytest test collection, all loader modules are imported. Several Qwen3.5 GGUF loaders (e.g. tvall43_qwen3_5_4b_heretic_v2_i1_gguf, mradermacher_qwen3_5_*, etc.) patch the global _gguf_utils.load_gguf_checkpoint with a wrapper function that has the old signature (gguf_path, return_tensors=False). When the cutelemonlili test runs later in the same session, transformers calls the globally-patched function with the new model_to_load=dummy_model kwarg, which raises TypeError. The cutelemonlili loader itself does not patch anything -- it is a victim of cross-contamination from 26 other loaders with the old signature. Additionally, the cutelemonlili model directory was missing its requirements.txt declaring gguf>=0.10.0.

## Fix
Two changes in tt_forge_models:
1. Updated all 26 _patched_load_gguf_checkpoint functions with the old explicit signature (gguf_path, return_tensors=False) to use (*args, **kwargs) and forward via _orig_load_gguf_checkpoint(*args, **kwargs). This was based on the existing branch remediation/drt-7b-i1-gguf-gguf-load-checkpoint-model-to-load-kwarg.
2. Added cutelemonlili_qwen2_5_0_5b_instruct_math_training_response_qwen2_5_32b_gguf/causal_lm/pytorch/requirements.txt containing gguf>=0.10.0.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    4m14s
- Tier A attempts: N/A

## Files changed
- cutelemonlili_qwen2_5_0_5b_instruct_math_training_response_qwen2_5_32b_gguf/causal_lm/pytorch/requirements.txt (added)
- 26 GGUF loaders: _patched_load_gguf_checkpoint(gguf_path, return_tensors=False) changed to (*args, **kwargs)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 23e809598c194e47d745b3b9d4e2aa3dfa53f4b1 |
| tt-forge-models | 7899fb97fbb4261abe1ce2b4587684b9114e2529 |

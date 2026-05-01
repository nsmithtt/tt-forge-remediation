# Remediation Summary: llama_3_2_3b_gguf-causal_lm-pytorch-3B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_2_3b_gguf/causal_lm/pytorch-3B_GGUF-single_device-inference]

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
At pytest collection time, `TorchDynamicLoader.setup_test_discovery` imports every model loader (including 26 qwen35/gpt-oss GGUF loaders) via `get_model_variants`. Each of those loaders monkey-patches `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow-signature wrapper `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. The patch persists for the entire pytest session. When the llama test later runs and `transformers.modeling_utils.from_pretrained` lazily imports `load_gguf_checkpoint` from the module namespace, it gets the narrow-sig patched version. transformers 5.2.0 added `model_to_load` as a new keyword argument and calls it as `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)`, causing the TypeError.

## Fix
Updated 26 loader files in `tt_forge_models` from the narrow signature `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` / `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `def _patched_load_gguf_checkpoint(*args, **kwargs):` / `_orig_load_gguf_checkpoint(*args, **kwargs)`. This makes all wrappers forward-compatible with transformers signature changes.

Files changed: 26 loader.py files across `tvall43_qwen3_5_*`, `mradermacher_qwen3_5_*`, `gpt_oss_swallow_*`, `qwen_3_5_imatrix_gguf`, `dmind_3_mini_i1_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `unified_reward_flex_qwen35_27b_gguf`.

Remediation branch: `remediation/llama_3_2_3b_gguf-causal_lm-pytorch-3B_GGUF-single_device-inference` in tt-forge-models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    325.16s (0:05:25)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 26 × `*/causal_lm/pytorch/loader.py` (narrow-sig fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bf7f278b8082db41e0ebd95dad9fc5dbc2f9b02e |
| tt-forge-models | ebdb03061df6947d3decaa131ec6de5a39045f8d |

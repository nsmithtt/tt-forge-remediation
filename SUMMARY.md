# Remediation Summary: gemma3_12b_glm_heretic_gguf-causal_lm-pytorch-12B_GLM_Heretic_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_12b_glm_heretic_gguf/causal_lm/pytorch-12B_GLM_Heretic_GGUF-single_device-inference]

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

(Original reported error was `RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)` — the slice-clamping fix for that bug was already present in tt-xla/python_package/tt_torch/torch_overrides.py on this branch. The test now fails at model load time for a separate reason.)

## Root cause
Transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`. Twenty-six GGUF loaders in tt_forge_models monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time using a narrow signature `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` that does not accept `**kwargs`. During pytest collection, `TorchDynamicLoader.setup_test_discovery` imports all loader modules, so the first narrow-signature loader encountered replaces the global `load_gguf_checkpoint` with a function that rejects the new `model_to_load` kwarg. This poisoned patch then breaks any GGUF model loaded later in the session, including the gemma3_12b_glm_heretic_gguf loader (which does not itself install a patch).

Additionally, the gemma3_12b_glm_heretic_gguf loader was missing a `requirements.txt` declaring `gguf>=0.10.0`.

## Fix
In `tt_forge_models` on branch `remediation/gemma3_12b_glm_heretic_gguf-causal_lm-pytorch-12B_GLM_Heretic_GGUF-single_device-inference` (commit 2581f49c6d):

1. Changed all 26 narrow-signature `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` definitions to `_patched_load_gguf_checkpoint(*args, **kwargs)` and updated each corresponding `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` call to `_orig_load_gguf_checkpoint(*args, **kwargs)`.

2. Added `gemma3_12b_glm_heretic_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`.

Files changed: 26 loader.py files across various GGUF model directories + 1 new requirements.txt.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    774.97s (0:12:54)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 26 × `*/causal_lm/pytorch/loader.py` (narrow → variadic signature)
- tt_forge_models: `gemma3_12b_glm_heretic_gguf/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aecc3c84820cbe13c6069b1b8500449e1125e0f2 |
| tt-forge-models | 2581f49c6d5a7ac24d44b45f023f5929ab161bac |

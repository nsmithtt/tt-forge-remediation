# Remediation Summary: cat_translate_7b_i1_gguf-causal_lm-pytorch-CAT_Translate_7b_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cat_translate_7b_i1_gguf/causal_lm/pytorch-CAT_Translate_7b_i1_GGUF-single_device-inference]

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
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
26 GGUF loaders in tt_forge_models monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a fixed signature `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. Transformers 5.2.0 added a `model_to_load=None` keyword argument to `load_gguf_checkpoint`, called at `modeling_utils.py:4016` as `load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)`. When pytest imports all model loaders during test collection, one of the 26 broken patchers installs itself as the global `load_gguf_checkpoint`. When the `cat_translate_7b_i1_gguf` loader then calls `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`, transformers dispatches to the broken patcher with the new `model_to_load` kwarg, causing TypeError. The `cat_translate_7b_i1_gguf` loader itself does not patch `load_gguf_checkpoint`; it is a victim of the cross-test contamination from the other loaders.

## Fix
Cherry-picked the fix from `remediation/drt-7b-i1-gguf-gguf-load-checkpoint-model-to-load-kwarg` in tt_forge_models. Two changes per affected file:
1. `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` → `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):`
2. `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)`

Applied to all 26 affected loaders. Also added `cat_translate_7b_i1_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`.

Files: 26 `*/causal_lm/pytorch/loader.py` files + `cat_translate_7b_i1_gguf/causal_lm/pytorch/requirements.txt`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    374.24s (0:06:14)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 26 `*/causal_lm/pytorch/loader.py` files (signature + kwargs forwarding fix)
- tt_forge_models: `cat_translate_7b_i1_gguf/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4ad60c30c5a2d37feef9b6601b3c7ad47f6faa0c |
| tt-forge-models | 012d55f8157a815c8507feb6936abf0912e8244e |

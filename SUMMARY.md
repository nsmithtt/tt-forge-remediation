# Remediation Summary: chimera_4b_sft_i1_gguf-causal_lm-pytorch-4B_SFT_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chimera_4b_sft_i1_gguf/causal_lm/pytorch-4B_SFT_i1_GGUF-single_device-inference]

## Result
SILICON_PASS — loader fixes (missing gguf requirement + stale monkey-patch) resolved the failures

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
The original CI failure was:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

In local reproduction (gguf already installed from previous test sessions):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Both are root-caused to missing `requirements.txt` (gguf not installed in a fresh env) and a stale monkey-patch from other loaders contaminating the session.

## Root cause
Two loader-layer bugs:

1. **Missing `requirements.txt`**: The chimera GGUF loader had no `requirements.txt` declaring `gguf>=0.10.0`. In a fresh environment where no prior GGUF test had installed gguf, `transformers` raises `ImportError("Please install torch and gguf>=0.10.0...")` when it tries to call `load_gguf_checkpoint`.

2. **Stale monkey-patch from other loaders**: `TorchDynamicLoader.discover_loader_paths` imports ALL pytorch loader modules at test-collection time to enumerate variants. 26 other GGUF loaders monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a narrow signature:
   ```python
   def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
   ```
   `transformers` 5.2.0 added a `model_to_load=dummy_model` keyword argument to that call site. The narrow-signature patches raise `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` when chimera's test subsequently calls `AutoModelForCausalLM.from_pretrained()`.

## Fix
Two changes in `tt_forge_models`, remediation branch `remediation/chimera_4b_sft_i1_gguf-causal_lm-pytorch-4B_SFT_i1_GGUF-single_device-inference`:

1. **`chimera_4b_sft_i1_gguf/causal_lm/pytorch/requirements.txt`** (new file):
   ```
   gguf>=0.10.0
   ```

2. **26 GGUF loaders** updated from narrow signature to variadic (cherry-picked from `cf4762d8e4`):
   ```python
   # Before
   def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
       _patch_arch_support()
       return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
   
   # After
   def _patched_load_gguf_checkpoint(*args, **kwargs):
       _patch_arch_support()
       return _orig_load_gguf_checkpoint(*args, **kwargs)
   ```

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    343.40s (0:05:43)
- Tier A attempts: N/A

## Files changed
- `chimera_4b_sft_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- 26 GGUF loader files with narrow `_patched_load_gguf_checkpoint` signature (see commit `865e2ec32b`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 9972c37bf86b6719f3911e6eea5966b613f35152 |

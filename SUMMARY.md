# Remediation Summary: llama_3_8b_magpie_align_sft_v0_1_gguf-causal_lm-pytorch-8B_Magpie_Align_SFT_v0_1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_8b_magpie_align_sft_v0_1_gguf/causal_lm/pytorch-8B_Magpie_Align_SFT_v0_1_GGUF-single_device-inference]

## Result
SILICON_PASS — loader fixed: added gguf>=0.10.0 requirements.txt and wide-sig load_gguf_checkpoint patch to forward model_to_load kwarg

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
Original reported failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Actual failure reproduced on branch (gguf was already installed in venv):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Traceback:
```
venv/lib/python3.12/site-packages/transformers/modeling_utils.py:4016: in from_pretrained
    state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
Two stacked loader bugs:

1. **Missing requirements.txt**: The loader had no `requirements.txt`, so `gguf>=0.10.0` was not guaranteed to be installed. When run in an environment without gguf, transformers raises `ImportError("Please install torch and gguf>=0.10.0...")` before reaching the GGUF loading code.

2. **Global load_gguf_checkpoint patching conflict (transformers 5.x)**: 40+ other GGUF loaders in tt_forge_models monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a narrow-sig wrapper `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. In transformers 5.x, `from_pretrained` now calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`. When any one of those narrow-sig patching loaders is collected by pytest before this test runs, the global `load_gguf_checkpoint` is replaced by the narrow-sig wrapper, causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` when this loader's `load_model()` invokes `AutoModelForCausalLM.from_pretrained()`.

## Fix
Two changes in `tt_forge_models/llama_3_8b_magpie_align_sft_v0_1_gguf/causal_lm/pytorch/`:

1. **`requirements.txt`** (new file): Added `gguf>=0.10.0` to ensure the gguf package is installed before the loader runs.

2. **`loader.py`**: Added two helper functions and a module-level call:
   - `_find_real_load_gguf_checkpoint()`: Traverses the patch chain installed by other loaders to recover the original transformers `load_gguf_checkpoint` function, identified by its source file path. Also walks `_orig_load_gguf_checkpoint` / closure variable chains.
   - `_ensure_gguf_patch()`: Uses the recovered real function to install a wide-sig `_patched_load_gguf_checkpoint(*args, **kwargs)` wrapper across all four binding sites (`_gguf_utils`, `_config_utils`, `_auto_tokenizer`, `_tok_utils`). The wrapper has no arch-patching logic — it simply forwards all arguments to the real function.
   - Called at module import time (`_ensure_gguf_patch()` at module level) and at the start of `load_model()` to win any ordering race with later-collected narrow-sig loaders.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    454.56s (0:07:34)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/llama_3_8b_magpie_align_sft_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/llama_3_8b_magpie_align_sft_v0_1_gguf/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 62bcef7b09a2ee1191f72e33fe8f8cb538580cc3 |
| tt-forge-models | daadfcc98acd2a9b00d30392c540889c557d61ba |

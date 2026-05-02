# Remediation Summary: llama_2_13b_chat_gguf-causal_lm-pytorch-13B_Chat_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_2_13b_chat_gguf/causal_lm/pytorch-13B_Chat_GGUF-single_device-inference]

## Result
SILICON_PASS — three loader bugs fixed; test passes on n150 in 358s

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
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```
(In the local environment with gguf already installed, the same session contamination manifested as:
`TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`)

## Root cause
Three loader bugs:

1. **Missing `gguf>=0.10.0` requirement**: The `llama_2_13b_chat_gguf` loader had no `requirements.txt`, so `gguf` was not installed in CI, causing the `ImportError` from `transformers.modeling_utils.load_gguf_checkpoint`.

2. **Narrow-sig `_patched_load_gguf_checkpoint` contamination**: 26 qwen35 GGUF loaders define a narrow-signature wrapper `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and install it globally over `transformers.modeling_utils.load_gguf_checkpoint` at module import time. Test collection imports all loaders, so the narrow-sig patch is in place before the llama test runs. `transformers 5.2.0` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` which the narrow-sig patch rejects with `TypeError`.

3. **No chat template fallback**: The loader's `load_inputs` called `tokenizer.apply_chat_template()` unconditionally, but the Llama 2 13B Chat GGUF tokenizer loaded via `gguf_file=` does not embed a chat template, causing `ValueError: tokenizer.chat_template is not set`.

## Fix
All fixes in `tt_forge_models` branch `remediation/llama_2_13b_chat_gguf-causal_lm-pytorch-13B_Chat_GGUF-single_device-inference`:

1. **`llama_2_13b_chat_gguf/causal_lm/pytorch/requirements.txt`** (new file): Added `gguf>=0.10.0`.

2. **26 qwen35 GGUF loader files**: Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):` and updated the inner call to `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)`.

3. **`llama_2_13b_chat_gguf/causal_lm/pytorch/loader.py`**: Added `if self.tokenizer.chat_template is not None:` guard before calling `apply_chat_template`, falling back to `sample_text` directly.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    358.53s (0:05:58)
- Tier A attempts: N/A

## Files changed
- `llama_2_13b_chat_gguf/causal_lm/pytorch/requirements.txt` (new)
- `llama_2_13b_chat_gguf/causal_lm/pytorch/loader.py`
- 26 × `<qwen35_loader>/causal_lm/pytorch/loader.py` (`_patched_load_gguf_checkpoint` **kwargs fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ada4faa91b505ba5a19c0e44f005fbd9d11192bc |
| tt-forge-models | 17c353d015a39f0af89ebcdd07102b8a6b4b709e |

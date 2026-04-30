# Remediation Summary: gemma_2_gguf-causal_lm-pytorch-lmstudio_community_gemma_2_2b_it_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_gguf/causal_lm/pytorch-lmstudio_community_gemma_2_2b_it_GGUF-single_device-inference]

## Result
SILICON_PASS — two loader/frontend bugs fixed; test passes on n150

## Stack layer
loader, tt-xla

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-22, 21], but got -4095)

(Proximate failure on reproduction was `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` due to cross-loader clobbering; the reported slice error is the second bug, exposed after fixing the loader.)

## Root cause
Two independent bugs:

1. **Loader (cross-loader clobbering):** 26 GGUF loaders in tt_forge_models monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow-signature wrapper `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. Transformers 5.2.0 now calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which the narrow wrapper rejects with `TypeError`. The patch is installed at module import time and persists for the whole pytest session, so even models without their own patch (like the gemma_2_gguf loader) are affected.

2. **tt-xla frontend (TorchFunctionOverride):** Gemma 2 uses sliding-window attention with `sliding_window=4096`. On the short test input (22 tokens), `cache_utils.py` calls `full_key_states[:, :, -4095:, :]` where dim-size is 22. PyTorch CPU silently clamps this to `[-22, 21]`, but the XLA backend validates the range and raises `RuntimeError: Value out of range (expected to be in range of [-22, 21], but got -4095)`.

## Fix
1. **tt_forge_models** (`remediation/gemma_2_gguf-causal_lm-pytorch-lmstudio_community_gemma_2_2b_it_GGUF-single_device-inference`): Cherry-picked commit `6491a412039ca5a014848a4c4ca1408b25f3b6d9` — updated all 26 narrow-signature `_patched_load_gguf_checkpoint` wrappers to use `*args, **kwargs` so future transformers kwarg additions are forwarded transparently.

2. **tt-xla** (`remediation/gemma_2_gguf-causal_lm-pytorch-lmstudio_community_gemma_2_2b_it_GGUF-single_device-inference`): Added a guard in `TorchFunctionOverride.__torch_function__` in `python_package/tt_torch/torch_overrides.py` to clamp `aten.slice.Tensor` start indices below `-dim_size` to `-dim_size`, matching PyTorch CPU eager semantics.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    379.14s (0:06:19)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models`: 26 GGUF loader files — `_patched_load_gguf_checkpoint` signature `(gguf_path, return_tensors=False)` → `(*args, **kwargs)`
- `tt-xla/python_package/tt_torch/torch_overrides.py` — slice start clamp in `TorchFunctionOverride.__torch_function__`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d3528eb4556c5baa22ebe8dede9ae13ef9ba4260 |
| tt-forge-models | 6491a412039ca5a014848a4c4ca1408b25f3b6d9 |

# Remediation Summary: medgemma_gguf-causal_lm-pytorch-4B_IT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[medgemma_gguf/causal_lm/pytorch-4B_IT_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

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
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
Two independent bugs:

**Bug 1 (loader):** Other GGUF loaders in the pytest session (e.g., `qwen_3_5_imatrix_gguf`) patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a function that drops the `model_to_load` kwarg added in transformers 5.x. When `medgemma_gguf.load_model()` calls `AutoModelForCausalLM.from_pretrained()`, transformers internally calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which fails because the globally-patched version doesn't accept that kwarg.

**Bug 2 (tt-xla frontend):** Gemma3's sliding-window KV cache uses `full_value_states[:, :, -self.sliding_window + 1:, :]` with `sliding_window=1024`, producing a slice start of `-1023`. XLA validates slice indices strictly (must be in `[-size, size]`); on short (128-token) sequences the tensor size is smaller than 1024, so `-1023` is out of range and raises `RuntimeError: Value out of range`.

## Fix
**Loader fix** (`tt_forge_models/medgemma_gguf/causal_lm/pytorch/loader.py`):
- Added `_find_original_load_gguf_checkpoint()` static method that walks the monkey-patch chain via both `__globals__` and `__closure__` cells to find the original transformers function that accepts `model_to_load`.
- Wrapped `from_pretrained()` in a context manager that temporarily restores the original function across all four modules (`modeling_gguf_pytorch_utils`, `configuration_utils`, `tokenization_auto`, `tokenization_utils_tokenizers`), then restores the patched versions afterward.
- Added `chat_template is not None` guard in `load_inputs()`.
- Added `requirements.txt` with `gguf>=0.10.0`.

**Compiler frontend fix** (`tt-xla/python_package/tt_torch/backend/passes.py` + `backend.py`):
- Added `clamp_out_of_range_slice_starts()` FX pass that iterates over `aten.slice.Tensor` nodes, finds ones with negative start indices that are less than `-size`, and clamps them to `-size`. Uses shape metadata (`node.meta["val"].shape`) for the clamping.
- Wired the pass into `torch_pass_pipeline()` after `bypass_assert_tensor_metadata`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    510.54s (0:08:30)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/medgemma_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/medgemma_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e0967efdeb7fd691262ef5db9bb953db699caee9 |
| tt-forge-models | e26bc1bc6d2b3c884060767b53506e6ca89cc1e9 |

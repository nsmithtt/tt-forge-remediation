# Remediation Summary: hans_wesker_1b_i1_gguf-causal_lm-pytorch-HANS_WESKER_1B_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hans_wesker_1b_i1_gguf/causal_lm/pytorch-HANS_WESKER_1B_I1_GGUF-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: GGUF loader model_to_load kwarg + XLA aten.slice.Tensor OOB start clamping

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
RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_29, 2, -511, 9223372036854775807), kwargs = {})

## Root cause
Two bugs stacked:

1. **Loader (tt_forge_models)**: During pytest collection, other GGUF loaders (e.g. `dmind_3_mini_i1_gguf`, `bartowski_coniccat_qwen3_5_27b_writer_gguf`, and ~25 others) patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with narrow signatures `(gguf_path, return_tensors=False)` that don't accept the `model_to_load` kwarg added in transformers 5.x. When `AutoModelForCausalLM.from_pretrained()` is called for hans_wesker it invokes `load_gguf_checkpoint(..., model_to_load=dummy_model)` through the broken patch, raising `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

2. **tt-xla compiler frontend**: Gemma3 uses a sliding-window KV cache that stores `full_value_states[:, :, -sliding_window+1:, :]`. With `sliding_window=512` and a sequence length of 23, the slice start is -511, which is out of the valid range [-23, 22] for the 23-element tensor. XLA validates slice start strictly and raises `RuntimeError: Value out of range`, whereas PyTorch CPU silently clamps. This manifests during `partition_fx_graph_for_cpu_fallback` in `TorchFunctionOverride.__torch_function__`.

## Fix
**Fix 1 (loader)** — `tt_forge_models/hans_wesker_1b_i1_gguf/causal_lm/pytorch/loader.py`:
Added `_find_real_load_gguf_checkpoint()` which traverses the module-level patch chain (via `__globals__` and closure inspection) to find the actual transformers function defined in `modeling_gguf_pytorch_utils.py`. Added `_ensure_gguf_patch()` which installs a wide-sig `(*args, **kwargs)` wrapper that calls the real function directly. Called in `load_model()` to override any narrow-sig patches installed by earlier-collected loaders.

**Fix 2 (tt-xla)** — `python_package/tt_torch/torch_overrides.py`:
Added a guard in `TorchFunctionOverride.__torch_function__`: when `func is torch.ops.aten.slice.Tensor` and `start < -dim_size`, clamp `start` to `-dim_size` before dispatching. This matches PyTorch CPU semantics and prevents the XLA bounds-check error for sliding-window attention models.

## Verification
- pytest exit: PASS
- Hardware:    n300
- Duration:    322.78s (0:05:22)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/hans_wesker_1b_i1_gguf/causal_lm/pytorch/loader.py` (loader fix)
- `tt_forge_models/hans_wesker_1b_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt-xla/python_package/tt_torch/torch_overrides.py` (slice OOB fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ec4a86b5a6e6ea765e0fb6a5ee5bf3b1d22ea748 |
| tt-forge-models | afbf2c634862ca1af20358d1e74cbcc3d3af7913 |

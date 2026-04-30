# Remediation Summary: gemma_3_12b_it_ultra_uncensored_heretic_i1_gguf-causal_lm-pytorch-12B_IT_ultra_uncensored_heretic_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_12b_it_ultra_uncensored_heretic_i1_gguf/causal_lm/pytorch-12B_IT_ultra_uncensored_heretic_i1_GGUF-single_device-inference]

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
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
(After loader fix, the anticipated second failure was `RuntimeError: Value out of range (expected to be in range of [-33, 32], but got -1023)` from SlidingWindowCache slice OOB — fixed by the tt-xla FX pass applied before reproducing it.)

## Root cause
Two bugs in sequence:

1. **Loader — broken `_patched_load_gguf_checkpoint` chain** (`gguf-load-checkpoint-model-to-load-kwarg`): Other GGUF loaders imported earlier in the pytest session monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow signature `(gguf_path, return_tensors=False)` that does not accept the `model_to_load` kwarg added in transformers 5.x. When this model's `AutoModelForCausalLM.from_pretrained` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, it hits the patched narrow function and raises `TypeError`.

2. **tt-xla — aten.slice.Tensor out-of-bounds start** (`aten-slice-tensor-out-of-bounds-start`): Gemma3 `SlidingWindowCache.update()` computes `start = position - sliding_window + 1`. When `seq_len < sliding_window` (e.g. seq_len=24, sliding_window=1024), this yields `start=-1023` on a dim of size 33, which is outside `[-33, 32]`. PyTorch eager silently clamps out-of-range indices; the XLA lazy backend raises `RuntimeError: Value out of range`. Tier A fix: FX pass + TorchFunctionOverride clamp negative start indices.

## Fix
1. **`tt_forge_models/gemma_3_12b_it_ultra_uncensored_heretic_i1_gguf/causal_lm/pytorch/loader.py`**: Added `_find_real_load_gguf_checkpoint()` BFS helper and `_real_gguf_load_ctx()` context manager that walks the patcher chain to restore the real transformers `load_gguf_checkpoint` during `from_pretrained`. Commit `e0aa633ed2` on tt-forge-models remediation branch.

2. **`tt-xla/python_package/tt_torch/backend/passes.py`** and **`tt-xla/python_package/tt_torch/backend/backend.py`**: Added `clamp_out_of_range_slice_starts` FX pass that iterates over `aten.slice.Tensor` nodes and clamps `start < -dim_size` to `-dim_size`. Cherry-picked from `0ed0567ae`. Commit `f872e5911` on tt-xla remediation branch.

3. **`tt-xla/python_package/tt_torch/torch_overrides.py`**: Added intercept in `TorchFunctionOverride.__torch_function__` for `aten.slice.Tensor` to clamp out-of-range start/end indices. Cherry-picked from `92af2e9e4`. Commit `b5ef160ca` on tt-xla remediation branch.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    757.28s (0:12:37)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/gemma_3_12b_it_ultra_uncensored_heretic_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`
- `tt-xla/python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4928f4f05a49d42e8ad19f03a4a2bb99f3e74e5e |
| tt-forge-models | e0aa633ed2d826d70ad8c79399f2d5b04df5e39e |

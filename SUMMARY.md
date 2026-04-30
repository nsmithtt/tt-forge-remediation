# Remediation Summary: gemma_3_12b_character_creator_v2_gguf-causal_lm-pytorch-12B_Character_Creator_V2_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_12b_character_creator_v2_gguf/causal_lm/pytorch-12B_Character_Creator_V2_GGUF-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-33, 32], but got -1023)

(Reproduction also surfaced a prior bug: TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load')

## Root cause
Two bugs coexisted:

1. **Loader layer**: Other GGUF loaders (alphabetically earlier in test collection) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time using a fixed signature `(gguf_path, return_tensors=False)` that does not accept the `model_to_load` kwarg added in transformers 5.x. When this loader's `from_pretrained` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, the broken patch raises `TypeError`.

2. **tt-xla layer**: Gemma 3 uses `SlidingWindowCache.update()` which slices the cached KV states as `full_value_states[:, :, -sliding_window+1:, :]`. With `sliding_window=1024` and `seq_len=33`, the start index is `-1023`. PyTorch CPU silently clamps indices beyond `[-dim_size, dim_size-1]`, but the XLA/TT backend validates strictly and raises `RuntimeError: Value out of range`.

## Fix
1. **tt_forge_models** (`gemma_3_12b_character_creator_v2_gguf/causal_lm/pytorch/loader.py`): Added `_find_real_load_gguf_checkpoint()` (BFS through the patcher chain via `__globals__['_orig_load_gguf_checkpoint']` and `__closure__` cells to find the real transformers function that accepts `model_to_load`) and `_real_gguf_load_ctx()` context manager that temporarily restores the real function during `from_pretrained`. Commit: `cb872fbec3`.

2. **tt-xla** (`python_package/tt_torch/torch_overrides.py`): Added a guard in `TorchFunctionOverride.__torch_function__` for `func is torch.ops.aten.slice.Tensor` that clamps `start` to `max(-size, start)` when `start < -size`, before dispatching to XLA. This fires before the FX graph is built, preventing the out-of-range error. Commit: `92af2e9e4`.

## Verification
- pytest exit: PASS
- Hardware:    n300
- Duration:    781.24s (0:13:01)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` (slice start clamping in TorchFunctionOverride)
- `tt-xla/third_party/tt_forge_models` (submodule pointer update)
- `tt-xla/third_party/tt_forge_models/gemma_3_12b_character_creator_v2_gguf/causal_lm/pytorch/loader.py` (_real_gguf_load_ctx context manager)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 92af2e9e40006e780afa8b61754f86926afa9431 |
| tt-forge-models | cb872fbec3ec34f278810424572da6a516888f4b |

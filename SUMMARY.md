# Remediation Summary: medgemma_gguf-conditional_generation-pytorch-1.5_4B_IT_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[medgemma_gguf/conditional_generation/pytorch-1.5_4B_IT_Q4_K_M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
transformers-5x-use-fast-default, gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Gemma3ImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Five bugs in two layers:

1. **Loader — gated repo**: `_PROCESSOR_NAME` pointed to `google/medgemma-1.5-4b-it` which is access-restricted. Changed to the public `unsloth/medgemma-1.5-4b-it` mirror.

2. **Loader — transformers 5.x `use_fast` default**: `AutoProcessor.from_pretrained` now defaults to `use_fast=True` for `Gemma3ImageProcessor`. Added `use_fast=False` to preserve behavior.

3. **Loader — GGUF loads text-only, `AutoModelForImageTextToText` fails**: transformers 5.x correctly identifies Gemma3 GGUFs as text-only (`gemma3_text` model_type). `AutoModelForImageTextToText` rejects `Gemma3TextConfig`. Fixed by loading as `Gemma3ForCausalLM`, then composing `Gemma3ForConditionalGeneration` with a full multimodal config (from the public processor repo), and remapping text weights `model.X → model.language_model.X`. Vision tower starts from random init (acceptable for compiler correctness testing).

4. **Loader — GGUF patcher chain breaks `model_to_load` kwarg**: transformers 5.x `from_pretrained` for GGUF files now passes `model_to_load=dummy_model` to `load_gguf_checkpoint`. Approximately 20+ tt_forge_models loaders patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with wrappers that don't accept `model_to_load`. The existing patcher chain traversal (walking `__globals__` for `_orig_load_gguf_checkpoint`) fails in practice. Fixed by using `gc.get_objects()` to scan all live Python objects for the real function (identified by `__module__ == 'transformers.modeling_gguf_pytorch_utils'` and `__name__ == 'load_gguf_checkpoint'`), which works regardless of chain depth or variable naming conventions.

5. **tt-xla — OOB slice start in sliding-window cache**: XLA validates `aten.slice.Tensor` start indices strictly; `start < -size` raises `RuntimeError: Value out of range`. The `StaticSlidingWindowLayer` in transformers computes `full_value_states[:, :, -sliding_window+1:, :]` during prefill. With `sliding_window=1024` but only 276 input tokens, the start is `-1023` which is out of range `[-276, 275]`. PyTorch silently clamps this to 0 but XLA does not. Added `clamp_out_of_range_slice_starts` FX pass in tt-xla `passes.py` that clamps negative slice starts to `-size` in the compiled GraphModule before XLA execution.

## Fix
**tt_forge_models** (`remediation/medgemma_gguf-conditional_generation-pytorch-1.5_4B_IT_Q4_K_M-single_device-inference`):
- `medgemma_gguf/conditional_generation/pytorch/loader.py`: (1) change `_PROCESSOR_NAME` to `unsloth/medgemma-1.5-4b-it`; (2) add `use_fast=False` to `AutoProcessor.from_pretrained`; (3) replace `AutoModelForImageTextToText` with manual `Gemma3ForCausalLM` + weight-copy into `Gemma3ForConditionalGeneration`; (4) add `_find_real_load_gguf_checkpoint()` using `gc.get_objects()` scan + BFS fallback, and `_use_real_load_gguf_checkpoint()` context manager.

**tt-xla** (`remediation/medgemma_gguf-conditional_generation-pytorch-1.5_4B_IT_Q4_K_M-single_device-inference`):
- `python_package/tt_torch/backend/passes.py`: add `clamp_out_of_range_slice_starts` pass
- `python_package/tt_torch/backend/backend.py`: import and call `clamp_out_of_range_slice_starts` after `bypass_assert_tensor_metadata` in `torch_pass_pipeline`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    765.83s (0:12:45)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/medgemma_gguf/conditional_generation/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5288cda62febaad59893715b87a3e02212fd069b |
| tt-forge-models | 7043c7a7d228e5d79c5e62a6c1b632bfcb643550 |

# Remediation Summary: gemma_the_writer_n_restless_quill_10b_gguf-causal_lm-pytorch-10B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_the_writer_n_restless_quill_10b_gguf/causal_lm/pytorch-10B_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-22, 21], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})
Original traceback:
  File ".../transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause

There were two bugs to fix:

**Bug 1 (loader, pre-existing fix):** Three loaders imported before the Gemma writer during test collection — `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, and `dmind_3_mini_i1_gguf` — each globally patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a function missing the `model_to_load` keyword argument added in transformers 5.x. When `AutoModelForCausalLM.from_pretrained` is called with `gguf_file=`, transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` and hits the patched function, causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. This fix was already committed in the tt_forge_models submodule at commit 47242bc804.

**Bug 2 (tt-xla, root cause of the reported error):** Gemma 2's `SlidingWindowCache` slices the KV cache with `full_value_states[:, :, -sliding_window+1:, :]` where `sliding_window=4096`. During first-token compilation the sequence dimension is only 22, so the slice start index `-4095` is far more negative than `-dim_size=-22`. PyTorch on CPU silently clamps this to zero, but XLA's eager execution strictly validates that the start must be within `[-dim_size, dim_size-1]` and raises the `Value out of range` error. The fix clamps the start index to `-dim_size` before passing it to XLA.

## Fix

**Bug 1:** Already fixed in the tt_forge_models submodule (commit 47242bc804): added `model_to_load=None` parameter to `_patched_load_gguf_checkpoint` in all three affected loaders and passed it through to the original function.

**Bug 2:** Added a scoped handler in `TorchFunctionOverride.__torch_function__` in `python_package/tt_torch/torch_overrides.py` that intercepts `aten.slice.Tensor` calls on XLA tensors and clamps the start index to `max(start, -dim_size)` before forwarding to XLA.

Files changed:
- `tt-xla/python_package/tt_torch/torch_overrides.py` (Tier A fix)
- `tt-xla/third_party/tt_forge_models` (submodule pointer update — pre-existing fix)

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    586.67s (0:09:46)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/third_party/tt_forge_models` (submodule pointer)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550 |
| tt-mlir         | 553c0632b |
| tt-xla          | c4a5f4d5a |
| tt-forge-models | d035606776 |

# Remediation Summary: boto_9b_i1_gguf-causal_lm-pytorch-9B_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[boto_9b_i1_gguf/causal_lm/pytorch-9B_I1_GGUF-single_device-inference]

## Result
FAIL — PCC=0.9872 on silicon vs required=0.99; CPU BF16 vs FP32 is 0.9997 so the gap is not BF16 accumulation; compiler-accuracy bug (consteval-on-host precision, tt-xla #1242)

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
RuntimeError: Value out of range (expected to be in range of [-14, 13], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})

Original traceback:
  File "transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Two-layer failure:

**Loader layer:** Multiple other GGUF loaders in tt_forge_models monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time without forwarding `**kwargs`. Transformers 5.2.0 added `model_to_load=None` to this signature; when pytest collects all tests, the chain of patchers causes a TypeError before boto can load. The tt_forge_models remediation branch (`remediation/boto_9b_i1_gguf-causal_lm-pytorch-9B_I1_GGUF-single_device-inference`) had already fixed this (4 prior commits: restore `model_to_load` kwarg, guard callable inspection, snapshot sys.modules, skip `apply_chat_template` for base models).

**Compiler frontend (tt-xla):** Boto-9B is a Gemma-2-based model with `sliding_window=4096`. With seq_len=14, `SlidingWindowCache.update()` emits `full_value_states[:, :, -4095:, :]`. PyTorch silently clamps start indices below `-dim_size` to 0; the XLA/TT backend raises `RuntimeError: Value out of range` because it validates slice bounds strictly. Fix: add `clamp_out_of_range_slice_starts` FX pass in `tt_torch/backend/passes.py` that clamps negative starts to `-dim_size` when `start < -dim_size`.

**Remaining PCC bug:** After the slice fix the model compiles and runs on silicon but produces PCC=0.9872 vs required=0.99. Measured CPU BF16 vs CPU FP32 PCC is 0.9997, so the gap is not BF16 accumulation — it is a compiler-accuracy regression (consteval-on-host precision, tracked in tt-xla #1242). This same PCC floor (~0.987–0.988) is observed across multiple similar Gemma-2-family GGUF models in the test config.

## Fix
**Tier A fix applied (tt-xla):** Added `clamp_out_of_range_slice_starts` pass to `python_package/tt_torch/backend/passes.py` and imported + called it in `python_package/tt_torch/backend/backend.py` after `bypass_assert_tensor_metadata`. Committed to `remediation/boto_9b_i1_gguf-causal_lm-pytorch-9B_I1_GGUF-single_device-inference` in tt-xla.

**Proposed fix for PCC bug (not attempted):** The consteval-on-host precision issue (tt-xla #1242) requires investigation into which operations lose precision when not evaluated on host. This is a cross-cutting accuracy issue affecting many GGUF models and is classified Tier B.

## Tier B justification (FAIL with Tier=A only — omit otherwise)

The Tier A slice fix was successfully applied. The remaining failure is a second compiler-stack bug (PCC accuracy). Per the rules, the first fix is committed and FAIL is filed for the second.

The PCC bug indicator: cross-cutting (consteval-on-host precision affects many models, tracked in #1242 with no clear single-file fix).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    578.96s (0:09:38) on final run with slice fix
- Tier A attempts: 1

## Files changed
**tt-xla** (`remediation/boto_9b_i1_gguf-causal_lm-pytorch-9B_I1_GGUF-single_device-inference`):
- `python_package/tt_torch/backend/passes.py` — add `clamp_out_of_range_slice_starts` function
- `python_package/tt_torch/backend/backend.py` — import and call `clamp_out_of_range_slice_starts` after `bypass_assert_tensor_metadata`

**tt-forge-models** (`remediation/boto_9b_i1_gguf-causal_lm-pytorch-9B_I1_GGUF-single_device-inference`, prior commits):
- `boto_9b_i1_gguf/causal_lm/pytorch/loader.py` — restore `model_to_load` kwarg forwarding, guard callable inspection, snapshot sys.modules, skip `apply_chat_template` for base models

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2f7e9d87e70d0c32b73ff21217a9b29f4992a148 |
| tt-forge-models | 44f7a30eff16001691379d4308e9db7c2c017201 |

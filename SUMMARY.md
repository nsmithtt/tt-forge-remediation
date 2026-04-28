# Remediation Summary: bielik_7b_instruct_v0_1-causal_lm-pytorch-7B_Instruct_v0.1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bielik_7b_instruct_v0_1/causal_lm/pytorch-7B_Instruct_v0.1-single_device-inference]

## Result
FAIL — PCC 0.9787 < 0.99 required after slice fix; WH BF16 matmul accumulation (Tier B, tt-xla #2861)

## Stack layer
tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-wh-bf16-matmul-accumulation

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-15, 14], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})

Original traceback points to transformers/cache_utils.py:214:
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Two issues:

**Issue 1 (fixed — Tier A):** Bielik-7B-Instruct-v0.1 is Mistral-7B-based and uses sliding-window attention with `sliding_window=4096`. On the first forward pass the KV cache has only 15 tokens, so `full_value_states[:, :, -4095:, :]` becomes `aten.slice.Tensor(..., start=-4095, ...)` on a dimension of size 15 (valid range [-15, 14]). PyTorch eager silently clamps this to all 15 elements; the XLA lazy backend raises "Value out of range" instead. Fix: pre-clamp negative start/end to `[-size, size]` in `TorchFunctionOverride.__torch_function__` in `tt-xla/python_package/tt_torch/torch_overrides.py`.

**Issue 2 (unfixed — Tier B):** After the slice fix, the model compiles and runs but PCC is 0.9787 vs 0.99 required. Baseline measurement on CPU shows BF16 vs FP32 PCC = 0.9802, confirming TT has additional precision loss (0.9787) beyond what BF16 accumulation alone gives (0.9802). TT vs CPU BF16 should approach 1.0 if the implementation is faithful; 0.9787 indicates WH BF16 matmul accumulation error, the same root cause as Gemma 7B (PCC ~0.915, tt-xla #2861) and Qwen3 4B (PCC 0.864). The gap is smaller for Bielik-7B (Mistral, intermediate_size=14336) than for Gemma (intermediate_size=24576) or Qwen3 (36 layers), consistent with the error scaling with MLP size and depth.

## Fix
**Issue 1 (applied):** Added `aten.slice.Tensor` pre-clamping to `TorchFunctionOverride.__torch_function__` in `tt-xla/python_package/tt_torch/torch_overrides.py`. When `func is torch.ops.aten.slice.Tensor` and the tensor dimension size is a known int, clamp `start` and `end` to `max(value, -size)` before forwarding the call. Committed and pushed on branch `remediation/bielik_7b_instruct_v0_1-causal_lm-pytorch-7B_Instruct_v0.1-single_device-inference` in tt-xla.

**Issue 2 (proposed fix):** Fix WH BF16 matmul accumulation in tt-mlir. Exact location tracked under tt-xla #2861. Requires changes to the matmul lowering path — see Tier B justification.

## Tier B justification
cross-cutting — The BF16 matmul accumulation error affects every matmul in a 32-layer model. No scoped fix exists; a correct solution requires either implementing FP32 accumulation in the WH matmul kernels or changing the lowering strategy globally, both of which are tracked separately under tt-xla #2861.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    96.86s
- Tier A attempts: 1

## Files changed
- tt-xla/python_package/tt_torch/torch_overrides.py (slice index clamping, Tier A fix — committed)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 08dc56bf9fd277aff6e3844e7605d2f718b33ad0 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

# Remediation Summary: mistral_7b_openorca-causal_lm-pytorch-7b_openorca-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral_7b_openorca/causal_lm/pytorch-7B_OpenOrca-single_device-inference]

## Result
FAIL — Tier A slice fix applied; residual PCC=0.9864 < 0.99 is ttmlir-bf16-matmul-precision-floor (Tier B)

## Stack layer
tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})
Original traceback:
  File "cache_utils.py", line 792, in update
    keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
  File "cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Two bugs:

**Bug 1 (Tier A — fixed):** The XLA/TT backend rejects `aten.slice.Tensor` nodes with
`start < -dim_size`. PyTorch eager silently clamps such start values to 0, but
tt-xla's lazy evaluation raises `RuntimeError: Value out of range`. Mistral's
`SlidingWindowCache.update()` unconditionally slices the key/value cache with
`start = -sliding_window + 1 = -4095`, but the test input is only 128 tokens so
`dim_size = 128`. The condition `-4095 < -128` triggers the error.

**Bug 2 (Tier B — not fixed):** After applying the slice clamp fix, the model runs
to completion but produces PCC = 0.9864617894405052 vs the CPU FP32 reference.
The CPU BF16 vs FP32 baseline for this model and input is PCC = 0.9994, well above
the 0.99 threshold. The additional precision gap (TT BF16 vs CPU FP32 = 0.9864,
vs expected BF16 floor = 0.9994) is the ttmlir-bf16-matmul-precision-floor bug on
BH p150b hardware: TT BF16 matmul accumulates more error than CPU BF16, with the
gap growing for deeper models and larger intermediate dimensions
(Mistral 7B: 32 layers, hidden=4096, intermediate=14336).

## Fix
**Bug 1 fixed in tt-xla** (`python_package/tt_torch/backend/passes.py` and
`python_package/tt_torch/backend/backend.py`):

Added `clamp_out_of_range_slice_starts` FX pass that iterates all
`aten.slice.Tensor` nodes in the compiled graph. For any node with a static
integer `start < 0`, it reads `dim_size` from `input_node.meta["val"].shape[dim]`.
If `start < -dim_size`, it clamps `start` to `-dim_size` to match PyTorch eager
semantics (which treats out-of-range negative indices as equivalent to 0).
The pass is called after `bypass_assert_tensor_metadata` in `torch_pass_pipeline`.

**Proposed fix for Bug 2:** Add F32 accumulation for BF16 matmuls in the TTIR-to-TTNN
lowering layer (`tt-mlir`), or apply math fidelity HiFi4 for affected ops. This is
a cross-cutting change affecting all model matmuls and would require coordinated
changes across multiple files in the compiler.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting
The BF16 matmul precision floor affects every matmul in every layer of the model.
Fixing it requires either raising math fidelity (HiFi4) for all BF16 matmuls in the
TTIR→TTNN lowering, or preserving FP32 intermediate accumulators throughout the
compute graph — both of which are cross-cutting changes spanning multiple files and
passes in tt-mlir.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    165.68s (0:02:45)
- Tier A attempts: 1

## Files changed
- tt-xla: `python_package/tt_torch/backend/passes.py` — added `clamp_out_of_range_slice_starts` FX pass
- tt-xla: `python_package/tt_torch/backend/backend.py` — import and call `clamp_out_of_range_slice_starts` after `bypass_assert_tensor_metadata`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9bb044829cb826a4b6bf79c2f060365d9c719681 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

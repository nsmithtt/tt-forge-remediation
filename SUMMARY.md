# Remediation Summary: llama4-causal_lm-pytorch-tiny_Llama4ForCausalLM-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama4/causal_lm/pytorch-tiny_Llama4ForCausalLM-single_device-inference]

## Result
FAIL — second compiler-stack bug: `aten.slice.Tensor` with start=-8191 rejected as out-of-range [-128, 127]; first bug (0-dim complex tensor) was fixed

## Stack layer
tt-xla

## Tier
B

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
RuntimeError: TT_THROW @ /home/ttuser/hf-bringup/tt-xla/pjrt_implementation/src/api/buffer_instance.cc:282: tt::exception
info:
Complex tensor with num_dims == 0 is not supported.

After fixing the above, a second failure:
RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -8191)
While executing %slice_2 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_1, 2, -8191, 9223372036854775807), kwargs = {})

## Root cause
**First bug (fixed):** `BufferInstance::calculateShape` in `tt-xla/pjrt_implementation/src/api/buffer_instance.cc` threw unconditionally when a complex tensor had `num_dims == 0`. Llama4's `Llama4TextRotaryEmbedding.forward` computes `freqs_cis = torch.polar(ones, freqs)` (complex) then multiplies by `self.attention_scaling` (a Python float). The XLA lazy executor promotes the float scalar to a 0-dim complex tensor to match the complex operand dtype, then tries to transfer that 0-dim complex tensor to the TT device, hitting the explicit throw. The existing complex-expansion path in `calculateShape` (which appends `2` for interleaved real/imag) already handles the general case; removing the guard lets 0-dim complex tensors be represented as shape `{2}`.

**Second bug (unfixed):** `Llama4TextSdpaAttention` uses `SlidingWindowCache` with `sliding_window=8192`. During the forward pass, the cache update does `full_value_states[:, :, -self.sliding_window + 1:, :]` = `full_value_states[:, :, -8191:, :]`. With the tiny test model having only ~128 tokens in the KV cache, the valid index range for that dimension is `[-128, 127]`. Standard PyTorch (CPU) silently clamps the start to 0 and returns the full slice. The XLA backend for `aten.slice.Tensor` instead validates the index against the actual tensor shape and raises when it is out of the clamped range. This is in the tt-xla dispatch path — no normalization of out-of-range slice indices is performed before handing off to the XLA IR builder.

## Fix
**First bug:** Removed the 4-line guard at `buffer_instance.cc:280-283` that threw for 0-dim complex tensors. The remaining code at lines 292-294 (`if (data_type_utils::isComplexPJRTType(data_type)) { shape.push_back(2); }`) already covers the 0-dim case: with no original dimensions, the result is shape `{2}` (real + imag), and `calculateStrides` with `num_dims=0` returns `{}` (empty), consistent with all non-zero complex tensor encodings.

**Second bug (proposed):** Add an `aten.slice.Tensor` custom decomposition in `tt-xla/python_package/tt_torch/backend/decompositions.py` that clamps `start` to `max(start, -dim_size)` and `end` to `min(end, dim_size)` before delegating to the underlying XLA slice. This replicates PyTorch CPU clamping semantics.

## Tier B justification
cross-cutting — `aten.slice.Tensor` index clamping must be applied to every slice call on XLA tensors; this is the second compiler-stack bug in the same test (one-fix-per-report rule prevents attempting it here).

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    34.57s (run with second bug present)
- Tier A attempts: 1 (first bug fixed; second bug not attempted)

## Files changed
- tt-xla: `pjrt_implementation/src/api/buffer_instance.cc` — remove 0-dim complex throw guard

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4f87f844b6c003a53c5d9b4af1819c97d8dd8afe |
| tt-forge-models | 0f7b734348c15e2a7cd43dcf9e1d8b23d34f0b14 |

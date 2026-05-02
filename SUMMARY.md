# Remediation Summary: moody_porn_mix_v9_gguf-pytorch-moodyPornMix_zitV9_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[moody_porn_mix_v9_gguf/pytorch-moodyPornMix_zitV9_Q4_K_M-single_device-inference]

## Result
FAIL — Tier B compiler bug: complex tensor types (view_as_complex / torch.polar) in ZImage RoPE are not supported in the TT MLIR lowering pipeline

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
tt-mlir-complex-type-not-supported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):
```
torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded

from user code:
   File ".../diffusers/quantizers/gguf/utils.py", line 520, in dequantize_gguf_tensor
    tensor = tensor.as_tensor()
  File ".../diffusers/quantizers/gguf/utils.py", line 545, in as_tensor
    return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)
  File ".../diffusers/quantizers/gguf/utils.py", line 564, in __torch_function__
    result = super().__torch_function__(func, types, args, kwargs)
  [Previous line repeated 158 more times]
```

Failure after loader fix (terminal Tier B bug):
```
loc("p2.9"): error: failed to legalize unresolved materialization from
('tensor<512x24x2xf32>') to ('tensor<512x24xcomplex<f32>>') that remained
live after conversion
ValueError: Error code: 13
```

## Root cause

Two bugs found in sequence:

**Bug 1 (loader, fixed):** `diffusers.quantizers.gguf.utils.GGUFParameter.as_tensor()` calls
`torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)` without
`torch._C.DisableTorchFunctionSubclass()`. Under torch.compile/dynamo, the `_make_subclass`
call triggers `GGUFParameter.__torch_function__`, which calls
`super().__torch_function__(func, types, args, kwargs)`. Because `GGUFParameter` is still
in `types`, the dispatch re-enters `GGUFParameter.__torch_function__` — infinite recursion.
The fix wraps the `_make_subclass` call in `DisableTorchFunctionSubclass` context to break
the dispatch cycle.

**Bug 2 (tt-mlir, Tier B):** The ZImage (Lumina2-based) transformer model uses complex
tensor operations in its RoPE (Rotary Position Embedding) computation. Specifically,
`transformer_z_image.py:_prepare_sequence` uses:
- `torch.view_as_complex(x.reshape(..., -1, 2))` — converts pairs of f32 to complex64
- `torch.polar(ones_like(freqs), freqs)` — produces complex64 RoPE frequency tensor
- `x * freqs_cis` — complex multiplication

These ops lower to `stablehlo.complex` / `stablehlo.real` / `stablehlo.imag` in the
XLA representation. The TT MLIR type conversion pipeline lacks lowerings for complex types:
the type conversion pass adds an unrealized materialization cast from `tensor<512x24x2xf32>`
to `tensor<512x24xcomplex<f32>>` that cannot be resolved, causing the legalization pass to
fail with INTERNAL error code 13.

## Fix

**Bug 1 fix** (`moody_porn_mix_v9_gguf/pytorch/src/model_utils.py` in tt_forge_models,
commit `bc7db5e5b1c4db50376346eee7701806ba8b41f4` on remediation branch):
Added `_patch_gguf_parameter_as_tensor()` helper that monkey-patches
`GGUFParameter.as_tensor` to use `DisableTorchFunctionSubclass`, called before
`ZImageTransformer2DModel.from_single_file`. Matches the fix pattern applied to
HunyuanVideo ComfyUI GGUF and Wan2 GGUF loaders.

**Bug 2 proposed fix** (tt-mlir): Add lowering patterns in the StableHLO → TTIR
conversion pass to handle complex tensor ops:
1. `stablehlo.complex(real, imag)` → view of interleaved real pairs (or decompose inline)
2. `stablehlo.real(z)` / `stablehlo.imag(z)` → slice of the 2-channel real tensor
3. Complex multiplication `a * b` → `(ac-bd, ad+bc)` via real arithmetic
4. Materialization callback: `tensor<N x complex<f32>>` ↔ `tensor<N x 2 x f32>`

Alternatively, decompose complex RoPE ops before reaching the type conversion pass:
replace `view_as_complex` + complex multiply + `view_as_real` with the equivalent
real-tensor rotation `(x_r*cos - x_i*sin, x_r*sin + x_i*cos)` using a StableHLO
decomposition pass.

## Tier B justification
Indicator: **new-infrastructure**.

Complex tensor type support (view_as_complex, view_as_real, polar, complex multiply)
is entirely absent from the TT MLIR lowering pipeline. The MLIR type conversion
framework has no callbacks to materialize or de-materialize complex types. Adding
this requires: new op lowering patterns for at least 4 StableHLO complex ops, a type
materialization callback in the conversion target, and testing that the real-tensor
encoding is consistent through the rest of the pipeline. This is more than a single
scoped fix in one named function.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    432.52s (0:07:12) to reach the terminal Tier B failure after Bug 1 fixed
- Tier A attempts: N/A

## Files changed
- `moody_porn_mix_v9_gguf/pytorch/src/model_utils.py` (tt_forge_models remediation branch)
  - `bc7db5e5b1`: Patch GGUFParameter.as_tensor with DisableTorchFunctionSubclass to fix dynamo recursion

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | bc7db5e5b1c4db50376346eee7701806ba8b41f4 |

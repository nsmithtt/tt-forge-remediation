# Remediation Summary: gguf_node_pytorch-aura_flow_0.3_q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gguf_node/pytorch-aura_flow_0.3_q4_0-single_device-inference]

## Result
FAIL — terminal Tier B: gguf-uint8-to-layout-device-untilize (UINT8 on-device layout unsupported in tt-metal)

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
gguf-uint8-to-layout-device-untilize

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded

from user code:
   File "diffusers/quantizers/gguf/utils.py", line 520, in dequantize_gguf_tensor
    tensor = tensor.as_tensor()
  File "diffusers/quantizers/gguf/utils.py", line 545, in as_tensor
    return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)
  File "diffusers/quantizers/gguf/utils.py", line 564, in __torch_function__
    result = super().__torch_function__(func, types, args, kwargs)
  [Previous line repeated 158 more times]

## Root cause
Two bugs in sequence:

1. **Loader (fixed):** diffusers' `GGUFParameter.as_tensor()` calls
   `torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)` without
   `DisableTorchFunctionSubclass`. Since `self` is a `GGUFParameter` subclass,
   `_make_subclass` triggers `__torch_function__` dispatch, which calls
   `super().__torch_function__` — which re-dispatches back to
   `GGUFParameter.__torch_function__` infinitely. Fixed by monkey-patching
   `GGUFParameter.as_tensor` to wrap in `torch._C.DisableTorchFunctionSubclass()`.

2. **tt-metal (Tier B, unfixed):** After the recursion fix, torch.compile traces
   the Q4_0 dequantization path. `dequantize_gguf_tensor` views the quantized
   weight as UINT8 (`tensor.view(torch.uint8)`). tt-mlir places this UINT8 buffer
   on device. `layout_converter.cpp` in tt-metal calls
   `canUntilizeDataTypeOnDevice`, which only allows BF16/F32/UINT32/INT32 — not
   UINT8 — causing a LOG_FATAL that surfaces as PJRT Error code: 13 (kInternal).
   Same class as the FLUX.1-Fill-dev Q4_0 terminal bug.

## Fix
**Loader fix (committed):**
- `tt_forge_models/gguf_node/pytorch/loader.py`: import `GGUFParameter` from
  `diffusers.quantizers.gguf.utils`; monkey-patch `as_tensor` to use
  `torch._C.DisableTorchFunctionSubclass()` before `_make_subclass`.
- Remediation branch: `remediation/gguf_node_pytorch-aura_flow_0.3_q4_0-single_device-inference`
  in `tenstorrent/tt-forge-models` (commit bd85186284a8).

**Proposed compiler fix (not attempted):**
- `tt-metal/ttnn/cpp/ttnn/operations/layout_converter.cpp`: add UINT8 to the set
  of data types supported by `canUntilizeDataTypeOnDevice`. The function currently
  returns false for UINT8, triggering the LOG_FATAL. Supporting UINT8 untilize
  requires ensuring the corresponding kernel handles the element size.
- This is new infrastructure (adding a new DRAM layout data-type path in tt-metal).

## Tier B justification
Indicator: **new-infrastructure**

Adding UINT8 support to `canUntilizeDataTypeOnDevice` and the underlying TTNN
layout kernels is new infrastructure in tt-metal — not a scoped bounds/formula
fix. The same Tier B was filed for FLUX.1-Fill-dev Q4_0 under the fingerprint
`gguf-uint8-to-layout-device-untilize`. All sub-byte GGUF quantization types
(Q4_0, Q4_K, Q5_K, etc.) will hit this or the related
`ttnn-bitcast-cross-size-dtype-unsupported` block after the loader recursion is
fixed.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    725.29s (0:12:05) for second run
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gguf_node/pytorch/loader.py` — monkey-patch GGUFParameter.as_tensor

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7c03193b97d87e7d29f4a30909aab8ff115f688a |
| tt-forge-models | bd85186284a8f2425c6e83ee36819d35aaf7f581 |

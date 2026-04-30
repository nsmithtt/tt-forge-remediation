# Remediation Summary: hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_Q5_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan_video_1_5_i2v_720p_gguf/pytorch-720p_i2v_Q5_K_M-single_device-inference]

## Result
FAIL — Q5_K dequantization performs a cross-size uint8→float16 bitcast (`dmin.view(torch.float16)`) that TT hardware rejects with INTERNAL Error code 13; ttnn::bitcast only supports same-size dtype pairs

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
ttnn-bitcast-cross-size-dtype-unsupported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The test was given with the reported failure message `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`. On hf-bringup-12, which already contains loader fixes for Bugs 1–4 (URL format, config type mismatches, GGUF→diffusers key mapping, timestep dtype), the actual failure observed was:

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

After adding the `GGUFParameter.as_tensor` fix (see Fix section), the terminal failure is:

```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

While executing %view_33 : [num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%slice_10, torch.float16), kwargs = {})
...
  File ".../diffusers/quantizers/gguf/utils.py", line 336, in dequantize_blocks_Q5_K
    dmin = dmin.view(torch.float16).to(dtype)
```

## Root cause

Two distinct issues:

**Bug A (loader) — GGUFParameter.as_tensor dynamo recursion**: During `torch.compile` tracing of `GGUFLinear.forward_native`, `dequantize_gguf_tensor` calls `tensor.as_tensor()`. The diffusers implementation uses `torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)` which, under dynamo, re-enters `GGUFParameter.__torch_function__` → `super().__torch_function__` → `torch.ops.aten.view.dtype` → again dispatches through `tt_torch`'s `__torch_function__` override → back into `GGUFParameter.__torch_function__` infinitely. Fix: wrap `_make_subclass` in `DisableTorchFunctionSubclass` to escape the subclass before dynamo sees the result.

**Bug B (tt-metal) — ttnn::bitcast cross-size dtype unsupported**: Q5_K dequantization in diffusers (`dequantize_blocks_Q5_K`, line 336) does `dmin.view(torch.float16)` which reinterprets a `uint8` tensor's bytes as `float16`. Since `uint8` is 1 byte per element and `float16` is 2 bytes per element, this is a cross-size bitcast. `ttnn::bitcast` only supports same-size reinterpretations. TT hardware returns INTERNAL Error code 13.

## Fix

**Bug A** (implemented, committed to remediation branch): Added `GGUFParameter.as_tensor` patch in `load_model` using `torch._C.DisableTorchFunctionSubclass`:

```python
from diffusers.quantizers.gguf.utils import GGUFParameter

def _safe_as_tensor(self):
    with torch._C.DisableTorchFunctionSubclass():
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)

GGUFParameter.as_tensor = _safe_as_tensor
```

File: `hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py` in `tt-forge-models`.

**Bug B** (not fixed — Tier B): The fix would need to implement cross-size bitcast support in `ttnn` (tt-metal). This would require a new kernel that handles element-count changes (e.g., N uint8 elements reinterpreted as N/2 float16 elements). Alternatively, diffusers' GGUF dequantization paths could be rewritten to avoid the bitcast, but that is a cross-cutting change to the GGUF dequant library.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

ttnn::bitcast in tt-metal only supports reinterpreting between dtype pairs of equal element size. Supporting cross-size bitcasts (uint8→float16, where 2 uint8 elements map to 1 float16 element) requires a new kernel or a tiling/reshape pass. This is new infrastructure work in tt-metal, not a scoped one-function fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    334.26s (0:05:34) — second run after loader fix
- Tier A attempts: N/A

## Files changed
- `hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py` (tt-forge-models remediation branch): added `GGUFParameter.as_tensor` patch with `DisableTorchFunctionSubclass`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 963f154e944d3bf80b4f6fa123fd983af6eea404 |

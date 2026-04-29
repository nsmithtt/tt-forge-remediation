# Remediation Summary: flux2_klein_gguf-pytorch-Klein_9B_KV_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux2_klein_gguf/pytorch-Klein_9B_KV_Q4_K_M-single_device-inference]

## Result
FAIL — after fixing the loader's as_tensor bug, the test hits `ttnn-bitcast-cross-size-dtype-unsupported` (uint8→float16 cross-size bitcast in Q4_K/Q5_K dequantization), a Tier B compiler bug with no implemented fix

## Stack layer
loader, tt-mlir

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
Original (from CI): `torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded`

Reproduced as: `AssertionError: Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'. Found in aten.mm.default(tensor([...], device='xla:0', size=(1, 256), dtype=torch.bfloat16), FakeTensor(..., device='xla:0', size=(256, 4096), dtype=torch.bfloat16))`

After loader fix: `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` at `aten.view.dtype(args=(%slice_2, torch.float16))` inside `dequantize_blocks_Q5_K` → `dmin.view(torch.float16)`

## Root cause

**Loader bug (fixed):** `_patch_gguf_parameter()` patched `GGUFParameter.as_tensor = lambda self: self.data`. On a real `GGUFParameter` tensor, `.data` returns the same `GGUFParameter` object — the subclass is NOT escaped. This means `dequantize_gguf_tensor` still returns a `GGUFParameter` (with the original `quant_type`) even after dequantization; then `F.linear(input, GGUFParameter_dequantized)` triggers `_safe_tf` which calls `_original_tf` and tries to dequantize again. During `torch.export.export` tracing, this manifests as a FakeTensor/real-tensor mismatch: the dequantized weight is captured as a FakeTensor constant in the exported graph, while the activation is a real XLA tensor during `partition_fx_graph_for_cpu_fallback → collector.run`.

**Residual compiler bug (Tier B):** After fixing `as_tensor`, the dequantization ops are traced correctly and reach tt-mlir. The Q4_K_M and Q5_K dequantization uses `tensor.view(torch.float16)` to bitcast uint8 bytes to float16. This is a cross-size reinterpret cast (8-bit → 16-bit). `ttnn::bitcast` requires both sides to have the same element bit width. The INTERNAL error 13 is PJRT's `INTERNAL` status for a non-recoverable operation failure inside the ttnn bitcast kernel. This is the same bug as `Wan2 GGUF Q4_K_M`.

## Fix

**Loader fix (committed):** Changed `GGUFParameter.as_tensor` in `flux2_klein_gguf/pytorch/loader.py`:

```python
# Before (broken):
GGUFParameter.as_tensor = lambda self: self.data  # .data returns GGUFParameter on real tensors

# After (correct):
def _as_tensor(self):
    with torch._C.DisableTorchFunctionSubclass():
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)
GGUFParameter.as_tensor = _as_tensor
```

`torch.Tensor._make_subclass(torch.Tensor, self, ...)` explicitly creates a plain `torch.Tensor` view. `DisableTorchFunctionSubclass` prevents `__torch_function__` re-dispatch under dynamo tracing. Same pattern as the Wan2 GGUF fix (commit `1a832a4c6a` in tt_forge_models).

**Residual Tier B bug (not fixed):** `ttnn-bitcast-cross-size-dtype-unsupported` — Q4_K/Q5_K dequantization uses `view(torch.float16)` (uint8→float16 cross-size bitcast). `ttnn::bitcast` supports only same-size pairs (UINT16↔BFLOAT16, UINT32↔FLOAT32). Implementing cross-size bitcast would require new ttnn kernel logic to handle the element-count change in TT tile layout.

## Tier B justification

Indicator: **new-infrastructure**. Implementing cross-size dtype bitcast in tt-mlir/tt-metal requires a new kernel that handles tile-layout reinterpretation where one element's bit width changes. This is not a scoped formula fix; it requires new ttnn device op infrastructure and cannot be done in one or two files.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 165.80s (after loader fix, before hitting bitcast error)
- Tier A attempts: N/A

## Files changed
- `flux2_klein_gguf/pytorch/loader.py` (tt_forge_models, commit `80ed7f9a9e`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 9ecefa4a6 |
| tt-forge-models | 80ed7f9a9e |

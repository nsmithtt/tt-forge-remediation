# Remediation Summary: hidream_gguf-text_to_image-pytorch-HiDream-I1-Dev-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hidream_gguf/text_to_image/pytorch-HiDream-I1-Dev-single_device-inference]

## Result
FAIL — ttnn-bitcast-cross-size-dtype-unsupported; GGUF Q4_K_S dequantization uses aten.view.dtype across element sizes which has no lowering in tt-mlir

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
Original reported failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Actual failure on reproduce (three sequential errors fixed in turn):

1. `AttributeError: 'ModelLoader' object has no attribute '_GGUF_FILES'` — class attribute defined in code comments but never declared
2. `torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded` — GGUFParameter.as_tensor() infinite __torch_function__ recursion under dynamo
3. Terminal:
```
view_3 = torch.ops.aten.view.dtype(optimized_mod_2, torch.float32)
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

Three sequential issues:

1. **Loader: missing `_GGUF_FILES`** — `hidream_gguf/text_to_image/pytorch/loader.py` references `self._GGUF_FILES` (for both `load_model` and `load_inputs`) but the class attribute was never declared. Also, `_VARIANTS` used the wrong `pretrained_model_name` values (`HiDream-ai/HiDream-I1-{Full,Dev}` instead of the GGUF repos `city96/HiDream-I1-{Full,Dev}-gguf`).

2. **Loader: GGUFParameter.as_tensor() recursion** — `diffusers.quantizers.gguf.utils.GGUFParameter.as_tensor()` calls `torch.Tensor._make_subclass(torch.Tensor, self, ...)`. Under torch dynamo tracing, `_make_subclass` is dispatched through `__torch_function__` on `GGUFParameter` (because `self` is a GGUFParameter). `__torch_function__` calls `super().__torch_function__()` (nn.Parameter → torch.Tensor → GGUFParameter), creating an infinite cycle until Python's recursion limit.

3. **Tier B compiler bug: aten.view.dtype cross-size** — After fixing the loader, the GGUF Q4_K_S dequantization path in `GGUFLinear.forward_native` emits `aten.view.dtype(tensor, torch.float32)` (interpreting packed uint8 block data as float32 scale values). This cross-element-size dtype view has no lowering in tt-mlir, producing `INTERNAL: Error code: 13` (PJRT kInternal). Same class as the `ttnn-bitcast-cross-size-dtype-unsupported` bug documented for Wan2 Q4_K_S GGUF and HunyuanVideo ComfyUI GGUF.

## Fix

**Loader fixes** (both committed to `remediation/hidream_gguf-text_to_image-pytorch-HiDream-I1-Dev-single_device-inference` in tt-forge-models):

- `hidream_gguf/text_to_image/pytorch/loader.py`:
  - Fixed `pretrained_model_name` for both variants to point to city96 GGUF repos (`city96/HiDream-I1-{Full,Dev}-gguf`)
  - Added missing `_GGUF_FILES` class attribute mapping variants to GGUF filenames (`hidream-i1-{full,dev}-Q4_K_S.gguf`)
  - Patched `GGUFParameter.as_tensor` with `DisableTorchFunctionSubclass` to prevent infinite `__torch_function__` recursion under dynamo
- `hidream_gguf/text_to_image/pytorch/requirements.txt`: added `gguf>=0.10.0`

**Proposed Tier B fix** (not implemented): Implement a lowering for `aten.view.dtype` with cross-element-size types in tt-mlir, or add a decomposition in the tt-xla frontend that rewrites the cross-size view into supported ops. This would also benefit all other GGUF Q4_K/Q5_K diffusers models.

## Tier B justification
The `aten.view.dtype` cross-element-size operation (e.g., uint8 → float32, where sizeof(float32) = 4 × sizeof(uint8)) requires either new infrastructure to lower this as a reshape + bitcast pair, or a new decomposition in the tt-xla GGUF-specific path. This is new-infrastructure work affecting the MLIR lowering layer.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    429.78s (0:07:09) per run
- Tier A attempts: N/A

## Files changed
- `hidream_gguf/text_to_image/pytorch/loader.py` (tt-forge-models)
- `hidream_gguf/text_to_image/pytorch/requirements.txt` (tt-forge-models, new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ea4a6822193ebe316b1e0864ab2e0baaa1957a8a |

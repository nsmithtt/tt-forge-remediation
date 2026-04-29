# Remediation Summary: calcuis_wan2_gguf-pytorch-2.2_I2V_HighNoise_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[calcuis_wan2_gguf/pytorch-2.2_I2V_HighNoise_Q4_K_M-single_device-inference]

## Result
FAIL — aten.view.dtype (uint8→float16 bitcast) not supported by ttnn::bitcast; Q4_K_M GGUF dequantization requires cross-size bitcast that TT hardware cannot execute

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
Original reported failure: `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`

Actual reproduced failures (in order):
1. `NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty()` — dispatch_model fails because Wan 2.1 config fields (added_kv_proj_dim, image_dim) create 208 meta-tensor parameters not present in the Wan 2.2 GGUF.
2. `torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded` — GGUFParameter.__torch_function__ calls super().__torch_function__ which under torch.compile/dynamo recurses back into GGUFParameter.__torch_function__ instead of torch.Tensor.__torch_function__.
3. `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` while executing `%view_3 = aten.view.dtype(%slice_2, torch.float16)` — ttnn::bitcast does not support cross-size dtype conversions; UINT8 (8-bit) → FLOAT16 (16-bit) is not in the supported pairs (UINT16↔BFLOAT16, UINT32↔FLOAT32, UINT32↔INT32).

## Root cause

Three bugs, two fixed, one Tier B:

**Bug 1 (loader, fixed)**: The Wan 2.2 I2V GGUF omits the image-conditioning modules (cross-attention add projections and image embedder) present in the Wan 2.1 I2V diffusers config. `WanTransformer3DModel.from_single_file` uses `init_empty_weights` (meta device) to construct the model from config, then `load_model_dict_into_meta` to populate from the GGUF checkpoint. Parameters created from `added_kv_proj_dim` and `image_dim` config fields (208 parameters) are not in the GGUF checkpoint, so they remain as meta tensors. `dispatch_model(model, device_map={"": "cpu"})` then calls `model.to("cpu")`, hitting these meta tensors and raising NotImplementedError.

**Bug 2 (loader, fixed)**: `GGUFParameter.__torch_function__` calls `super().__torch_function__(func, types, args, kwargs)` to delegate to torch.Tensor. Under torch.compile/dynamo, this dispatches back to `GGUFParameter.__torch_function__` instead of using `DisableTorchFunctionSubclass()` to call the function directly, causing infinite recursion. The `as_tensor()` method is specifically designed to return a plain tensor (its comment says "avoid __torch_function__ overhead"), but `__torch_function__` wraps the result back into GGUFParameter anyway.

**Bug 3 (tt-metal, Tier B)**: The GGUF Q4_K_M dequantization kernel (`dequantize_blocks_Q4_K`) reinterprets pairs of uint8 bytes as float16 values via `tensor.view(torch.float16)`. This lowers to `stablehlo.bitcast_convert` → `ttnn.bitcast_convert` → `ttnn::bitcast`. The tt-metal bitcast documentation explicitly states: "Must have the same bit size as input dtype. Supported pairs: UINT16↔BFLOAT16 (both 16 bits), UINT32↔FLOAT32 (both 32 bits), UINT32↔INT32 (both 32 bits)." UINT8→FLOAT16 is a cross-size conversion (8-bit → 16-bit) and is not supported. The runtime crashes with TT_FATAL/INTERNAL error 13.

## Fix

**Bug 1 fix** (`calcuis_wan2_gguf/pytorch/loader.py` in tt_forge_models, commit `bc142ca89d`):
Load the Wan 2.1 I2V config, strip `added_kv_proj_dim` and `image_dim`, write to a temp directory, and pass via `config=` to `from_single_file`. Also switch from bare URL string to `hf_hub_download` for proper caching. Handles both HighNoise and LowNoise I2V variants via `_I2V_VARIANTS` set.

**Bug 2 fix** (`calcuis_wan2_gguf/pytorch/loader.py` in tt_forge_models, commit `1a832a4c6a`):
Monkey-patch `GGUFParameter.as_tensor` to use `torch._C.DisableTorchFunctionSubclass()`, matching the documented intent of the method. This breaks the `super().__torch_function__` infinite recursion cycle in the dynamo tracing path.

**Bug 3 proposed fix** (tt-metal): Add support for cross-size dtype bitcast in `ttnn::bitcast`. For UINT8→FLOAT16, the operation packs pairs of consecutive uint8 bytes into float16 values (reinterpreting 2×8-bit as 1×16-bit). The fix would require:
1. Detecting cross-size bitcast in `unary_ng_impl` / `UnaryNgDeviceOperation`
2. Adjusting the output shape (last dim divided by dst_bits/src_bits ratio)
3. Implementing the kernel to handle the element packing correctly on TT tile architecture

Alternatively, `stablehlo.bitcast_convert` could be decomposed in `StableHLOToTTIRPatterns.cpp` for cross-size cases: reshape + UINT16 intermediate + same-size bitcast. But this changes semantics for cross-endian packing and may not be correct for all cases.

## Tier B justification
Indicator: **new-infrastructure**.

`ttnn::bitcast` is explicitly limited to same-size dtype pairs. UINT8→FLOAT16 (and UINT8→any wider dtype) requires cross-size element packing, which is a new capability not currently present in the tt-metal kernel. The GGUF Q4_K_M dequantization path uses this cross-size bitcast extensively (multiple times per layer during forward), so adding support is non-trivial: the kernel must correctly handle TT's tiled memory layout while packing/unpacking across tile boundaries.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    338.99s (0:05:38) to reach the Bug 3 failure after Bugs 1 and 2 were fixed
- Tier A attempts: N/A

## Files changed
- `calcuis_wan2_gguf/pytorch/loader.py` (tt_forge_models, 2 commits)
  - `bc142ca89d`: Strip Wan 2.1-specific config fields (added_kv_proj_dim, image_dim) that cause meta tensor NotImplementedError
  - `1a832a4c6a`: Patch GGUFParameter.as_tensor() to use DisableTorchFunctionSubclass, fixing infinite recursion under torch.compile

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ae1187bc3274cb1a93d4dd3de95294450d9a505c |
| tt-forge-models | 1a832a4c6a17805ce721aeca15364396cf804327 |

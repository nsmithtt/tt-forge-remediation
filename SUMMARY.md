# Remediation Summary: flux_1_fill_dev_gguf-pytorch-Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_1_fill_dev_gguf/pytorch-Q4_0-single_device-inference]

## Result
FAIL — GGUF Q4_0 dequantization produces UINT8 tensors; ttnn to_layout does not support device-to-device untilize for UINT8

## Stack layer
loader, tt-mlir

## Tier
A

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
Original failure:
```
E   torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded
```

After loader fixes:
```
loc("power.25"): error: failed to legalize operation 'tosa.pow'
E   ValueError: Error code: 13
```

After tt-mlir Tier A fix:
```
FATAL: Currently to_layout does not support device to device untilize for output data type or memory layout: UINT8
LOG_FATAL @ runtime/lib/ttnn/types/layout_converter.cpp:249
E   RuntimeError: Fatal error
```

## Root cause

Three independent bugs were found, in sequence:

**Bug 1 (loader)**: The original RecursionError came from `GGUFParameter.__torch_function__` being invoked recursively when `torch.compile` called `_make_subclass` on a `GGUFParameter`. Fixed by monkey-patching `as_tensor()` with `torch._C.DisableTorchFunctionSubclass()`.

**Bug 2 (loader)**: The loader used a HTTPS `resolve/main/` URL for `from_single_file`. `diffusers 0.37.1` `_extract_repo_id_and_weights_name` only strips `blob/main/`, causing a doubled prefix → 404. Fixed by using `hf_hub_download` to get a local path. Also: missing `out_channels=64` in `_TRANSFORMER_CONFIG` (proj_out predicts noise only), gated base repo dependency, and missing `gguf>=0.10.0` requirement.

**Bug 3 (tt-mlir, Tier A)**: After loader fixes, `ttir.pow` in the CPU fallback pipeline was lowered via `ElementwiseBinaryOpToTosaPattern<ttir::PowOp, tosa::PowOp>` in `TTIRToLinalg/EltwiseBinary.cpp`. The resulting `tosa.pow` was not legalized by `tosa::addTosaToLinalgPasses` in the nested `CPUModuleOp` context. Fixed by moving `ttir.pow` to a direct `ElementwiseBinaryOpToMathPattern<ttir::PowOp, math::PowFOp>` lowering (bypassing TOSA entirely).

**Bug 4 (Tier B, terminal)**: After the tt-mlir fix, compilation succeeds but execution fails at runtime. The FLUX GGUF Q4_0 model uses `GGUFLinear` layers with on-the-fly dequantization. During `torch.compile` tracing, the dequantization path creates UINT8 tensors (raw quantized weight data). These UINT8 tensors are placed on device, but `ttnn::to_layout` for `UINT8` is not supported for device-to-device untilize (`layout_converter.cpp:247`). This is the same class of bug as `ttnn-bitcast-cross-size-dtype-unsupported` seen in other GGUF models (Wan2, HunyuanVideo 1.5, FLUX ArcticLatent). The model is ~6-8 GB in Q4_0; dequantizing to bf16 would require ~24-32 GB, exceeding device DRAM — so eager dequantization is not feasible.

## Fix

**Loader fixes** (tt-forge-models, `flux_1_fill_dev_gguf/pytorch/loader.py`):
- Patch `GGUFParameter.as_tensor` with `DisableTorchFunctionSubclass` to break recursion
- Use `hf_hub_download` instead of direct HTTPS URL for `from_single_file`
- Add inline `_TRANSFORMER_CONFIG` with `out_channels=64` to avoid gated base repo
- Synthesize inputs in `load_inputs` from config constants (no pipeline needed)
- Add `gguf>=0.10.0` to `requirements.txt`

**tt-mlir fix** (`lib/Conversion/TTIRToLinalg/EltwiseBinary.cpp`):
- Move `ttir.pow` from `populateTTIRToTosaEltwiseBinaryPatterns` (TOSA path) to `populateTTIRToLinalgEltwiseBinaryPatterns` using `ElementwiseBinaryOpToMathPattern<ttir::PowOp, math::PowFOp>`.
- Update `test/ttmlir/Conversion/TTIRToLinalg/binary_eltwise.mlir` to expect `linalg.generic` + `math.powf` instead of `tosa.pow`.

**Proposed fix for terminal bug (Tier B)**:
The `canUntilizeDataTypeOnDevice` function in `runtime/lib/ttnn/utils/utils.cpp` needs to support `UINT8` (if the underlying tt-metal `untilize` kernel supports it), OR the GGUF dequantization path in diffusers needs to be replaced so that no raw UINT8 weight tensors are traced by `torch.compile`. The latter would require either: (a) pre-dequantizing all GGUF layers to bf16 at load time (impractical: ~4x memory), or (b) marking the GGUF dequantization functions with `torch.compiler.disable` to prevent tracing them (would require CPU-based dequantization before device dispatch, which is a compiler frontend infrastructure change).

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure — Supporting UINT8 device operations requires either changes to the tt-metal untilize kernel (verify UINT8 is hardware-supported, update `canUntilizeDataTypeOnDevice`), or a new mechanism to prevent GGUF dequantization from being traced by torch.compile. Both require infrastructure beyond a scoped single-file fix, and the dequantization path in diffusers involves multiple interdependent operations on UINT8 data (scale extraction via view, nibble unpacking via bitwise ops, rescaling) that the TT device does not support as a chain.

## Verification
- pytest exit: FAIL (RuntimeError: Fatal error from UINT8 to_layout)
- Hardware: n150
- Duration: 138.96s (0:02:18)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/flux_1_fill_dev_gguf/pytorch/loader.py` (loader fixes)
- `tt-xla/third_party/tt_forge_models/flux_1_fill_dev_gguf/requirements.txt` (added gguf>=0.10.0)
- `tt-mlir/lib/Conversion/TTIRToLinalg/EltwiseBinary.cpp` (ttir.pow → math.powf)
- `tt-mlir/test/ttmlir/Conversion/TTIRToLinalg/binary_eltwise.mlir` (update expected output)

## Submodule hashes
| Submodule       | Commit                                     |
|-----------------|--------------------------------------------|
| tt-metal        | 3fa4d75355a5ff8c1eec78a7f1e7e26e73a6bef6  |
| tt-mlir         | 36eaa83fa7e2d2b7226fcdfb7e96be72e0e9ab86  |
| tt-xla          | 94362e631                                  |
| tt-forge-models | 57c0704065ac49a2ca5e7c48e31fee09acccb4f9  |

# Remediation Summary: bielik_11b_v3_0_instruct_fp8_dynamic-causal_lm-pytorch-11B_v3.0_Instruct_FP8_Dynamic-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bielik_11b_v3_0_instruct_fp8_dynamic/causal_lm/pytorch-11B_v3.0_Instruct_FP8_Dynamic-single_device-inference]

## Result
FAIL — loader ImportError fixed (requirements.txt added), but tt-xla PJRT layer does not support IEEE FP8 (float8_e4m3fn) output types produced by the compressed-tensors quantized model

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
pjrt-fp8-float-type-unsupported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
**Stage 1 (original, fixed):**
```
ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization. Please install it with `pip install compressed-tensors`.
```

**Stage 2 (after loader fix, remaining):**
```
ERR| TT_THROW: Unsupported float type
ValueError: TT_THROW @ tt-xla/pjrt_implementation/src/utils/data_type_utils.cc:163: tt::exception
info:
Unsupported float type
```
Call stack: `convertMLIRToPJRTDataType → collectOutputTypes → buildModule → compileMlirProgram → _xla_warm_up_cache`

## Root cause

**Stage 1 (loader):** `speakleash/Bielik-11B-v3.0-Instruct-FP8-Dynamic` uses the `compressed-tensors` quantization format. `transformers.AutoModelForCausalLM.from_pretrained` raises `ImportError` if `compressed_tensors` is not installed. The loader had no `requirements.txt`, so the package was never installed.

**Stage 2 (tt-xla compiler):** The model has 350 of 803 parameters in `torch.float8_e4m3fn` (IEEE FP8). When torch_xla traces the forward pass, these FP8 weight tensors appear in the compiled MLIR module's output types (XLA passes parameter state as both inputs and outputs). The function `convertMLIRToPJRTDataType` in `data_type_utils.cc` handles `F64`, `F32`, `F16`, `BF16` but falls through to `TT_THROW("Unsupported float type")` for any other `mlir::FloatType`, including `mlir::Float8E4M3FNType`.

The PJRT C API header already defines `PJRT_Buffer_Type_F8E4M3FN` and related F8 variants, but the MLIR-to-PJRT type conversion function, the PJRT-to-runtime type conversion function (`convertPJRTToRuntimeDataType`), and the downstream buffer allocation and flatbuffer compilation pipeline (`tt-mlir`, `tt-metal`) do not handle IEEE FP8 types. The `tt::target::DataType` enum has Tenstorrent-specific `BFP_Float8`/`BFP_BFloat8` formats but not IEEE `float8_e4m3fn`.

## Fix

**Stage 1 fix (committed):** Added `requirements.txt` containing `compressed-tensors` to `bielik_11b_v3_0_instruct_fp8_dynamic/causal_lm/pytorch/`. This causes the test runner's requirements manager to install the package before model loading.

**Stage 2 proposed fix:**
1. `tt-xla/pjrt_implementation/src/utils/data_type_utils.cc`, `convertMLIRToPJRTDataType`: add cases for `mlir::Float8E4M3FNType → PJRT_Buffer_Type_F8E4M3FN` (and other F8 variants: `Float8E5M2Type → F8E5M2`, `Float8E4M3FNUZType → F8E4M3FNUZ`, etc.)
2. `tt-xla/pjrt_implementation/src/utils/data_type_utils.cc`, `convertPJRTToRuntimeDataType`: add cases mapping `PJRT_Buffer_Type_F8E4M3FN` → a runtime DataType. Requires adding `Float8E4M3FN` (or similar) to `tt::target::DataType` in tt-mlir.
3. `tt-xla/pjrt_implementation/src/api/buffer_instance.cc`: update buffer allocation to handle F8 element type.
4. `tt-mlir`: add FP8 lowering patterns for TTIR/TTNN, or add a pre-lowering cast to BF16 for F8 tensors.
5. `tt-metal`: support FP8 tensor allocation and computation (or verify F8 ops are cast before reaching the device).

## Tier B justification
new-infrastructure — FP8 (float8_e4m3fn) support requires adding the type to `tt::target::DataType`, then propagating through the PJRT conversion functions (`data_type_utils.cc`, `buffer_instance.cc`), the flatbuffer schema and tt-mlir TTIR/TTNN lowering passes, and verifying tt-metal hardware support. This spans at least 3 files across 2 repos (tt-xla + tt-mlir) and may require tt-metal changes.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 108.28s (stage 2 failure run with TT_METAL_RUNTIME_ROOT set)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: bielik_11b_v3_0_instruct_fp8_dynamic/causal_lm/pytorch/requirements.txt` — added `compressed-tensors` dependency (commit b39828fb98 on remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c783e849217444825d7effefb04768f25664ee3f (main); b39828fb98 (remediation branch with requirements.txt fix) |

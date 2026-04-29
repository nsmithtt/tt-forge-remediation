# Remediation Summary: flux2_klein_sdnq_4bit-pytorch-Klein-9B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux2_klein_sdnq_4bit/pytorch-Klein-9B-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
stablehlo-shift-right-arithmetic-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   ValueError: Error code: 13

(PJRT INTERNAL error from `torch_xla._XLAC._xla_warm_up_cache` during
StableHLO→TTIR compilation. Preceded by two loader-layer failures that
blocked reproduction.)

## Root cause

Three bugs were present, each masking the next:

**Loader bug 1**: `sdnq` package not imported before loading the model.
SDNQ (a 4-bit dynamic quantization format) registers its quantizer into
diffusers' `AUTO_QUANTIZER_MAPPING` only when imported. Without the import,
`DiffusersAutoQuantizer.from_dict` raised `ValueError: Unknown quantization
type, got sdnq`.

**Loader bug 2**: After SDNQ registration, the loader called
`self.transformer.to(dtype_override)` on an already-quantized model.
diffusers 0.37.x forbids re-casting a quantized model via `.to()`;
the correct approach is to pass `torch_dtype` to `from_pretrained` only,
which the loader already did.

**Compiler bug (tt-mlir)**: SDNQ dequantization unpacks 4-bit weights from
packed uint8 tensors using `stablehlo.shift_right_arithmetic` to extract the
upper nibble (shift by 4). `StableHLOToTTIRPatterns.cpp` had a pattern for
`ShiftRightLogicalOp → LogicalRightShiftOp` but no pattern for
`ShiftRightArithmeticOp`. The missing pattern caused
`module_builder.cc:889 ERR| Failed to convert from SHLO to TTIR module`,
surfaced as PJRT error code 13 (INTERNAL).

For SDNQ's specific use case, values are packed uint8 cast to int64
(always in [0, 255], never negative), so arithmetic and logical right shift
produce identical results. Mapping `ShiftRightArithmeticOp →
LogicalRightShiftOp` is semantically correct for this data range.

## Fix

**tt_forge_models** (`remediation/flux2_klein_sdnq_4bit-pytorch-Klein-9B-single_device-inference`):

- `flux2_klein_sdnq_4bit/pytorch/requirements.txt` (new): adds `sdnq` dependency.
- `flux2_klein_sdnq_4bit/pytorch/loader.py`: adds `import sdnq` at module top
  to register the SDNQ quantizer with diffusers before `from_pretrained`.
  Removes the redundant `self.transformer.to(dtype_override)` call that
  diffusers forbids on quantized models.

**tt-mlir** (`remediation/flux2_klein_sdnq_4bit-pytorch-Klein-9B-single_device-inference`):

- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`: adds
  `ShiftRightArithmeticOp → LogicalRightShiftOp` conversion pattern alongside
  the existing `ShiftRightLogicalOp → LogicalRightShiftOp` pattern.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    1350.78s (0:22:30)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/flux2_klein_sdnq_4bit/pytorch/requirements.txt` (new)
- `tt_forge_models/flux2_klein_sdnq_4bit/pytorch/loader.py`
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | d4f9070db9e8a1e0c329a583bde66d5205ca1d49 |
| tt-xla          | 7ad0d7b258d2c4d16ed6890e54f53818707a6e51 |
| tt-forge-models | 17ade9fb58a44384a7c5513bfcae242260843fb2 |

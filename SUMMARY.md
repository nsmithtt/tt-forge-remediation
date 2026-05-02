# Remediation Summary: lumina_gguf-pytorch-Q8_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lumina_gguf/pytorch-Q8_0-single_device-inference]

## Result
FAIL — complex<f64> gather/multiply operations not lowered through StableHLO→TTIR ComplexDataTypeConversion pass; triggers Error code 13 in partition_fx_graph_for_cpu_fallback

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
complex-f64-gather-lowering-not-implemented

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
loc("concatenate.60"): error: failed to legalize unresolved materialization from
('tensor<1x4224x16xcomplex<f64>>') to ('tensor<1x4224x16x2xf64>') that remained live after conversion
Failed to convert from SHLO to TTIR module
...
ValueError: Error code: 13
```

## Root cause

Two loader bugs were fixed (existing remediation branch `a25a9436a0`):

1. **URL doubling** (`resolve/main/resolve/main/…`): `from_single_file` re-appended
   `resolve/main/` to the already-full URL. Fixed by using `hf_hub_download` to
   obtain a local path and passing that to `from_single_file`.

2. **GGUFParameter Dynamo recursion** (`RecursionError: maximum recursion depth
   exceeded`): the original failure. `GGUFQuantizationConfig` leaves
   `GGUFParameter` objects in the model, which caused infinite recursion when
   Dynamo traced them. Fixed by calling `model.dequantize()` + clearing
   `is_quantized` + `.to(compute_dtype)` to convert all weights to plain
   `nn.Linear` before compilation.

After the loader fixes, the remaining failure is in the compiler stack.

**MLIR failure — layer: tt-mlir**

`Lumina2RotaryPosEmbed._get_freqs_cis()` is compiled as a separate subgraph by
`torch.compile`. The precomputed `self.freqs_cis[i]` tensors are
`complex<f64>` (diffusers picks float64 on any non-MPS device). The compiled
graph contains:

- `GatherOp` on `complex<f64>` inputs (from `torch.gather(freqs.unsqueeze(0).repeat(...), dim=1, index=index)`)
- `ConcatenateOp` on the gather results

`StableHLOComplexDataTypeConversionPass` handles `ConcatenateOp`,
`ConstantOp`, `ReshapeOp`, `SliceOp`, and `BroadcastInDimOp` with complex
element types, but **not `GatherOp`**. The gather's `complex<f64>` output is
never converted; when the concatenate pattern fires and calls
`adaptor.getOperands()`, the framework inserts a source materialization from
`complex<f64>` to `f64[...,2]` for the unconverted gather result. No pattern
resolves that materialization, so `applyPartialConversion` fails with
"unresolved materialization … that remained live after conversion".

Additionally, `rope_embedder.forward()` line 271 calls
`attention_mask.sum(dim=1).tolist()` on a TT tensor at execution time
— a second Tier B `pjrt-device-to-host-transfer` issue.

The full complex<f64> arithmetic path used by Lumina2 also includes
`torch.view_as_complex → x * freqs_cis → torch.view_as_real` in
`apply_rotary_emb` (use_real=False path). Even after fixing `GatherOp`,
the complex `MulOp` in that path also has no lowering pattern. The entire
complex<f64> compute path through TTIR is not implemented.

After the MLIR failure, `partition_fx_graph_for_cpu_fallback` is invoked.
`_xla_warm_up_cache` inside it raises `ValueError: Error code: 13`
(pjrt-device-to-host-transfer, Tier B).

## Fix

**Loader (committed on remediation branch `a25a9436a0`):**

- `lumina_gguf/pytorch/loader.py`: replace the broken `resolve/main` HF URL
  with `hf_hub_download`; call `model.dequantize()` + clear `is_quantized` +
  `.to(compute_dtype)` to remove `GGUFParameter` objects before compilation.

**Compiler stack (proposed, not implemented):**

The fix would live in
`tt-mlir/lib/Dialect/StableHLO/Transforms/ComplexDataTypeConversion.cpp`.
A `ComplexGatherOpConversionPattern` needs to be added that rewrites
`GatherOp` with complex inputs by:
1. Converting the input from `complex<f64>[…,N]` to `f64[…,N,2]`
2. Updating `slice_sizes` to append `2` for the new trailing real/imag dimension
3. Updating `offset_dims` to include the new trailing dimension index

Additionally, a complex `MulOp` decomposition
(`(a+bi)(c+di) = (ac−bd) + (ad+bc)i`) is needed for the `apply_rotary_emb`
path. The `.tolist()` at `rope_embedder.forward():271` requires a separate
fix (avoid device-to-host transfer or graph-break handling).

## Tier B justification

- `new-infrastructure`: The complete complex<f64> computation path
  (GatherOp, MulOp, view_as_complex, view_as_real) through the
  StableHLO→TTIR pipeline is not implemented. Adding GatherOp alone does not
  make the test pass because MulOp on complex is also missing. Cross-cutting
  changes across multiple patterns are required. Additionally, the
  `pjrt-device-to-host-transfer` Error code 13 in
  `partition_fx_graph_for_cpu_fallback` is a known Tier B infrastructure bug.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    69.15s (1:09) with loader fixes applied
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/lumina_gguf/pytorch/loader.py` (remediation branch `a25a9436a0`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | a25a9436a05fa0554db58a95a7b138b8214e250f |

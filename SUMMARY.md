# Remediation Summary: granite_tspulse-pytorch-r1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granite_tspulse/pytorch-r1-single_device-inference]

## Result
FAIL — TSPulse uses torch.fft.rfft/irfft which lowers to stablehlo.fft; tt-mlir has no stablehlo.fft → TTIR lowering

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
stablehlo-fft-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   ValueError: Error code: 13
```

Full traceback excerpt:
```
venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py:483: in extract_graph_helper
    torch_xla._XLAC._xla_warm_up_cache(args_and_out_tensor_only, [])
E   ValueError: Error code: 13
```

## Root cause
`TSPulseForReconstruction` runs `get_fft()` on every forward pass (config has `fuse_fft=True`):

```python
# modeling_tspulse.py:2332
rfft_result = torch.fft.rfft(inputs, dim=1)  # complex result
real_part = rfft_result.real
imag_part = rfft_result.imag
rfft_mag = torch.abs(rfft_result[:, 1:, :])
...
# modeling_tspulse.py:3326
reconstructed_ts_from_fft = torch.fft.irfft(rfft_result, n=..., dim=1)
```

`torch.fft.rfft` decomposes to `aten._fft_r2c` which XLA lowers to `stablehlo.fft`.
`torch.fft.irfft` decomposes to `aten._fft_c2r` which XLA lowers to `stablehlo.fft`.

tt-mlir's compilation pipeline (`StablehloComplexMathExpanderPass` +
`StableHLOComplexDataTypeConversionPass`) handles complex arithmetic but does **not**
handle `stablehlo.fft`. The pipeline converts complex tensor types to float-pair
representation, but `stablehlo.fft` nodes still reference the original complex type,
causing a materialization failure. This surfaces as `ValueError: Error code: 13`
at `_xla_warm_up_cache`.

Model config key parameters:
- `fuse_fft: True` (FFT always enabled, even in eval mode)
- `context_length: 512`, `patch_stride: 8`, `num_patches: 128`
- `d_model: 24`, `num_input_channels: 1`
- `self_attn: False` (no SDPA — SDPA chunk-size constraint does not apply)

This is the same root cause as the previously reported HyenaDNA models
(`report/hyenadna-causal_lm-pytorch-tiny-1k-seqlen-single_device-inference` and
`report/hyenadna-causal_lm-pytorch-large-1m-seqlen-single_device-inference`), both of
which also filed FAIL with Tier B / `stablehlo-fft-no-lowering`.

## Fix
No fix attempted (Tier B). Proposed fix: implement `stablehlo.fft` → TTIR lowering in
`tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`, adding a
pattern that converts `stablehlo.fft` to a new `ttir.fft` op (requiring a matching
TTNN implementation in tt-metal). This requires new TTIR/TTNN ops plus a hardware
FFT kernel — a new-infrastructure change spanning ≥ 3 files across ≥ 2 repos.

## Tier B justification
Indicator: **new-infrastructure**.

Implementing FFT requires (1) a new `stablehlo.fft` → TTIR lowering pattern in
`tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`, (2) a new
`ttir.fft` op declaration and TTNN lowering pattern, and (3) a corresponding TTNN
kernel implementation in tt-metal. No single isolated change suffices; this is a
coordinated multi-repo, multi-file addition.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    115.27s (0:01:55) — test ran to completion, failed at compilation
- Tier A attempts: N/A

## Files changed
None — Tier B, no fix attempted.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7cf0e9b8df122b73c9a40dc67624f25d0232d3ee |

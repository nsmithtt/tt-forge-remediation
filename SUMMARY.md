# Remediation Summary: hyenadna-causal_lm-pytorch-tiny-1k-seqlen-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[hyenadna/causal_lm/pytorch-tiny-1k-seqlen-single_device-inference]

## Result
FAIL — stablehlo.fft has no TTIR lowering in tt-mlir; HyenaDNA's Hyena filter uses FFT convolution which cannot be compiled for TT hardware

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
E   RuntimeError: TT_THROW @ /home/ttuser/hf-bringup/tt-xla/pjrt_implementation/src/api/buffer_instance.cc:282: tt::exception
info:
Complex tensor with num_dims == 0 is not supported.
```

Local reproduction:
```
2026-04-28 10:06:45.574 (   7.700s) [        D5BB2080]               assert.h:121    ERR| TT_THROW: Complex tensor with num_dims == 0 is not supported.
E   RuntimeError: TT_THROW @ .../buffer_instance.cc:282: tt::exception
While executing %div : call_function[target=torch.ops.aten.div.Tensor](args = (%_fft_r2c, 66), kwargs = {})
  File "modeling_hyena.py", line 22, in fftconv
    k_f = torch.fft.rfft(k.to(torch.float32), n=fft_size) / fft_size
```

## Root cause
HyenaDNA's Hyena filter is implemented via FFT-based convolution (`fftconv` in `modeling_hyena.py`):

```python
k_f = torch.fft.rfft(k.to(torch.float32), n=fft_size) / fft_size
u_f = torch.fft.rfft(u.to(dtype=torch.float32), n=fft_size)
y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
```

`torch.fft.rfft` and `torch.fft.irfft` decompose to `aten._fft_r2c` and `aten._fft_c2r`, which XLA lowers to `stablehlo.fft`. tt-mlir's compilation pipeline (`StablehloComplexMathExpanderPass` + `StableHLOComplexDataTypeConversionPass`) handles complex arithmetic but does **not** handle `stablehlo.fft`. The pipeline converts complex tensor types to float-pair representation, but `stablehlo.fft` nodes still reference the original complex type, causing "unresolved materialization" at link time.

The immediate symptom (TT_THROW at `buffer_instance.cc:282`) is a secondary effect: before the graph compiles, XLA applies type promotion when processing `rfft_result / fft_size`, promoting the integer scalar `fft_size` to a 0-dimensional complex tensor. TT explicitly rejects 0-dim complex buffers.

This is the same root cause as the previously reported `large-1m-seqlen` variant
(`report/hyenadna-causal_lm-pytorch-large-1m-seqlen-single_device-inference`),
which also filed FAIL with Tier B / `stablehlo-fft-no-lowering`. The `tiny-1k-seqlen`
model uses shorter sequences (1k vs 1M) but the same Hyena filter architecture and
identical FFT convolution code path.

## Fix
Not applied. The proposed fix would require:

1. **TTIR FFT op** — define `ttir.fft` / `ttir.rfft` / `ttir.irfft` ops in the TTIR dialect (tt-mlir).
2. **StableHLO → TTIR lowering** — add `stablehlo.fft` → `ttir.fft` pattern in `StableHLOToTTIRPatterns.cpp` covering RFFT, IRFFT, FFT, IFFT variants.
3. **tt-metal kernel** — implement an FFT kernel for Wormhole that `ttir.fft` can lower to.

Alternatively, `stablehlo.fft` could be expanded into real arithmetic (DFT matrix multiplication) before reaching TTIR, but this approach is numerically expensive and not practical for sequence lengths in HyenaDNA.

## Tier B justification
**Indicator**: new-infrastructure

FFT support does not exist in any layer of the TT stack (tt-mlir defines no TTIR FFT op; tt-metal has no FFT kernel; `StableHLOToTTIRPatterns.cpp` has no `FftOp` pattern). Implementing FFT end-to-end requires coordinated changes across tt-mlir (dialect + lowering) and tt-metal (kernel), touching at minimum 3–4 files in 2 repos.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: ~44s (to failure)
- Tier A attempts: N/A (same Tier B bug confirmed from prior large-1m-seqlen report; no Tier A probe needed)

## Files changed
None.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

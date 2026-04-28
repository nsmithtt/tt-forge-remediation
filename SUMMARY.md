# Remediation Summary: hyenadna-causal_lm-pytorch-large-1m-seqlen-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[hyenadna/causal_lm/pytorch-large-1m-seqlen-single_device-inference]

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
```

Local reproduction:
```
2026-04-28 09:48:00.150 (  12.883s) [        82082140]               assert.h:121    ERR| TT_THROW: Complex tensor with num_dims == 0 is not supported.
E   RuntimeError: TT_THROW @ .../buffer_instance.cc:282: tt::exception
While executing %div : call_function[target=torch.ops.aten.div.Tensor](args = (%_fft_r2c, 66), kwargs = {})
  File "modeling_hyena.py", line 22, in fftconv
    k_f = torch.fft.rfft(k.to(torch.float32), n=fft_size) / fft_size
```

## Root cause
HyenaDNA implements its Hyena filter via FFT-based convolution (`fftconv` in `modeling_hyena.py`):

```python
k_f = torch.fft.rfft(k.to(torch.float32), n=fft_size) / fft_size
u_f = torch.fft.rfft(u.to(dtype=torch.float32), n=fft_size)
y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
```

`torch.fft.rfft` and `torch.fft.irfft` decompose to `aten._fft_r2c` and `aten._fft_c2r`, which XLA lowers to `stablehlo.fft`. tt-mlir's compilation pipeline (`StablehloComplexMathExpanderPass` + `StableHLOComplexDataTypeConversionPass`) handles complex arithmetic but does **not** handle `stablehlo.fft`. The pipeline converts complex tensor types to float-pair representation, but `stablehlo.fft` nodes still reference the original complex type, causing "unresolved materialization" at link time.

The immediate symptom (TT_THROW at `buffer_instance.cc:282`) is a secondary effect: before the graph compiles, XLA applies type promotion when processing `rfft_result / fft_size`, promoting the integer scalar `fft_size=66` to a 0-dimensional complex tensor. TT explicitly rejects 0-dim complex buffers. Removing that guard (Tier A probe) revealed the deeper `stablehlo.fft` failure:

```
loc("p1.2"): error: failed to legalize unresolved materialization
  from ('tensor<256x34x2xf32>') to ('tensor<256x34xcomplex<f32>>')
  that remained live after conversion
ValueError: Error code: 13
While executing %_fft_c2r : call_function[target=torch.ops.aten._fft_c2r.default]
```

## Fix
Not applied. The proposed fix would require:

1. **TTIR FFT op** — define `ttir.fft` / `ttir.rfft` / `ttir.irfft` ops in the TTIR dialect (tt-mlir).
2. **StableHLO → TTIR lowering** — add `stablehlo.fft` → `ttir.fft` pattern in `StableHLOToTTIRPatterns.cpp` covering RFFT, IRFFT, FFT, IFFT variants.
3. **tt-metal kernel** — implement an FFT kernel for Wormhole that `ttir.fft` can lower to.

Alternatively, `stablehlo.fft` could be expanded into real arithmetic (DFT matrix multiplication) before reaching TTIR, but this approach is numerically expensive for long sequences and not practical for HyenaDNA's 1M-seqlen context.

## Tier B justification
**Indicator**: new-infrastructure

FFT support does not exist in any layer of the TT stack (tt-mlir defines no TTIR FFT op; tt-metal has no FFT kernel; `StableHLOToTTIRPatterns.cpp` has no `FftOp` pattern). Implementing FFT end-to-end requires coordinated changes across tt-mlir (dialect + lowering) and tt-metal (kernel), touching at minimum 3–4 files in 2 repos.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: ~52s (to failure)
- Tier A attempts: 1 (probe removing 0-dim complex guard in buffer_instance.cc; reverted when deeper FFT error surfaced)

## Files changed
None (Tier A probe reverted).

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

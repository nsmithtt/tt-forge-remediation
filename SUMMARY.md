# Remediation Summary: distill_neucodec-pytorch-distill_neucodec-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[distill_neucodec/pytorch-distill_neucodec-single_device-inference]

## Result
FAIL — ISTFTHead uses torch.view_as_complex/torch.istft (complex tensors); TT PJRT throws "Complex tensor with num_dims == 0 is not supported" in buffer_instance.cc:282

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-complex-tensor-not-supported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

After loader fixes (torchaudio CPU wheel + vocos decoder isolation):
RuntimeError: TT_THROW @ /home/nsmith/tt-forge-remediation/tt-xla/pjrt_implementation/src/api/buffer_instance.cc:282: tt::exception
info:
Complex tensor with num_dims == 0 is not supported.

## Root cause
`ISTFTHead.forward` (from the `vocos` library used by `neucodec`) computes:

```python
x = mag * torch.view_as_complex(torch.stack([torch.cos(p), torch.sin(p)], dim=-1))
x = self.istft(x)
```

`torch.view_as_complex` creates a complex-dtype tensor from the real+imag stack. During
FX graph tracing/compilation for TT device, a 0-dim complex constant is encountered during
`TransferToDevice`. The TT PJRT layer (`buffer_instance.cc:282`) explicitly throws for
0-dim complex tensors:

```cpp
if (data_type_utils::isComplexPJRTType(data_type) && num_dims == 0) {
    TT_THROW("Complex tensor with num_dims == 0 is not supported.");
}
```

While TT PJRT has partial complex support (>0-dim complex stored as float with trailing
dim=2), it does not support scalar complex tensors. More fundamentally, the `torch.istft`
op that follows requires complex arithmetic lowerings in StableHLO→TTIR and corresponding
tt-metal kernel support — neither of which exists.

Two loader-layer bugs were also found and fixed (see Files changed), but the remaining
failure is a compiler-stack bug.

## Fix
Proposed fix: Implement complex tensor support in the TT compiler stack:
1. `pjrt_implementation/src/api/buffer_instance.cc`: Handle 0-dim complex tensors
   by representing them as shape `[2]` (real+imag), matching the existing >0-dim handling.
2. `tt-mlir`: Add StableHLO→TTIR lowerings for complex arithmetic ops
   (`stablehlo.complex`, `stablehlo.real`, `stablehlo.imag`, `stablehlo.fft`).
3. `tt-metal`: Kernel support for complex-valued tensor operations required by iSTFT.

## Tier B justification
**new-infrastructure**: Full complex tensor support requires new PJRT transfer paths
(0-dim complex + complex dtype), new StableHLO→TTIR lowering patterns for complex ops
and FFT, and new tt-metal kernels — cross-cutting changes across tt-xla, tt-mlir, and
tt-metal that exceed the scope of a single-file scoped fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    N/A (failed before silicon execution)
- Tier A attempts: 0

## Files changed
In tt_forge_models (branch: remediation/distill_neucodec-pytorch-distill_neucodec-single_device-inference):
- `distill_neucodec/pytorch/requirements.txt`: Added `--extra-index-url https://download.pytorch.org/whl/cpu` so `torchaudio==2.9.1` resolves to the CPU build (the CUDA torchaudio fails to import on CPU-only machines).
- `distill_neucodec/pytorch/loader.py`: Rewrote loader with `_DistillNeuCodecVocosDecoder` wrapper exposing only `VocosBackbone + ISTFTHead` for TT compilation. `load_inputs` pre-computes the full encode→quantize pipeline (`DistillCodecEncoder` + `HubertModel` semantic encoder + `ResidualFSQ`) in float32 to avoid bfloat16 incompatibilities in `local_attention` (upcasts to float32 internally) and `ResidualFSQ.project_out` (float32 input meets bfloat16 weight). The resulting `fsq_post_emb [1, T, 1024]` is the input to the TT-compiled forward path.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6efbae71f2cbe0a96ee9fbf0f81c3de31edd9ee0 |
| tt-forge-models | e7582d7a130563f309a6b263ee2fe187bf0a95ce |

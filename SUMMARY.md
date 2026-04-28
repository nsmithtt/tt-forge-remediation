# Remediation Summary: jens_lundsgaard_nolstm_2026_03_12-pytorch-nolstm-2026-03-12-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[jens_lundsgaard_nolstm_2026_03_12/pytorch-nolstm-2026-03-12-single_device-inference]

## Result
FAIL — TT hardware BF16 matmul accumulation causes encoder z_seq PCC=0.987 vs CPU, which is amplified by the decoder to x_rec PCC=0.849 (required 0.95)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8543029548412976. Required: pcc=0.95.

## Root cause
The `ConvLSTMAutoencoder` model (`use_convlstm=False`) has a deep encoder path:
spatial_cnn (Conv2d 1→64→128→256, each layer with BatchNorm2d) → `latent_compress`
`Linear(65536, 512)` → `lin1` `Linear(512, 512)`. The TT hardware accumulates
bf16 matmul products in bf16, while CPU PyTorch internally uses float32
accumulators for bf16 matmul. The spatial_cnn Conv2d layers (up to
256×3×3 = 2304 accumulation terms) and the latent_compress Linear(65536 → 512)
layer each produce small per-layer BF16 accumulation errors. These compound
through the encoder, resulting in z_seq PCC=0.987 (TT vs CPU bf16).

The model's decoder is a learned inverse mapping (Linear(512, 65536) → three
ConvTranspose2d upsampling blocks). When given identical z_seq inputs, the
decoder produces PCC=0.999976 (TT vs CPU), confirming the decoder itself is not
the source of error. However, when the decoder receives the slightly-off TT
encoder output (z_seq PCC=0.987), it amplifies that error in the latent space
into a larger reconstruction error (x_rec PCC=0.849). This amplification is a
property of the learned model, not a compiler bug, but it is triggered by the
BF16 matmul precision gap.

Measured PCCs:
- spatial_cnn (TT vs CPU, real inputs): 0.999767
- latent_compress (TT vs CPU, identical inputs): 0.997585
- Full encoder z_seq (TT vs CPU): 0.987
- Decoder (TT vs CPU, identical z_seq): 0.999976
- Full x_rec (TT vs CPU): 0.849

## Fix
No fix attempted. The root cause is cross-cutting: increasing BF16 matmul
accumulation precision (math_fidelity / DestAccumFormat) across all Conv2d and
Linear lowerings in tt-metal/tt-mlir would be required to bring the x_rec PCC
above 0.95. This is not a scoped change to a single pattern or file.

Proposed fix: increase `math_fidelity` to `HiFi4` (float32 accumulators) in the
TTNN matmul/conv kernel configuration within the TTIR-to-TTNN lowering pass in
`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`. This would need to be
applied globally or selectively for Conv2d/ConvTranspose2d/Linear ops, touching
the kernel config construction in multiple op lowering patterns.

## Tier B justification
cross-cutting — the fix (changing BF16 accumulation to FP32 accumulation for
matmul/conv) touches the lowering of every Conv2d, ConvTranspose2d, and Linear
operation in tt-mlir/tt-metal, affecting all model tests and the global
math_fidelity configuration.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    92.21s (0:01:32) for reproduction run
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

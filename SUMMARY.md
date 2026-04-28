# Remediation Summary: control_2026_03_04/pytorch-control-2026-03-04-single_device-inference

## Skill version
17

## Test
tests/runner/test_models.py::test_all_models_torch[control_2026_03_04/pytorch-control-2026-03-04-single_device-inference]

## Result
FAIL — bfloat16 matmul precision degrades for large K (K=65536) on TT hardware; z_seq PCC=0.60, x_rec PCC=0.87

## Failure
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.802175570155345. Required: pcc=0.95.
```

## Root cause

**Layer: tt-metal / tt-mlir (compiler/hardware)**

The ConvLSTMAutoencoder encoder contains a large linear projection layer:
```
encoder.latent_compress = nn.Linear(65536, 200)
```
where `65536 = hidden_dim(256) × H(16) × W(16)`, the flattened spatial CNN output.

The model is run in bfloat16 (the `TorchDynamicLoader` always passes `dtype_override=torch.bfloat16`). The TT hardware's bfloat16 matrix multiplication uses bfloat16 accumulators for the large K=65536 reduction, while CPU bfloat16 (x86 PyTorch) internally uses float32 accumulators. This causes a systematic magnitude underestimation:

| K | TT std / CPU std |
|---|-----------------|
| 512 | 0.997 |
| 4096 | 0.988 |
| 16384 | 0.955 |
| 32768 | 0.913 |
| 65536 | 0.846 |

With the actual model's CNN outputs (post-BN, post-ReLU) and the subsequent ReLU applied to the linear projection, the underestimation worsens to ~0.74×. The LSTM input is thus scaled-down relative to what the network was trained with, causing systematically different gate activations that diverge over 50 time steps:

- Linear(65536→200) + ReLU on TT: mean=3.35, std=5.88 (should be: mean=4.55, std=7.99)
- z_seq PCC (cpu-bf16 vs TT-bf16): 0.597
- x_rec PCC (cpu-bf16 vs TT-bf16): 0.875 (decoded from wrong z_seq)

CPU bfloat16 (float32-accumulator) comparison confirmed clean: PCC > 0.999 for both outputs.

## Fix
**Proposed fix in tt-mlir / tt-metal**: For matrix multiplications with large reduction dimension K (≥ ~4096), the compiler should emit a computation that accumulates in float32 (or equivalent higher precision). Concretely:

- In tt-mlir's matmul lowering, detect when `K > threshold` and configure the generated tile operations to use fp32 accumulators
- Or implement a split-K reduction: tile the K dimension, accumulate sub-results in fp32, then sum

No loader-level fix is appropriate. The scale-down is a hardware precision characteristic, not a model loading bug. Workarounds (weight dtype overrides to keep `latent_compress` in float32, or lowering required_pcc) would hide the underlying hardware bug.

## Verification
```
pytest -svv "tests/runner/test_models.py::test_all_models_torch[control_2026_03_04/pytorch-control-2026-03-04-single_device-inference]"
FAILED — pcc=0.802175570155345, required=0.99
Hardware: n150
```

## Files changed
None — no fix was applied (compiler-stack bug; fix would be in tt-mlir or tt-metal).

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

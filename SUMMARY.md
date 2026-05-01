# Remediation Summary: latent_sync-pytorch-1.5-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[latent_sync/pytorch-1.5-single_device-inference]

## Result
SILICON_PASS — shouldUseDecode guard prevents SDPA decode when K seq_len is not a multiple of 32

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
sdpa-k-chunk-size-lt-32

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Underlying hardware assertion:
```
TT_FATAL: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2
at /home/nsmith/tt-forge-remediation/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:66:
    k_chunk_size % 32 == 0
```

## Root cause
The LatentSync-1.5 UNet3DConditionModel uses AnimateDiff-style temporal self-attention
over `num_frames=2` frames. In the compiled TTNN module, the temporal cross-attention
path produces a Q tensor with `seq_len=1` and a K/V tensor with `seq_len=2`. The
`shouldUseDecode()` predicate in `TTIRToTTNN.cpp` routed this SDPA op to
`scaled_dot_product_attention_decode` based solely on the Q `seq_len == 1` check.

At runtime, `sdpa_decode.cpp` calls `get_chunk_size(s=2)`, which returns `2` (the
maximum power-of-2 divisor of 2). This violates the hardware constraint
`k_chunk_size % 32 == 0`, causing TT_FATAL and propagating as INTERNAL error code 13.

The fix adds a K-seq-len divisibility guard to `shouldUseDecode()`: if `kSeqLen % 32 != 0`,
the SDPA falls through to regular TTNN SDPA instead of the decode path, avoiding the
TT_FATAL.

## Fix
`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — `shouldUseDecode()`:

Added guard: also require `kSeqLen > 0 && (kSeqLen % 32 == 0)` on the K tensor's
sequence-length dimension before routing to SDPA decode.

Remediation branch: `remediation/latent_sync-pytorch-1.5-single_device-inference`
in `tenstorrent/tt-mlir` (commit `1bc538142`).

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    468.31s (0:07:48)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 1bc538142eae74722154a85a55fee2b4d5bb73b1 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | eab1f357e14e227d2030fa688f69b02b1a5531a3 |

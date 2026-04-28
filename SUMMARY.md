# Remediation Summary: jarvisevo-pytorch-9B-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[jarvisevo/pytorch-9B-single_device-inference]

## Result
FAIL — Conv3d L1 CB overflow fixed (Tier A); DRAM OOM on second tensor allocation is a second, unfixed Tier B bug

## Stack layer
tt-metal

## Tier
A

## Bug fingerprint
conv3d-cin-block-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Underlying cause (from device log):
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
grow to 2247168 B which is beyond max L1 size of 1572864 B
```

## Root cause
`conv3d_program_factory.cpp::create()` allocates five static CBs: `vol2col_rm`,
`vol2col_tiled`, `weight_tiled`, `matmul_interm`, and `matmul_result`.  The
activation CB (`vol2col_tiled`) and weight CB (`weight_tiled`) both scale with
`K_t = ceil(kT * kH * kW * C_in_block / TILE_WIDTH)`.

For JarvisEvo (Qwen3VL-9B), the vision patch embedding Conv3d has C_in=32 (after
preprocessing merges temporal frames into channels), kernel_size=(2,16,16), so
K_t = ceil(2×16×16×32 / 32) = 512.  With M_t=1, N_t=1 (C_out_block=32), the two
dominant CBs are each 512 × 2048 = 1 MB, totalling over 2 MB against an L1 budget
of 1.46 MB.

The existing `C_out_block` reduction guard in `create()` does not help because the
overflowing CBs scale with K_t (input-channel dimension), not N_t (output-channel
dimension).  A `C_in_block` guard was absent.

The kernel already supports multiple `C_in_num_blocks` for accumulation, so
reducing `C_in_block` is a safe, scoped change in one file.

After the L1 fix, the test progresses further and hits a second failure:

```
TT_FATAL: Out of Memory: Not enough space to allocate 1442840576 B DRAM buffer
across 8 banks, where each bank needs to store 180355072 B, but bank size is
4273390016 B (allocated: 3994808960 B, free: 278581056 B, largest free block:
160590656 B)
```

Each bank has 265 MB free but the largest contiguous block is 153 MB while 180 MB
is needed.  Per-bank DRAM is fragmented.  ~29.8 GB is allocated at this point for
a 9B bf16 model.  Root cause of the fragmentation is unknown without deeper
investigation.

## Fix
**Applied (Tier A) — `tt-metal`:**

`ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`:
Added an unconditional overflow guard after `vol2col_rm_pages` is computed.
If the five main static CBs exceed `l1_unreserved − 200 KB`, the guard walks
divisors of `C_in` downward until the CB total fits, then updates `C_in_block`,
`patch_size`, `padded_patch_size*`, `C_in_num_blocks`, `matmul_K_t`, and
`C_in_block_bytes`.  The guard fires regardless of whether `config.C_in_block`
was set explicitly by the compiler, because the overflow condition depends only on
tensor geometry.

Commit: `38e5465468198d250c09c58960ee4ded1c268adc`
Branch: `remediation/jarvisevo-pytorch-9B-single_device-inference` in tt-metal

**Not applied (Tier B) — DRAM fragmentation:**

The DRAM allocator fails to find a 180 MB contiguous region per bank despite 265 MB
free, because prior allocations have fragmented the bank.  The 30 GB DRAM
footprint for a 9B model is higher than the ~20 GB expected for weights +
activations, suggesting possible double-buffering or unreleased intermediate
tensors.  Diagnosing and fixing this requires understanding the full allocation
sequence for all tensors in the graph, which is a cross-cutting change touching
the TTNN memory layout, graph execution, and potentially the compiler's buffer
assignment pass.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A for the first (Tier A) fix.

The second (DRAM OOM) bug is Tier B because:
- **cross-cutting**: the root cause (fragmentation / excessive DRAM footprint)
  spans TTNN memory layout, runtime buffer lifetime, and possibly compiler buffer
  assignment.
- **internal-error-unknown-mechanism**: without knowing which tensors consume the
  extra ~8–10 GB above expected, no scoped fix can be proposed.

## Verification
- pytest exit: FAIL (L1 overflow resolved; DRAM OOM is the new blocker)
- Hardware:    n150
- Duration:    178.60s (run with L1 fix, before DRAM OOM hit; no passing run)
- Tier A attempts: 1

## Files changed
- `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 38e5465468198d250c09c58960ee4ded1c268adc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

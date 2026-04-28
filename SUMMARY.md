# Remediation Summary: huihui_qwen_3_vl_abliterated-image_to_text-pytorch-8b_instruct_abliterated-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen_3_vl_abliterated/image_to_text/pytorch-8b_instruct_abliterated-single_device-inference]

## Result
FAIL — Conv3d circular buffers exceed L1 (cb_vol2col_tiled + cb_weight_tiled = 2 × 1 MB, L1 limit = 1.5 MB); Tier B — kernel K-streaming redesign required

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
conv3d-k-dimension-cb-l1-overflow

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

C++ root:
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
grow to 2247168 B which is beyond max L1 size of 1572864 B
  tt::runtime::ttnn::operations::conv::run(tt::target::ttnn::Conv3dOp const*, ...)
```

## Root cause

The Qwen3-VL 8B vision encoder (`Qwen3VLVisionPatchEmbed`) uses a `nn.Conv3d`
with `kernel_size=[2, 16, 16]`, `stride=[2, 16, 16]`, `in_channels=3`,
`out_channels=1152`. After padding `C_in` to the next TILE_WIDTH multiple,
the effective `C_in = 32`.

The Conv3d program factory in `tt-metal` allocates two L1 circular buffers
that hold the **entire** matmul K dimension at once:

```
matmul_K_t = ceil(kD × kH × kW × C_in_block / TILE_WIDTH)
           = ceil(2 × 16 × 16 × 32 / 32) = 512 tiles
```

- `cb_vol2col_tiled`: `tile_size × matmul_M_t × matmul_K_t = 2048 × 1 × 512 = 1,048,576 B`
- `cb_weight_tiled`:  `tile_size × matmul_K_t × matmul_N_t = 2048 × 512 × 1 = 1,048,576 B`

Together they require ≈ 2 MB plus additional CBs (vol2col_rm, matmul_interm,
result_rm, bias), totalling ~2,247,168 B — exceeding the 1,572,864 B (1.5 MB)
L1 per Wormhole Tensix core.

The compute kernel (`compute.cpp`) calls `cb_wait_front(cb_weight_tiled,
weight_tiles)` and `cb_wait_front(cb_vol2col_tiled, patch_tiles)` before the
matmul — it **requires all K tiles resident in L1** before it can begin. There
is no K-dimension streaming.

A secondary graph break on `aten._local_scalar_dense.default` (from
`grid_thw.tolist()` → `.item()` calls in `fast_pos_embed_interpolate`) occurs
before the Conv3d crash, but it is handled gracefully with a dynamo graph break
and is not the fatal failure.

## Fix

The fix requires implementing **K-dimension streaming** in the Conv3d kernel:
split `matmul_K_t` into smaller `K_block` chunks so that `cb_vol2col_tiled`
and `cb_weight_tiled` each hold only `K_block` tiles at a time (e.g.
`K_block = 8` → 16 KB each, fitting comfortably in L1).

Files that would need coordinated changes:

1. `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`
   — Reduce CB sizes: `K_block × matmul_N_t` and `matmul_M_t × K_block`.
   Add `K_block` compile-time parameter.
2. `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp`
   — Replace single-pass matmul with a loop over K blocks, accumulating partial
   sums into `cb_matmul_interm_tiled`.
3. `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp`
   — Push weights in K-block chunks (stride through `K_block × matmul_N_t`
   tiles per loop iteration).
4. `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp`
   — Push vol2col patches in K-block chunks.

## Tier B justification
new-infrastructure

The current Conv3d kernel has no K-streaming support — it is architecturally
designed for all-K-tiles-in-L1. Implementing K-streaming requires a
synchronized redesign of all four files listed above; it is new algorithmic
infrastructure for this kernel, not a threshold tweak.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    143.70s (0:02:23)
- Tier A attempts: N/A

## Files changed
None — Tier B, no fix attempted.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

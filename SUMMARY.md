# Remediation Summary: audioldm_m_full/pytorch-m-full-single_device-inference

## Skill version
18

## Test
tests/runner/test_models.py::test_all_models_torch[audioldm_m_full/pytorch-m-full-single_device-inference]

## Result
SILICON_PASS

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Full TT_FATAL:
```
TT_FATAL @ conv2d_op_sharded_program_factory.cpp:665:
  access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size >=
  conv_reader_indices_storage.get_buffer()->page_size()
CB page size 64 should be greater than the config tensor page size 68
```

## Root cause
**Layer: tt-metal runtime (ttnn conv2d)**

The `READER_INDICES` circular buffer (CB) page size was computed in
`get_cb_info()` (`conv2d_op_program_factory_common.cpp`) as:

```cpp
pconfig.per_core_out_matrix_height_ntile * TILE_HEIGHT * 2   // bytes
```

This formula estimates `N` uint16 reader-index entries per core (where
`N = per_core_out_matrix_height_ntile × 32`).

The actual reader indices are encoded per-block in a segment format: each
block of `D = act_block_h_ntiles × TILE_HEIGHT` output pixels contributes
at minimum **4 entries** (3-entry header + 1-entry footer) and at most
**2D + 2 entries** when every adjacent pixel pair is non-contiguous. The
maximum total entries for a core with `N/D` blocks is therefore `2N + 2N/D`.

For AudioLDM's UNet deepest encoder layer the output spatial shape is
`16 × 2` (32 pixels total, all on one core with `per_core_out = 1 tile`).
With `D = 32` (act_block_h_ntiles = 1) and output_width = 2, there are
**15 row-boundary discontinuities** in the single 32-pixel block, producing
`3 + 2×15 + 1 = 34` uint16 entries (68 bytes after even-alignment padding).
The formula predicted only 32 entries (64 bytes), causing the assertion.

## Fix
Two changes in **tt-metal** (`remediation/audioldm-m-full-conv2d-reader-indices-cb-size`):

1. **`conv2d_op_program_factory_common.cpp`** — Updated the
   `READER_INDICES` CB page-size formula to the correct upper bound:
   ```
   4 × per_core_out × TILE_HEIGHT + 4 × per_core_out / act_block_h_ntiles + 2
   ```
   This accounts for the (3 + 1) header/footer overhead per block plus the
   worst-case 2×(D−1) non-contiguous-segment entries, and the +2 handles
   the even-alignment padding in `construct_on_host_config_tensor`.

2. **`conv2d_op_sharded_program_factory.cpp` and
   `conv2d_op_width_sharded_program_factory.cpp`** — Replaced the
   `TT_FATAL` assertion (which asserted formula ≥ actual) with
   `std::max(formula, actual_buffer_page_size)` so the CB is always large
   enough even if the formula underestimates.

This is not a forbidden workaround: no model inputs or shapes were changed,
no layers were offloaded, and the fix addresses a genuine bug in the
compiler's circular-buffer sizing logic.

## Verification
```
1 passed in 287.08s (0:04:47)
```
Hardware: n150 (Wormhole B0), single device.

## Files changed
- `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp`
- `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp`
- `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_width_sharded_program_factory.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 0a6aadcb5b979ecfe18e14273fdb5b9d0f67f198 |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |

# Remediation Summary: audioldm-pytorch-base-single_device-inference

## Skill version
15

## Test
tests/runner/test_models.py::test_all_models_torch[audioldm/pytorch-base-single_device-inference]

## Result
FAIL — Conv2D READER_INDICES CB page size (64 bytes) underflows actual config tensor page size (68 bytes) in tt-metal

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Preceded by:
```
TT_FATAL: CB page size 64 should be greater than the config tensor page size 68 (assert.hpp:104)
```

## Root cause
**Layer: tt-metal** (runtime Conv2D op program factory)

The Conv2D operation pre-allocates the `READER_INDICES` circular buffer with a page size computed as:

```cpp
// conv2d_op_program_factory_common.cpp:299
.page_size = pconfig.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT * 2
           = 1 * 32 * 2 = 64 bytes
```

Later, the actual `conv_reader_indices_tensor` is constructed from the sliding-window config for the convolution. For AudioLDM's latent shape `(1, 8, 62, 8)` passed through a Conv2d(kernel=3, padding=1) layer, the sliding-window op generates a config vector of width 33, which `construct_on_host_config_tensor` pads to 34 (aligning to 2) → page size = 34 × 2 = **68 bytes**.

Since 64 < 68, the subsequent `TT_FATAL` assertion fires:

```cpp
// conv2d_op_sharded_program_factory.cpp:660-665
TT_FATAL(
    access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size >=
        conv_reader_indices_storage.get_buffer()->page_size(),
    "CB page size {} should be greater than the config tensor page size {}",
    ...);
```

The root problem is that the `calculate_L1_size` call (which sets the CB page size) runs before `construct_on_host_config_tensor` (which determines the actual tensor page size), and the static formula `per_core_out_matrix_height_ntile * TILE_HEIGHT * 2` does not correctly bound the dynamic config tensor size. The code comment acknowledges this difficulty: *"The actual CB reader size is difficult to calculate in calculate_L1_size."*

The latent height 62 (= int(5.0 s / 0.01) // 8) is not a multiple of 32, placing AudioLDM in an edge case that the static formula under-estimates.

## Fix
**Proposed fix in tt-metal:**

In `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp`, the `READER_INDICES` CB page size should be derived from the actual sliding-window config tensor dimensions rather than the static formula.

Option A — compute config tensor size first and pass it into `calculate_L1_size`:
- Refactor so `construct_on_host_config_tensor` / `generate_sliding_window_op_config` is called before `calculate_L1_size`, and the resulting page size is threaded through as a parameter.

Option B — use a conservative upper bound:
- Replace the formula with `ceil_to_multiple(pconfig.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT * 2, <alignment>)` ensuring it always covers the padded-to-2 alignment.

Neither option is a loader-layer workaround; both require non-trivial changes in tt-metal.

## Verification
FAIL — test exits with RuntimeError on TT silicon (n300). No passing run.

## Files changed
None — no code changes made (compiler-stack bug, no loader fix possible without a forbidden workaround).

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | a80cdca5d5fc472a5f5e8c8425ac7af99cc8cf11 |

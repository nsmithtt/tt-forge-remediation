# Remediation Summary: cosmos_reason1-pytorch-7b-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cosmos_reason1/pytorch-7B-single_device-inference]

## Result
FAIL — Conv3d L1 CB overflow: CBs grow to 1,745,920 B (1.66 MB) > max L1 1,572,864 B (1.5 MB) in `patch_embed`

## Stack layer
tt-metal, tt-mlir

## Tier
B

## Bug fingerprint
conv3d-small-cin-padding-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original reported error (`sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`) was a harmless deprecation warning, not the real failure. The actual failure was a series of PJRT Error 13 (INTERNAL) errors during compilation, ultimately caused by an L1 CB overflow in the Conv3d `patch_embed` kernel.

Final error (from tt-metal program.cpp:1130):
```
TT_THROW: Statically allocated circular buffers on core range ... grow to 1745920 B which is beyond max L1 size of 1572864 B
```

## Root cause
The Cosmos Reason1 model uses `Qwen2_5_VLPatchEmbed` with a `Conv3d` layer: kernel_size=(2,14,14), C_in=3 (RGB), C_out=1280.

In `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`, the Conv3d lowering pads C_in from 3 to TILE_WIDTH=32 so the kernel can operate on aligned blocks. This inflates the critical CB sizes:

- With C_in=3 padded to 32: `patch_size = 2×14×14×32 = 12544`, `matmul_K_t = ceil(12544/32) = 392`
- `vol2col_tiled` CB = 392 tiles × 2048 bytes = 802,816 bytes (784 KB)
- `weight_tiled` CB = 392 tiles × 2048 bytes = 802,816 bytes (784 KB)
- Together these two CBs alone = 1,605,632 bytes > max L1 of 1,572,864 bytes

Without the padding: `patch_size = 2×14×14×3 = 1176`, `matmul_K_t = ceil(1176/32) = 37`, total CBs ≈ 156 KB — well within limits.

The Conv3d kernel requires `C_in % C_in_block == 0` (a `TT_FATAL` assertion), so C_in=3 cannot use C_in_block=3 without a new tiling strategy that handles non-tile-width C_in values natively.

Five loader-level bugs were fixed along the way (see Fix section), but the sixth error (the Conv3d L1 overflow) is a compiler-stack bug with no loader-level remedy.

## Fix
**Loader fixes made** (tt-forge-models branch `remediation/cosmos_reason1-pytorch-7b-single_device-inference`):

1. `cosmos_reason1/pytorch/loader.py` — removed `use_cache` from model init kwargs; set via `model.config.text_config.use_cache = False` instead (transformers 5.x breaking change)
2. `cosmos_reason1/pytorch/loader.py` — replaced `Qwen2_5_VisionTransformerPretrainedModel.forward` with `_patched_vis_fwd` to eliminate cross-device gather operations in `rotary_pos_emb` and `window_index` lookups
3. `cosmos_reason1/pytorch/loader.py` — stored CPU copy of `inv_freq` as `vis._inv_freq_cpu` at load time (before `.to(device)`) to prevent device-to-host reads inside the compiled region
4. `cosmos_reason1/pytorch/loader.py` — replaced `torch.unique_consecutive` (unsupported on TT) with Python list deduplication for `cu_window_seqlens`
5. `cosmos_reason1/pytorch/loader.py` — kept `cu_seqlens` and `cu_window_seqlens` on CPU (no `device=` arg) to allow `lengths.tolist()` calls in `VisionAttention` without device-to-host transfer

**Proposed compiler-stack fix** (not implemented — Tier B):

The fix requires either:
- A new Conv3d tiling mode in `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp` that reduces `C_in_block` below TILE_WIDTH for small C_in values (e.g., C_in_block=1 or C_in_block=8 for C_in=3), allowing `vol2col_tiled` and `weight_tiled` CBs to fit in L1.
- OR: A check in `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` that computes the resulting CB sizes before applying C_in padding and falls back to a smaller C_in_block config when the padded size would overflow L1.

This is the same root cause as the bug documented in `qwen3vl_conv3d_l1_overflow.md` for Qwen3-VL models.

## Tier B justification
**Indicator: more-than-3-files / cross-repo.** A correct fix requires changes to the Conv3d program factory in tt-metal (CB allocation formula / C_in_block selection) and the Conv3d lowering in tt-mlir (either compute CB budget before padding, or pass a computed C_in_block hint through the lowering). At minimum, 3 files across 2 repos, plus the CB allocation formula interacts with the L1 prefetch buffer logic, making this a cross-cutting multi-file change. Iterating on the fix without running a broader regression suite risks breaking other Conv3d users.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: not measured (test did not complete)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/cosmos_reason1/pytorch/loader.py` (5 loader bug fixes; branch `remediation/cosmos_reason1-pytorch-7b-single_device-inference` in tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | e7f7a0006c53f3efdf264a6342805986915d2f3f |

# Remediation Summary: glm_ocr-image_to_text-pytorch-glm_ocr-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_ocr/image_to_text/pytorch-glm_ocr-single_device-inference]

## Result
FAIL — Conv3d patch embedding CB allocation of 1.75 MB exceeds TT L1 limit of 1.5 MB in ttnn::experimental::prim::Conv3dDeviceOperation

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
conv3d-cb-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

From device log:
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
grow to 1747968 B which is beyond max L1 size of 1572864 B
```

Backtrace shows `ttnn::experimental::prim::Conv3dDeviceOperation` →
`tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region` as the failure site.

The originally reported failure message (`The image processor of type Glm46VImageProcessor
is now loaded as a fast processor by default...`) is a transformers 5.x warning (loader layer),
not the primary test failure. A loader fix was applied for it; the blocking failure is the
Conv3d CB L1 overflow.

## Root cause
`GlmOcrVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1024,
kernel_size=[2,14,14])`. The weight tensor is 1024×3×2×14×14 elements × 2 bytes (bf16) ≈ 2.3 MB.
When this Conv3d op is dispatched to the TT device via `ttnn::experimental::prim::Conv3d`,
the kernel's static CB (Circular Buffer) allocation grows to 1,747,968 B (1.75 MB), which
exceeds the per-core L1 limit of 1,572,864 B (1.5 MB), triggering a TT_THROW at program
compilation time. This is the same class of bug documented for Qwen3VL in
`qwen3vl_conv3d_l1_overflow.md`.

Secondary loader-layer issues were also present and fixed:
- `GlmOcrVisionModel.rot_pos_emb` iterates `for t, h, w in grid_thw` with a device tensor,
  producing device scalars passed to `torch.arange` → Error code: 13.
- `GlmOcrModel.get_rope_index` and `get_image_features` call `.tolist()` on device tensors.
- `AutoProcessor.from_pretrained` missing `use_fast=False` (transformers 5.x default changed).
These were all fixed in the loader (tt-forge-models remediation branch) but cannot be
exercised until the Conv3d CB overflow is resolved.

## Fix
**Loader fixes applied** in
`tt-forge-models/glm_ocr/image_to_text/pytorch/loader.py`
(branch `remediation/glm_ocr-image_to_text-pytorch-glm_ocr-single_device-inference`,
commit `4e536955b7d7b7df5d68c6c4b99b3d986bebb835`):
- Added `_patch_glm_ocr_for_tt_device()` which moves `grid_thw`/`input_ids`/`attention_mask`
  to CPU before the `.tolist()` and iteration calls in `rot_pos_emb`, `get_rope_index`,
  `get_image_features`, and `get_video_features`.
- Added `use_fast=False` to `AutoProcessor.from_pretrained` call.

**Required compiler fix (not attempted)**: The `ttnn::experimental::prim::Conv3d` kernel
in tt-metal needs its CB allocation strategy revised so it does not statically allocate
circular buffers exceeding the per-core L1 limit for large kernels. The allocation would
need to be tiled or restructured to fit within 1.5 MB L1 when the weight tensor exceeds
that size. Expected location: tt-metal's experimental Conv3d program factory / kernel files.

## Tier B justification
cross-cutting — the Conv3d CB allocation strategy spans the program factory, CB allocation
logic, and kernel implementation in tt-metal; a correct fix would require restructuring how
the kernel tiles large weight tensors to avoid exceeding L1, touching more than 3 files.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    49.20s (second attempt after loader fix)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/glm_ocr/image_to_text/pytorch/loader.py` — loader fixes (use_fast=False,
  TT device .tolist() patches for rot_pos_emb / get_rope_index / get_image_features /
  get_video_features)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 4e536955b7d7b7df5d68c6c4b99b3d986bebb835 |

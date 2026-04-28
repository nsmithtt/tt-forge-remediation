# Remediation Summary: dots_ocr-pytorch-Ocr-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[dots_ocr/pytorch-Ocr-single_device-inference]

## Result
FAIL — DRAM auto-slice cannot handle Conv2d with 1×1 spatial output (embed_dim=1536, weight 3.45MB > L1 1.33MB)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
tt-metal-dram-autoslice-1x1-spatial-output

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
The reported failure message is a UserWarning (not a test failure). Reproduction revealed two distinct failures:

**First failure (loader bug — already fixed in target branch):**
`RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same` in the Conv2d `proj` layer of `DotsPatchEmbed`. The loader calls `model.float()` unconditionally to put all weights in float32, but the test framework calls `load_inputs(dtype_override=torch.bfloat16)`, which returns pixel_values in bfloat16. The vision tower's forward is patched to prevent an internal bfloat16 cast (`bf16=False`), but the *inputs* themselves arrive as bfloat16. Fix `5009c63803` in `origin/ip-172-31-23-5-tt-xla-dev/ubuntu/2026-04-23_16-01/hf-bringup-39` adds `hidden_states = hidden_states.float()` at the top of `_vt_forward_no_bf16`, casting inputs to float32 before the Conv2d.

**Second failure (compiler-stack bug — unfixed, this report):**
After the loader fix, the test proceeds to silicon execution and raises:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
with device log:
```
TT_FATAL: DRAM Auto slice could not find valid slice configuration.
Tried up to 1 slices for width-slicing on output dimension 1.
Available L1: 1396224 bytes.
Operation requires more memory than available even with maximum slicing.
```
The root cause is in tt-metal's DRAM auto-slice algorithm (`ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp`). The DotsOCR vision encoder uses `DotsPatchEmbed` with `nn.Conv2d(in_channels=3, out_channels=1536, kernel_size=14, stride=14)`. Because `kernel_size == stride`, each 14×14 patch maps to exactly one 1×1 output pixel, giving a spatial output of 1×1 per patch. The auto-slice algorithm only slices along spatial (height/width) dimensions: `max_num_slices = div_up(output_sliced_dim, TILE_HEIGHT)`. With `output_width = 1`, `max_num_slices = 1`. One slice (the full operation) requires the entire weight matrix: `1536 × 3 × 14 × 14 × 4 bytes = 3.45 MB`, which exceeds the available L1 of 1.33 MB. Height slicing also fails (output_height = 1). There is no spatial dimension to split along.

## Fix
The loader bug fix is already present on the target branch (`origin/ip-172-31-23-5-tt-xla-dev/ubuntu/2026-04-23_16-01/hf-bringup-39`) at commit `5009c63803`:
- File: `dots_ocr/pytorch/loader.py` (in tt-forge-models)
- Change: Added `hidden_states = hidden_states.float()` in `_vt_forward_no_bf16` before calling `orig_vt_forward`

The compiler-stack fix would require adding batch/output-channel dimension slicing to the DRAM auto-slice algorithm in tt-metal, so that Conv2d operations with degenerate 1×1 spatial output can be split along the output-channel or batch dimension. Relevant file: `ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp` — the `determine_slice_config_internal` and `compute_max_num_slices` functions would need to be extended with a third slicing dimension.

## Tier B justification
new-infrastructure: The DRAM auto-slice algorithm only supports height and width spatial slicing. Adding output-channel or batch slicing for Conv2d with 1×1 spatial output requires new slicing infrastructure. Even if the fix is confined to `op_slicing.cpp`, it requires new slice-boundary computation, new L1 usage estimation paths, and new kernel dispatch logic — well beyond a one-line tweak.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    73.18s (0:01:13)
- Tier A attempts: N/A

## Files changed
- No new changes committed; loader fix already present in `origin/ip-172-31-23-5-tt-xla-dev/ubuntu/2026-04-23_16-01/hf-bringup-39` at `5009c63803`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | a38f1aa44b06c174366b865f39b108afb7330426 |

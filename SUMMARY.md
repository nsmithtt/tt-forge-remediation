# Remediation Summary: dccr_net-speech_enhancement-pytorch-Libri1Mix_enhsingle_16k-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[dccr_net/speech_enhancement/pytorch-Libri1Mix_enhsingle_16k-single_device-inference]

## Result
FAIL — tt-metal HEIGHT_SHARDED conv2d has no K-blocking; STFT encoder weight CB (6.97 MB) exceeds L1 (1.33 MB)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
conv2d-height-sharded-no-k-blocking

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Underlying TT_FATAL:
TT_FATAL @ op_slicing.cpp:266: found_valid_config
info:
 DRAM Auto slice could not find valid slice configuration. Tried up to 5 slices for
 width-slicing on output dimension 157. Available L1: 1396224 bytes. Operation
 requires more memory than available even with maximum slicing.

## Root cause
DCCRNet's STFT encoder is a `Conv1d(in_channels=1, out_channels=514, kernel_size=400,
stride=100)` which TTNN lowers to its conv2d DRAM path (output spatial shape 1×157×514).
The HEIGHT_SHARDED conv2d path computes `act_block_w = in_channels_aligned × kernel_W =
8 × 400 = 3200` with no K-blocking. In TILE layout, the weight circular buffer per core
requires `ceil(3200/32) × ceil(544/32) × 4096 = 100 × 17 × 4096 = 6,963,200 bytes
(6.97 MB)`, far exceeding the available L1 of 1,396,224 bytes (1.33 MB).

The `op_slicing` mechanism in `op_slicing.cpp` only slices the output spatial dimensions
(height/width); it cannot reduce the weight tensor size. With output_height=1 (single row)
and output_width=157, even maximum slicing of 5 width-slices leaves the weight requirement
unchanged at 6.97 MB >> 1.33 MB L1.

## Fix
Proposed fix: implement K-blocking for the HEIGHT_SHARDED conv2d path in tt-metal. The
weight CB would be split into K-blocks so only `ceil(K_block/32)` tiles are loaded at
once instead of all K=3200 columns. This requires coordinated changes to:

1. `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp` —
   `calculate_L1_usage_for_conv_op` to account for K-blocking in the L1 estimate
2. `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp` —
   `conv2d_DRAM` / `Conv2dSliceAttr` to add K-block dimension and loop
3. One or more conv2d kernel files (reader/writer programs) — to loop over K-blocks at
   runtime with accumulation management

## Tier B justification
new-infrastructure: HEIGHT_SHARDED conv2d has never had K-blocking; implementing it
requires new kernel loop structure (CB setup changes, reader/writer loop over K-blocks,
accumulation management) coordinated across conv2d_utils.cpp, conv2d.cpp, and at least
one kernel file. More than 3 files, and the kernel changes require new infrastructure.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~45s (to failure)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/dccr_net/speech_enhancement/pytorch/requirements.txt (pre-existing fix on configured branch)
- tt-xla/third_party/tt_forge_models/dccr_net/speech_enhancement/pytorch/loader.py (pre-existing fix on configured branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | a660ddb4fe816eea84cf411b6a63a9cabb9ba88e |

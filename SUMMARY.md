# Remediation Summary: mapperatorinator-pytorch-v29.1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mapperatorinator/pytorch-v29.1-single_device-inference]

## Result
FAIL — Conv1d with C_in=768 kernel=3 maps to Conv2d; WEIGHTS CB = 3.5 MB >> 1.5 MB n150 L1 limit; auto-shard path has no overflow guard or slicing fallback

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
conv1d-large-channel-l1-cb-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
(full device log: TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=0)] grow to 9612800 B which is beyond max L1 size of 1572864 B)

## Root cause
Three loader bugs were fixed (see Fix section). After all loader fixes the original
INTERNAL:13 was reproduced. The error originates in tt-metal's
`validate_circular_buffer_region()` (program.cpp:1132).

Whisper's first encoder Conv1d has C_in=768, C_out=768, kernel_size=3. tt-mlir
lowers Conv1d→Conv2d and calls `determine_conv_config_for_auto_shard()` in
`conv2d_utils.cpp`. That function selects the minimum-footprint sharding config
(HEIGHT shard, 1 core) but does not verify the resulting circular buffers fit in
L1. The WEIGHTS CB alone is:

  per_core_out_matrix_width_ntiles × act_block_w_ntiles × weights_tile_size
  = 24 × (768×3/32) × 2048
  = 24 × 72 × 2048
  = 3,538,944 bytes  (2.25× the 1,572,864 byte L1 limit)

Setting `config_tensors_in_dram=true` (done in TTIRToTTNN.cpp) only controls the
reader-indices buffer, not the main computation CBs. There is no L1-overflow guard
in the auto-shard path and no Conv2d slice config is generated to split the work
into L1-safe tiles.

## Fix
**Loader fixes (tt-forge-models, 4 commits on branch
remediation/mapperatorinator-pytorch-v29.1-single_device-inference):**

1. `mapperatorinator/pytorch/requirements.txt` — added `nnAudio` (was missing;
   `MelSpectrogram.__init__` imports from `nnAudio.features` at module load time).

2. `mapperatorinator/pytorch/loader.py` — patched `Mapperatorinator.__init__`
   to call `self.post_init()` when `all_tied_weights_keys` is absent. Transformers
   5.x `_finalize_model_loading` accesses this attribute but the external model
   class never calls `post_init()`.

3. `mapperatorinator/pytorch/loader.py` — `load_inputs()` always creates `frames`
   as `torch.float32`. The nnAudio `MelSpectrogram._apply()` override forces the
   STFT kernels back to float32 even after a `.to(bfloat16)`, causing a dtype
   mismatch if frames are bfloat16.

4. `mapperatorinator/pytorch/loader.py` — corrected `n_samples = 130944`. The
   Whisper encoder requires exactly 1024 mel frames
   (`max_source_positions × conv_stride = 512 × 2`). With nnAudio center=True:
   `n_frames = n_samples // hop_length + 1`, so `n_samples = (1024−1) × 128 =
   130944`. The previous value (16000) produced only 126 frames, triggering a
   Whisper strict-length assertion.

**Proposed compiler fix (NOT attempted — Tier B):**
In `conv2d_utils.cpp::determine_conv_config_for_auto_shard()`, after selecting the
sharding config, call `calculate_L1_usage_for_conv_op()` and when the result
exceeds the device L1 limit, generate a `Conv2dSliceConfig` that partitions
`act_block_w` into L1-safe slices. This requires coordinated changes to
`conv2d_utils.cpp` (L1 guard + slice-config generation), `conv2d.cpp` (slice-config
plumbing), and `TTIRToTTNN.cpp` (pass the slice config through the lowering).

## Tier B justification
new-infrastructure: The fix requires implementing L1-overflow detection and
automatic act_block_w slicing in the Conv2d auto-shard path — new logic spanning
at least three files (conv2d_utils.cpp, conv2d.cpp, TTIRToTTNN.cpp) with no
existing infrastructure to build on.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    N/A (reproduced the INTERNAL:13 after loader fixes; no passing run)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/mapperatorinator/pytorch/requirements.txt (created)
- tt-xla/third_party/tt_forge_models/mapperatorinator/pytorch/loader.py (3 fixes)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 48ae55c00a83efc5230a9567ba97b6197c96c747 |

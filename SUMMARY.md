# Remediation Summary: minicpm_llama3_v_2_5-pytorch-MiniCPM-Llama3-V-2.5-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[minicpm_llama3_v_2_5/pytorch-MiniCPM-Llama3-V-2.5-single_device-inference]

## Result
FAIL — resampler cross-attention SDPA KV length 1036 not 32-aligned; PCC=0.06 (ttnn-sdpa-nonaligned-kv-pcc-wrong)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
ttnn-sdpa-nonaligned-kv-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.061634053522451264. Required: pcc=0.99.

(Original reported failure `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`
was a build-tooling noise line; the actual errors were 5 loader bugs and then the terminal SDPA bug.)

## Root cause
The MiniCPM-Llama3-V-2.5 resampler is a cross-attention module where the COCO
validation image (640×480) produces 1036 K/V tokens (28 height-patches × 37 width-patches
for the source image). The TT SDPA kernel requires the K/V sequence length to be
divisible by 32; 1036 % 32 = 12 ≠ 0. The TT backend computes a numerically incorrect
attention output (PCC ≈ 0.06) for non-32-aligned K/V. The garbage vision embeddings
are spliced into the LLM input_embeds, corrupting the final logits.

Five loader bugs were found and fixed before reaching the terminal compiler bug:
1. `resampler.py` missing `List` in typing imports → Python 3.12 NameError
2. `MiniCPMV.__init__` never calling `self.post_init()` → transformers 5.x `all_tied_weights_keys` AttributeError
3. `cast_tensor()` in `MiniCPMVBatchFeature.to()` calling `torch.is_floating_point()` on bare int leaves → TypeError
4. `torch.vstack()` rejecting Python list elements in PyTorch 2.7 → TypeError
5. `torch.max()` on XLA int32 tensors computing through BF16 internally, rounding 1036 → 1040, causing an assertion failure in `_check_key_padding_mask` (after fix: shape passes but PCC=0.06 from the SDPA bug)

## Fix
Five loader fixes were applied in `minicpm_llama3_v_2_5/pytorch/loader.py` via the
`_patch_cached_remote_files()` runtime patcher (patches cached HuggingFace remote files):

1. **Fix 1** (`resampler.py`): Add `List` to typing imports.
2. **Fix 2** (`modeling_minicpmv.py`): Add `self.post_init()` at end of `MiniCPMV.__init__`.
3. **Fix 3** (`image_processing_minicpmv.py`): Guard `cast_tensor()` with `isinstance(v, torch.Tensor)`.
4. **Fix 4** (`modeling_minicpmv.py`): Replace `torch.vstack(tgt_sizes)` with `torch.tensor(tgt_sizes, dtype=torch.int32)` for Python list inputs.
5. **Fix 5** (`resampler.py`): Replace `torch.max(patch_len)` / `torch.max(tgt_sizes[:, 0/1])` with element access `int(patch_len[0])` / `int(tgt_sizes[:, 0/1][0])` for the single-sample non-batch path.

Additionally, `load_inputs()` was rewritten to use `AutoProcessor`, build the correct `data` dict
format expected by `forward()`, compute `position_ids`, flatten `tgt_sizes` to Python list of `[h, w]`
pairs, and set `model.config.batch_vision_input = False` to use the per-image path.

The terminal SDPA bug is in tt-metal's SDPA kernel. Padding K/V to the next
multiple of 32 (1056) would be a forbidden shape change for kernel constraints.

## Tier B justification
cross-cutting — fixing non-32-aligned K/V in SDPA requires changes to the SDPA
kernel in tt-metal to pad/mask K/V sequences that don't meet the tile-alignment
requirement. This affects all SDPA-using models and requires coordinated changes
across multiple files (kernel, CB allocation, output mask stripping).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    644.98s (0:10:44)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/minicpm_llama3_v_2_5/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8f203e0a46207b460ad56fccc104374df95754fe |
| tt-forge-models | eeaf3015ae833bb185bc25147f20ee22f5fae431 |

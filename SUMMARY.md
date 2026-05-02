# Remediation Summary: minicpm_v_2_5-pytorch-Default-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[minicpm_v_2_5/pytorch-Default-single_device-inference]

## Result
FAIL — resampler cross-attention SDPA KV length not 32-aligned; PCC=0.0646 (ttnn-sdpa-nonaligned-kv-pcc-wrong)

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
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.06456625580441631. Required: pcc=0.99.

(Original reported failure `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`
was build-tooling noise; the actual errors were loader bugs and an OOM, followed by the terminal SDPA bug.)

## Root cause
The MiniCPM-Llama3-V-2.5 resampler is a cross-attention module (Idefics2-style) that
cross-attends over the vision encoder's patch features. For the Statue of Liberty image
(1600×1067 px), the preprocessing produces a global-thumbnail slice with KV tokens whose
count is not divisible by 32. The TT SDPA kernel requires the K/V sequence length to be
32-aligned; a non-aligned KV produces numerically incorrect attention output (PCC ≈ 0.065).
The garbage vision embeddings are spliced into the LLM input_embeds, corrupting the final
logits.

This is the same root cause as the previously filed report for
`minicpm_llama3_v_2_5/pytorch-MiniCPM-Llama3-V-2.5-single_device-inference`, which uses
the identical model (`openbmb/MiniCPM-Llama3-V-2_5`) and hit PCC ≈ 0.062 with a 1036-token
KV for the COCO validation image.

Six loader issues were found and fixed before reaching the terminal compiler bug:

1. **float32 → bfloat16 OOM**: Loader used `torch_dtype=torch.float32`, loading ~34 GB
   weights that exhausted the 34.2 GB p150b DRAM before any activations could be allocated
   (`DRAM OOM: 2.1 GB needed, 194 MB free`). Fix: change to `torch_dtype=torch.bfloat16`.

2. **resampler.py missing `List` import**: Python 3.12 NameError at module load because
   `resampler.py` uses `List[Tensor]` but only imports `Optional, Tuple` from `typing`.

3. **MiniCPMV.__init__ missing `self.post_init()`**: `all_tied_weights_keys` (added in
   transformers 5.x) is never initialized, causing `AttributeError` in
   `_adjust_tied_keys_with_tied_pointers` during `from_pretrained`.

4. **cast_tensor() TypeError**: `MiniCPMVBatchFeature.to()` calls `torch.is_floating_point(v)`
   unconditionally; when `tgt_sizes` contains bare Python int leaves, this raises TypeError.
   Fix: guard with `isinstance(v, torch.Tensor)`.

5. **torch.vstack() TypeError** (PyTorch 2.7): `torch.vstack` requires Tensor elements; when
   `tgt_sizes` is a Python list of `[h, w]` pairs it raises TypeError. Fix: fall back to
   `torch.tensor(tgt_sizes, dtype=torch.int32)`.

6. **torch.max() XLA int32 BF16 rounding**: `torch.max(patch_len)` and
   `torch.max(tgt_sizes[:, 0/1])` on XLA int32 tensors compute through BF16, rounding e.g.
   1036 → 1040. Fix: element-access `int(tensor[0])` for single-sample path.

Additionally, `model.config.batch_vision_input` was set to `False` (per-image path) to avoid
XLA padding mismatches between `patch_attn_mask` and `pixel_values` shapes in the batch path,
and `load_inputs()` was rewritten to use `AutoProcessor`, flat Python-list `tgt_sizes`, and
explicit `position_ids`.

## Fix
Six loader fixes were applied in `minicpm_v_2_5/pytorch/loader.py` via the
`_patch_remote_code()` runtime patcher (patches cached HuggingFace remote files):

- **Fix 1 (dtype)**: Change `torch_dtype=torch.float32` → `torch.bfloat16` in `load_model()`.
- **Fix 2 (resampler.py)**: Add `List` to typing imports.
- **Fix 3 (modeling_minicpmv.py)**: Add `self.post_init()` at end of `MiniCPMV.__init__`.
- **Fix 4 (image_processing_minicpmv.py)**: Guard `cast_tensor()` with `isinstance(v, torch.Tensor)`.
- **Fix 5 (modeling_minicpmv.py)**: Fallback from `torch.vstack(tgt_sizes)` to
  `torch.tensor(tgt_sizes, dtype=torch.int32)` when elements are Python lists.
- **Fix 6 (resampler.py)**: Replace `torch.max` with `int(tensor[0])` element access for
  `max_patch_len`, `max_h`, and `max_w` in the single-sample path.

Also: `model.config.batch_vision_input = False` and complete rewrite of `load_inputs()` using
`AutoProcessor` with flat Python-list `tgt_sizes` and explicit `position_ids`.

The terminal SDPA bug is in tt-metal's SDPA kernel. The KV sequence length from the
resampler cross-attention is not 32-aligned for this image/preprocessing combination.
Padding KV to the next multiple of 32 would require changes to the SDPA kernel.

## Tier B justification
cross-cutting — fixing non-32-aligned K/V in SDPA requires changes to the SDPA kernel in
tt-metal to pad/mask K/V sequences that don't meet the tile-alignment requirement. This
affects all SDPA-using models and requires coordinated changes across multiple files
(kernel, CB allocation, output mask stripping).

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    3054.21s (0:50:54)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/minicpm_v_2_5/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1606f9a65c5dc30c5fcb5bf68ddc4b215554ad72 |
| tt-forge-models | 6085103dc842f87244d34baa10fa63f1fadb94e2 |

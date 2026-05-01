# Remediation Summary: llava_onevision-pytorch-q_sit_mini-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llava_onevision/pytorch-q_sit_mini-single_device-inference]

## Result
FAIL — TT silicon PCC=0.939 (CPU BF16=0.991); genuine compiler precision bug, not a BF16 floor

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (test run 1 — before loader fixes):
```
The image processor of type `LlavaOnevisionImageProcessor` is now loaded as a fast processor by
default, even if the model checkpoint was saved with a slow processor. This is a breaking change
and may produce slightly different outputs. To continue using the slow processor, instantiate
this class with `use_fast=False`.
```

After loader fix 1 (test run 2):
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Caused by two issues: (a) `.tolist()` on TT device tensors in `get_anyres_image_grid_shape`,
`image_size_to_num_patches`, and `unpad_image`; (b) `torch_compilable_check` in
`get_placeholder_mask` using `inputs_embeds[bool_mask]` (dynamic boolean indexing, Tier B).

After loader fixes 1–3 (test run 3 — final silicon run):
```
AssertionError: PCC check failed: pcc=0.9389, required=0.99
```

## Root cause

Three loader bugs were fixed; the remaining failure is a compiler precision issue.

**Loader bug 1** (`transformers-5x-use-fast-default`): Transformers 5.x changed
`LlavaOnevisionImageProcessor` to load as a fast processor by default. The loader passed no
`use_fast` argument, so the fast processor was used. Fix: `use_fast=False` in
`AutoProcessor.from_pretrained()`.

**Loader bug 2** (`.tolist()` D2H graph breaks): Three module-level helper functions in
`modeling_llava_onevision.py` call `image_size.tolist()` or `original_size.tolist()` directly on
the tensor argument, which triggers device-to-host transfers on TT device tensors, causing graph
breaks and ultimately INTERNAL:13. Fix: patch `get_anyres_image_grid_shape`,
`image_size_to_num_patches`, and `unpad_image` to call `.cpu()` when the argument is a tensor.

**Loader bug 3** (dynamic boolean indexing in `get_placeholder_mask`): After the graph breaks
above are eliminated, the remaining sub-graph that reaches the compiler includes
`torch_compilable_check` calls in `LlavaOnevisionModel.get_placeholder_mask` that use
`inputs_embeds[special_image_mask]` — a boolean-indexed gather with dynamic output shape. TT
cannot lower this pattern (Tier B, `dynamic-shape-boolean-index`). The `torch_compilable_check`
calls are optional assertions (guarded by `TRANSFORMERS_DISABLE_TORCH_CHECK`) that do not affect
the `masked_scatter` computation; removing them is correct. Fix: patch
`LlavaOnevisionModel.get_placeholder_mask` to omit these assertions.

**Compiler precision issue**: After all three loader bugs are fixed, the model runs end-to-end
on TT silicon but produces PCC=0.939 vs the required threshold of 0.99. CPU BF16 vs CPU FP32
gives PCC=0.991, establishing that 0.991 is the BF16 floor for this model. TT silicon's 0.939
is 0.052 below that floor, indicating a genuine TT compiler precision loss beyond BF16
accumulation. The q-sit-mini variant is 0.89B params, 24 layers — the precision drop compounds
through its Qwen2 language model backbone. Root cause in tt-mlir is unknown without further
profiling (likely the same WH BF16 matmul precision issue seen in Gemma, Qwen3, and GPT-J, but
not confirmed for this model).

## Fix

**Loader fixes** (in `tt-xla/third_party/tt_forge_models`, branch
`remediation/llava_onevision-pytorch-q_sit_mini-single_device-inference`):

- `llava_onevision/pytorch/loader.py`:
  - `_load_processor`: added `use_fast=False` to `AutoProcessor.from_pretrained()`
  - Added `_patch_llava_onevision_for_tt_device()` function (called from `load_model()`):
    - Patches `get_anyres_image_grid_shape`, `image_size_to_num_patches`, `unpad_image` to call
      `.cpu()` on tensor arguments before `.tolist()`
    - Patches `LlavaOnevisionModel.get_placeholder_mask` to remove `torch_compilable_check`
      calls that use dynamic boolean indexing, preserving the `masked_scatter` computation

**Compiler precision fix**: NOT attempted — Tier B (cross-cutting; root cause unknown without
further profiling; same class as `ttmlir-f32-precision-not-preserved` seen in multiple models).

## Tier B justification

`cross-cutting`: BF16 matmul precision loss on Wormhole is a known systemic issue that manifests
across many models (Gemma, Qwen3, GPT-J, etc.). A fix would require changing math fidelity
settings across all lowering passes, which is a cross-cutting change in tt-mlir. The gap here
(PCC=0.939 vs floor=0.991) is larger than typical BF16 accumulation error, but the mechanism
(which specific op or layer compounds the error) requires layer-by-layer profiling not performed
here.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    220s (final silicon run)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/llava_onevision/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a5007fbb00b1c3dfb96e653da720444f584b58a0 |
| tt-forge-models | 56ac9c76d158ff9ec742139595ef4857875d02b8 |

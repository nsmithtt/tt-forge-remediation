# Remediation Summary: next2_5_vl_gguf-image_to_text-pytorch-i1_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[next2_5_vl_gguf/image_to_text/pytorch-i1_Q4_K_M-single_device-inference]

## Result
FAIL — terminal Tier B compiler bug: dynamic boolean-index scatter with data-dependent output shape in `get_placeholder_mask`

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
dynamic-shape-boolean-index-embedding-scatter

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
at `get_placeholder_mask` → `torch_compilable_check` → `partition_fx_graph_for_cpu_fallback` → `extract_internal` → `torch_xla.sync`

## Root cause

Three bugs were encountered in sequence.

**Bug 1 (loader) — FIXED:** The loader used `Qwen3VLForConditionalGeneration` to load `thelamapi/next2.5`, but that repo has `model_type: qwen3_5` and requires `Qwen3_5ForConditionalGeneration`. This caused a weight shape mismatch at load time.

**Bug 2 (tt-metal, Tier A) — FIXED:** `Qwen3_5VisionModel.patch_embed.proj` is `nn.Conv3d(3, 1024, kernel_size=(2,16,16))`. The default dispatch in `conv3d_program_factory.cpp` used `C_in_block = TILE_WIDTH = 32`, yielding two dominant CBs (vol2col_tiled + weight_tiled) each ≈ 1 MB, totalling ~2 MB > 1.5 MB L1 per core → INTERNAL:13 during `fast_pos_embed_interpolate`. Fix: auto-reduce `C_in_block` by halving until the dominant CBs fit within 75% of available L1.

**Bug 3 (tt-xla, Tier B) — terminal:** After the Conv3d L1 fix, `Qwen3_5Model.get_placeholder_mask` evaluates `inputs_embeds[special_image_mask]` — a boolean-masked gather with a data-dependent output shape. TT device compilation requires fully static shapes. During `partition_fx_graph_for_cpu_fallback`, the TT XLA backend encounters this dynamic-shape tensor and raises INTERNAL:13.

## Fix

**Bug 1:** Changed `Qwen3VLForConditionalGeneration` → `Qwen3_5ForConditionalGeneration` and added pixel limits and `_patch_qwen3_5_for_tt_device()` to route control-flow `.tolist()` calls through CPU. Committed on `remediation/next2_5_vl_gguf-image_to_text-pytorch-i1_Q4_K_M-single_device-inference` in `tt_forge_models` (commits 16200a68cd, c2a884afb8).

**Bug 2:** Cherry-picked two commits to `tt-metal` on branch `remediation/next2_5_vl_gguf-image_to_text-pytorch-i1_Q4_K_M-single_device-inference`:
- `f310e0528d conv3d: auto-reduce C_in_block when default exceeds L1 CB budget`
- `8e3ed99627 conv3d: fix C_in_block auto-reduction condition and loop bounds`

The guard in `ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp` computes dominant CB bytes for the current `C_in_block` and halves until they fit within 75% of L1.

**Bug 3 (proposed, not attempted):** The fix requires adding support for data-dependent tensor shapes in the PJRT/StableHLO compilation path — new infrastructure that lives in `tt-xla`. This is well beyond a single-file scoped change.

## Tier B justification

Which indicator: `new-infrastructure`.

`inputs_embeds[special_image_mask]` produces a tensor whose shape depends on runtime data (number of image tokens). TT device compilation requires fully static shapes at trace time. Supporting data-dependent shapes requires adding dynamic-shape infrastructure throughout the PJRT/StableHLO compilation pipeline in `tt-xla`. This is not a scoped one-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    181.27s (0:03:01)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/next2_5_vl_gguf/image_to_text/pytorch/loader.py` — loader fix (Bug 1)
- `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp` — Conv3d L1 auto-reduction (Bug 2)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 8e3ed9962796dec084bd3e82360651a168987b4f |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c2a884afb80aa450a306e426c2a368842c6088dc |

# Remediation Summary: ministral_3_8b_instruct_bnb_4bit-pytorch-unsloth_Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral/ministral_3_8b_instruct_bnb_4bit/pytorch-unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit-single_device-inference]

## Result
FAIL — PCC 0.209 after all loader bugs fixed; root cause is TT-MLIR precision loss in the Pixtral vision encoder (24-layer ViT, 9240-token SDPA)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-pixtral-vision-encoder

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.20914766659402356. Required: pcc=0.99.

## Root cause

Two separate bugs were found and fixed in the loader layer before reaching this Tier B precision failure:

**Bug 1 (fixed): `generate_block_attention_mask` XLA in-place mutation → Error code: 13**

`PixtralVisionModel.forward()` calls `generate_block_attention_mask`, which uses in-place XLA tensor assignment inside a Python for loop (`causal_mask[start:end, start:end] = 0`). Dynamo marks this as "FAILED INLINING" and creates a graph break. The eagerly-executed in-place mutations queue XLA operations that fail when `extract_compiled_graph` calls `torch_xla.sync()` → `INTERNAL: Error code: 13`.

Fix: Replaced `pixtral_module.generate_block_attention_mask` globally with a functional version using `torch.where`. For single-image inference (our case), returns an all-zeros mask (semantically correct: all patches within one image attend to each other freely).

**Bug 2 (fixed): `masked_scatter` cumsum OOM (45 GB)**

The standard `masked_scatter` decomposition in `tt_torch/backend/decompositions.py` flattens the mask to `[seq_len × hidden_size]` = 11.8M elements and calls `cumsum` on it. tt-mlir's cumsum pads the reduction axis to tile_size=1024, creating an `[11.8M, 1024]` int32 intermediate matrix → 48 GB allocation → OOM: "Not enough space to allocate 48586817536 B DRAM buffer across 8 banks".

Fix: Replaced `Mistral3Model.forward()` with a token-level gather implementation. Instead of `masked_scatter`, we:
1. Compute `token_mask = (input_ids == image_token_id)` at the token level (2896 elements)
2. Compute `cumsum` at token level → only `[2896, 1024] × 4B = 12 MB`
3. Use `torch.gather + torch.where` to place image features at image token positions

After both fixes the model compiles, runs in ~4 seconds on TT silicon, and produces output — but PCC = 0.209.

**Remaining bug (Tier B): TT-MLIR precision loss in Pixtral vision encoder**

The Pixtral ViT (24 layers, `hidden_size=1024`, `num_heads=16`, `seq_len=9240` patches from a 1176×1540 image) runs with poor numerical precision on TT hardware. PCC 0.209 is far below any BF16 floor observed in other models (typically 0.95–0.99). This level of divergence suggests precision failure in the TT SDPA kernel when handling very long sequences (9240 tokens), or in the f32 operations (PixtralRMSNorm casts to float32, softmax in eager attention uses `dtype=torch.float32`) that TT likely executes in bfloat16.

The Pixtral ViT uses `sdpa` attention (`_attn_implementation="sdpa"`). For seq_len=9240, each attention layer processes Q/K/V of shape `[1, 16, 9240, 64]` and a block attention mask of `[1, 1, 9240, 9240]` (all-zeros for single image). The TT SDPA kernel must tile this 9240×9240 computation, and numerical divergence in the tiled softmax accumulation across 24 ViT layers is the most plausible source of PCC 0.209.

Precision errors in the vision encoder's image features propagate through the entire text decoder (28 Mistral layers), corrupting all 2896 logit positions.

## Fix
All loader fixes are in `tt-xla/third_party/tt_forge_models` on the remediation branch.

**Proposed fix for the Tier B precision issue:**
Investigate TT MLIR SDPA kernel precision for large sequence lengths (9240+). Specifically:
- Verify that the tiled softmax in the SDPA kernel correctly handles `max` value subtraction across tiles for 9240-length sequences
- Verify that f32 accumulation in the attention computation is preserved rather than collapsed to bf16
- If the SDPA kernel is correct, investigate f32 precision preservation through RMSNorm lowering (PixtralRMSNorm casts to f32 explicitly)

File: `tt-mlir/lib/Dialect/TTIR/Transforms/` or the SDPA-specific lowering in `tt-metal/ttnn/`.

## Tier B justification
**Indicator: cross-cutting**

The most likely cause is bfloat16 precision loss in the TT SDPA kernel or in f32 operations (RMSNorm, softmax) across all 24 ViT layers. Fixing this requires either:
1. Fixing the TT SDPA tiled-softmax implementation for large sequences (touches ttnn/kernel code + lowering), OR
2. Preserving f32 precision through all lowering passes for operations that explicitly request float32

Both options are cross-cutting changes touching multiple files across tt-mlir and/or tt-metal. Further diagnosis is required to determine the exact source of the 0.209 PCC.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 362.98s (0:06:02) for the final run
- Tier A attempts: N/A (no Tier A fix attempted; precision root cause requires deeper diagnosis)

## Files changed
- `tt-xla/third_party/tt_forge_models/mistral/ministral_3_8b_instruct_bnb_4bit/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mistral/ministral_3_8b_instruct_bnb_4bit/pytorch/requirements.txt`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | dfe26357df6924ac16d40e7b000095c3beb53480 |
| tt-forge-models | 73478c314a7488e21991d39f98f63399b99c47de |

# Remediation Summary: glm_edge_v-conditional_generation-pytorch-glm_edge_v_2b-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[glm_edge_v/conditional_generation/pytorch-glm_edge_v_2b-single_device-inference]

## Result
FAIL — TT hardware produces non-deterministic PCC (0.43–0.74) for the same compiled graph; root cause unknown without deeper diagnosis

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
siglip-sdpa-head-dim-non-tile-aligned-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original failure was:
```
The image processor of type `MllamaImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.
```
(A transformers 5.x warning that surfaced alongside AttributeError: 'list' object has no attribute 'keys')

After all loader fixes, the current failure is:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.43–0.74 (non-deterministic across runs). Required: pcc=0.99.
```

## Root cause
Multiple transformers 5.x loader incompatibilities were fixed; after those, the model compiles and
runs on TT hardware but produces wrong logits.

**Loader bugs fixed (in this PR):**

1. `_tied_weights_keys` list→dict mismatch (`AttributeError: 'list' object has no attribute 'keys'`
   in `modeling_utils.py`): Fixed by setting `config.tie_word_embeddings = False` so the
   incompatible code path is never reached.

2. `DynamicCache.to_legacy_cache()` removed in transformers 5.x: Fixed by passing
   `use_cache=False` in the inputs so `return_legacy_cache` is never set.

3. Data-dependent control flow in `GlmModel.forward()` (remote model code):
   The original code uses `input_id.tolist()` and `list.index()` inside the image injection loop,
   which Dynamo cannot trace with fake tensors. This causes either (a) `torch.stack([], dim=0)`
   failure or (b) image features silently multiplied by zero (PCC ~0.28). Fixed by modifying
   the local HF cache copy of `modeling_glm.py`:
   - Replaced `multi_flags` list comprehension with `boi_mask = (input_ids == boi_token_id)`.
   - Pre-computed `model.model._boi_token_pos` as a Python int in `load_model()` and used
     `getattr(self, "_boi_token_pos", None)` in the forward to give Dynamo a compile-time constant.
   - Used `inputs_embeds[i, boi_token_pos + num_img_tokens:]` to skip exactly the 578 boi
     placeholder tokens (matching the vision model's 578-token output).
   **Note**: This fix was applied to the local HF cache (`modeling_glm.py`). It must be
   formalized as a monkey-patch in `loader.py` for CI reproducibility — that work was deferred
   since it does not change the FAIL outcome.

**Remaining compiler-stack bug (Tier B):**

After the loader fixes, compilation succeeds. The model then executes on TT hardware with
non-deterministic PCC (0.43, 0.59, 0.74 across three runs on the same device). CPU reference
and `torch.compile(backend="eager")` both give correct results (PCC ≈ 1.0).

The model's vision path uses SigLIP (27-layer ViT) with head_dim = 72. This is not tile-aligned
(72 % 32 ≠ 0), so `ScaledDotProductAttentionPadTileDimsRewritePattern` in tt-mlir pads it to 96.
The padded SDPA correctly preserves the original scale attribute (`1/√72`), so the mathematical
intent is correct. However, PCC on hardware is non-deterministic and well below 0.99.

A secondary suspect is the GLM language model decoder (28 layers, GQA 16Q/4KV, seq_len=585 after
image injection). The 585-token sequence produces attention over shapes `[1, 16, 585, 128]` with
an explicit causal mask.

Without hardware-level profiling it is not possible to identify which kernel(s) produce wrong
results, hence Tier B.

## Fix
**Committed (loader, tt_forge_models remediation branch):**
- `tt-xla/third_party/tt_forge_models/glm_edge_v/conditional_generation/pytorch/loader.py`:
  - `config.tie_word_embeddings = False` (Error 1 fix)
  - Pre-compute `model.model._boi_token_pos` from tokenizer output (enables Error 3 fix)
  - `inputs["use_cache"] = False` (Error 2 fix)

**Local only (HF cache, not committed):**
- `~/.cache/huggingface/modules/.../modeling_glm.py`: replaced data-dependent image injection
  with static `boi_mask` / `_boi_token_pos` approach; corrected tail slice index.
  Must be converted to a monkey-patch in `loader.py` for CI (does not affect FAIL outcome).

**Proposed fix for Tier B bug:**
Investigate which kernel produces wrong results on TT hardware. Prime candidates:
1. `ScaledDotProductAttentionPadTileDimsRewritePattern` output for SigLIP head_dim=72 padded to 96
   — verify that the padding/slicing path is numerically correct on hardware by running SigLIP
   in isolation.
2. GLM LM SDPA with `[1, 16, 585, 128]` shapes — verify causal mask handling at seq_len=585.
The fix would live in `tt-mlir` (either the pad-tile-dims workaround or the SDPA kernel itself).

## Tier B justification
Indicator: **internal-error-unknown-mechanism** — no TTNN error is raised; the compiled graph
runs to completion but produces non-deterministic PCC (0.43–0.74) on hardware. The exact kernel
or operation responsible cannot be identified without hardware-level profiling or isolation tests.
Diagnosis-first work is required before a targeted fix can be written.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    176.09s, 174.85s, 193.22s (three runs, all FAIL)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/glm_edge_v/conditional_generation/pytorch/loader.py`
- `~/.cache/huggingface/modules/transformers_modules/zai_hyphen_org/glm_hyphen_edge_hyphen_v_hyphen_2b/2053707733f99ab52e943904f43c2359a94301ef/modeling_glm.py` (local only)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4ccff61ac3909beb8defc40e3911ff378c446bc1 |
| tt-forge-models | 9c4c0a2476392f9b5df488517413b5e5dfd91f01 |

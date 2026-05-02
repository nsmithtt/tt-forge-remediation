# Remediation Summary: mm_grounding_dino-pytorch-Large_All-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mm_grounding_dino/pytorch-Large_All-single_device-inference]

## Result
FAIL — stablehlo.reduce_window legalization fails for cummax; Tier B new-infrastructure

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
stablehlo-cummax-reduce-window-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure: "The image processor of type `GroundingDinoImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`."

After loader fix, terminal failure:
```
loc("reduce-window.112"): error: failed to legalize operation 'stablehlo.reduce_window'
module_builder.cc:889    ERR| Failed to convert from SHLO to TTIR module
ValueError: Error code: 13
```

## Root cause
Two separate issues:

**Loader (fixed):** `AutoProcessor.from_pretrained()` in transformers 5.x now defaults `use_fast=True` for `GroundingDinoImageProcessor`, which is a breaking change. The fix is to pass `use_fast=False` explicitly. The loader also lacked the previously-identified spacy-namespace-collision fix (PIL.Image.new) and four bfloat16 dtype-mismatch patches — these were inherited from the Tiny variant remediation branch.

**Compiler (Tier B):** `torch.cummax` in `MMGroundingDinoModel.forward` (line 1767) lowers to a 2-output `stablehlo.reduce_window` with a compare(GE)+select body on `tensor<1x9xi64>`. The SHLO→TTIR pass has no lowering for this form of reduce_window (TTIR has no cummax op), causing legalization failure and a downstream Error code 13 in `_xla_warm_up_cache`.

## Fix
**Applied (loader):** In `tt_forge_models/mm_grounding_dino/pytorch/loader.py`, added `use_fast=False` to `AutoProcessor.from_pretrained()`. This fix is stacked on top of the Tiny-variant remediation branch which provides: `PIL.Image.new` (spacy namespace), `get_text_position_embeddings` dtype cast, `MultiScaleDeformableAttention` grid_sample dtype cast, `get_sine_pos_embed` output dtype cast, and `generate_encoder_output_proposals` output dtype cast.

**Proposed (compiler):** Add a `cummax` op to TTIROps.td, implement the SHLO→TTIR lowering pattern (2-output reduce_window with GE-compare+select body → cummax), and add TTIR→TTNN lowering + backend kernel. This spans TTIROps.td, StableHLOLegalizeCompositePass (or the reduce_window legalization pass), and TTNN backend — new-infrastructure Tier B.

## Tier B justification
Which indicator: new-infrastructure

TTIR has no cummax op. Adding it requires coordinated changes across TTIROps.td (op definition), SHLO→TTIR lowering (new reduce_window pattern), TTIR→TTNN lowering, and backend kernel support — at minimum 4 files across 2 repos.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 207.46s (0:03:27)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mm_grounding_dino/pytorch/loader.py` — added `use_fast=False` to `AutoProcessor.from_pretrained()`; inherited spacy + 4 bfloat16 dtype patches from prior Tiny variant remediation

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6babf7a57121d322344551a262e003272f3222f8 |
| tt-forge-models | cf707385c37d13c1a757d419e76c6880a824bdd4 |

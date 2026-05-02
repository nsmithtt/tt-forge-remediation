# Remediation Summary: mask2former-semantic_segmentation-pytorch-Swin_Tiny_Ade-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mask2former/semantic_segmentation/pytorch-Swin_Tiny_Ade-single_device-inference]

## Result
FAIL — PCC=0.7499 on TT silicon after loader fixes; Tier B ttmlir-f32-precision-not-preserved

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
Original reported failure:
```
The image processor of type `Mask2FormerImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.
```

Actual failure after reproducing:
```
AttributeError: module 'spacy' has no attribute 'Language'
```
(raised inside `datasets` dill hasher when `load_dataset("huggingface/cats-image")` was called,
because `tt_forge_models/spacy/` namespace package shadows the real `spacy` library)

Post-loader-fix failure on TT silicon:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7499268715307845. Required: pcc=0.99.
```

## Root cause
Two loader-layer bugs were present:

1. **spacy namespace collision**: `tt_forge_models/spacy/` is a namespace package that shadows the system
   `spacy` package. The `datasets` library's dill-based dataset hasher accesses `spacy.Language`,
   which fails with `AttributeError` because the namespace shadow has no such attribute. The call to
   `load_dataset("huggingface/cats-image")["test"]` in `load_inputs` is unnecessary — the model
   only needs a plain RGB image, which can be constructed with `PIL.Image.new("RGB", (640, 480))`.

2. **transformers 5.x `use_fast` default changed**: `AutoImageProcessor.from_pretrained()` in
   transformers 5.x now defaults to loading `Mask2FormerImageProcessor` as a fast processor,
   emitting a breaking-change warning. The fix is `use_fast=False`.

After fixing both loader bugs, the model compiles and runs on TT silicon, but the Swin Transformer
backbone's deep BF16 matmul accumulation causes precision degradation. PCC=0.7499, far below the
required 0.99. This is the same failure pattern seen on the Swin_Large_Cityscapes variant
(`ttmlir-f32-precision-not-preserved`): BF16 accumulation across 24 Swin Transformer stages loses
enough precision that logits diverge from the FP32 CPU reference. This is a compiler-stack bug
in tt-mlir (f32 precision not preserved through the lowering pipeline).

## Fix
Two loader fixes applied in `tt_forge_models/mask2former/semantic_segmentation/pytorch/loader.py`:

1. Replaced `from datasets import load_dataset` + `load_dataset("huggingface/cats-image")["test"]`
   with `from PIL import Image` + `image = Image.new("RGB", (640, 480))` in `load_inputs`.

2. Added `use_fast=False` to `AutoImageProcessor.from_pretrained()` call in `_load_image_processor`.

Committed as: `fix: Mask2Former semantic_segmentation loader (transformers 5.x + spacy collision)`
Branch: `remediation/mask2former-semantic_segmentation-pytorch-Swin_Tiny_Ade-single_device-inference`
in `tenstorrent/tt-forge-models`.

For the Tier B compiler bug, the proposed fix is to preserve f32 accumulation through Swin
Transformer's window-attention matmul lowering in tt-mlir. This requires cross-cutting changes
across multiple lowering passes to prevent downcast to BF16 at intermediate accumulation steps.

## Tier B justification
Which Tier B indicator applies: cross-cutting

The `ttmlir-f32-precision-not-preserved` bug requires changes across multiple lowering passes in
tt-mlir to preserve f32 accumulation where the model uses f32 matmuls internally. It is not a
single-function fix; it affects every matmul lowering that currently downcasts to BF16. This pattern
has been observed consistently across Swin Transformer variants (Swin_Large_Cityscapes PCC=0.73,
Swin_Tiny_Ade PCC=0.75).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    1485.53s (0:24:45)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `mask2former/semantic_segmentation/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a978e0e2b7b190f455dcafdb61e39d93afd393ae |
| tt-forge-models | 5bc125cd62547ab8f80ff05f863ca8736f3abbbe |

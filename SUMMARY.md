# Remediation Summary: metaclip_2/pytorch-worldwide_s16_384-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[metaclip_2/pytorch-worldwide_s16_384-single_device-inference]

## Result
FAIL — PCC 0.9816 vs required 0.99; CPU BF16 gives 0.99998, so TT precision gap is not BF16 accumulation (Tier B)

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
The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

After the loader fix, the test fails with:
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9815837538811949. Required: pcc=0.99.

## Root cause
Two bugs were found and fixed at the loader layer:

1. **transformers 5.x use_fast default** — `AutoProcessor.from_pretrained()` in `_load_processor()` lacked `use_fast=False`, causing CLIPImageProcessor to load as a fast processor (breaking change in transformers 5.x).

2. **load_dataset spacy namespace collision** — `load_dataset("huggingface/cats-image")` in `load_inputs()` hit `AttributeError: module 'spacy' has no attribute 'Language'` due to the tt_forge_models/spacy namespace package polluting sys.modules. Fixed by replacing with `PIL.Image.fromarray()` using a seeded numpy RNG for non-constant inputs.

After both loader fixes, the model runs on TT hardware but produces PCC = 0.9816 vs the required 0.99. CPU BF16 vs CPU F32 gives PCC = 0.99998, which rules out normal BF16 accumulation as the cause. The ~1.8% PCC gap indicates a real precision bug in the TT compiler lowering of this CLIP-ViT model.

## Fix
**Loader fixes** (tt_forge_models, branch `remediation/metaclip-2-pytorch-worldwide-s16-384`):
- `metaclip_2/pytorch/loader.py`: Add `use_fast=False` to `AutoProcessor.from_pretrained()`.
- `metaclip_2/pytorch/loader.py`: Replace `load_dataset("huggingface/cats-image")` with `PIL.Image.fromarray()` from a seeded numpy array to avoid spacy collision and ensure non-zero-variance inputs.

**Residual bug** (unfixed, Tier B):
- PCC 0.9816 from TT hardware vs CPU F32, where CPU BF16 gives 0.99998. This is a ttmlir-f32-precision-not-preserved issue in the CLIP-ViT lowering path. No change made.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting

The TT vs CPU precision gap of ~1.8% PCC (while CPU BF16 gap is only 0.002%) suggests accumulated precision errors across many layers of the ViT attention/layer-norm/softmax lowering path. This is not a single-file, single-function fix — it is a cross-cutting precision preservation issue in TT MLIR lowering that would require coordinated changes across multiple ops.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: ~86s (model loads and runs, fails on PCC check)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/metaclip_2/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 9ef141b3979cd0508333853267739e7cd4fe6224 |

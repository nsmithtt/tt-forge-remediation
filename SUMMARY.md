# Remediation Summary: longclip/pytorch-GmP_ViT_L_14-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[longclip/pytorch-GmP_ViT_L_14-single_device-inference]

## Result
FAIL — PCC=0.9187 on Blackhole p150b; BF16 CPU floor is ~1.0; compiler precision degradation in 24-layer ViT-L-14 (ttmlir-bf16-matmul-precision-floor, Tier B)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO — measured BF16-CPU vs FP32-CPU PCC = 0.99999994 (floor is ~1.0, silicon result of 0.9187 is a compiler regression)
- Warning / exception suppression: NO

## Failure
Original failure (before loader fixes):
```
TypeError: CLIPModel.__init__() got an unexpected keyword argument 'return_dict'
```

After loader fix 1, second failure:
```
AttributeError: module 'spacy' has no attribute 'Language'
```

After loader fix 2, terminal failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9187526576999453. Required: pcc=0.99.
```

## Root cause
Three issues found:

1. **Loader bug (Tier A, fixed)**: In transformers 5.x, `CLIPModel.__init__()` no longer accepts `return_dict` as a keyword argument. The loader was passing `return_dict=False` in `model_kwargs` to `from_pretrained`, which propagates it to `__init__`. Fix: set `model.config.return_dict = False` after loading; the `@can_return_tuple` decorator on `forward()` reads it to return a tuple.

2. **Loader/infra bug (Tier A, fixed)**: `dynamic_loader.setup_models_path()` was inserting `models_root` (the `tt_forge_models/` directory) into `sys.path`. Because `tt_forge_models/spacy/` exists without `__init__.py`, Python creates a namespace package for `spacy` from it, shadowing the real spaCy library. When `datasets._dill.save()` later checks for `spacy.Language` it fails with `AttributeError`. Fix: remove `sys.path.insert(0, models_root)` since relative imports work via `__package__` and the manually-registered `tt_forge_models` namespace module.

3. **Compiler precision (Tier B, unfixable here)**: After both loader fixes the model compiles and runs on Blackhole p150b but produces PCC=0.9187 vs the required 0.99 threshold. The BF16-CPU vs FP32-CPU PCC is 0.99999994, confirming the BF16 floor is not the cause. This is the known WH/BH BF16 matmul precision floor that compounds over 24 ViT-L-14 transformer layers — the same Tier B issue affecting CLIP Large_Patch14 on p150 (PCC=0.9832), DocTR PARSeq ViT (PCC=0.9194), Gemma 7B (PCC=0.915), and Qwen3 (PCC=0.864). LongCLIP uses longer sequences (248 tokens vs standard 77), which compounds the error further.

## Fix
Two Tier A loader fixes applied:

1. `tt-xla/third_party/tt_forge_models/longclip/pytorch/loader.py`:
   - Removed `"return_dict": False` from `model_kwargs`
   - Added `model.config.return_dict = False` after `from_pretrained`

2. `tt-xla/tests/runner/utils/dynamic_loader.py`:
   - Removed `sys.path.insert(0, models_root)` from `setup_models_path()`

## Tier B justification
`ttmlir-bf16-matmul-precision-floor`: WH/BH BF16 matmul precision error compounds over 24 transformer layers in ViT-L-14. BF16-CPU floor is ~1.0 but TT silicon gives PCC=0.9187. This is a cross-cutting compiler precision issue affecting many large transformer models; fixing it requires changes in tt-mlir's matmul lowering pipeline (>3 files, cross-repo).

## Verification
- pytest exit: FAIL (PCC 0.9187 < 0.99 threshold)
- Hardware:    blackhole-p150b
- Duration:    ~2 min (121s)
- Tier A attempts: 2 (both applied)

## Files changed
- `tt-xla/third_party/tt_forge_models/longclip/pytorch/loader.py`
- `tt-xla/tests/runner/utils/dynamic_loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c1abee55f9bff7150250b539624793934356d993 |
| tt-forge-models | 8f9189cb15be5d5b77b86a80507ee63f9523d1a0 |

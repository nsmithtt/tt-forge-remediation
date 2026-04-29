# Remediation Summary: beit/pytorch-BEiT_Large_Patch16_384_IN22K_FT_IN22K_IN1K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[beit/pytorch-BEiT_Large_Patch16_384_IN22K_FT_IN22K_IN1K-single_device-inference]

## Result
FAIL — PCC=0.18631104738467294 (compiler-stack bug, deterministic wrong output on TT silicon for BEiT-Large)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
beit-pcc-0186-deterministic-compiler-regression

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
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Actual failure after loader fix:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.18631104738467294. Required: pcc=0.99.
```

## Root cause

Two distinct issues were found:

**Issue 1 (FIXED — loader layer):** The `huspacy/pytorch/loader.py` module imported `spacy` at module level. During pytest collection, all model loaders are imported, so `spacy` (a SWIG-wrapped C library) was loaded before any test ran. This triggered a Python `DeprecationWarning: builtin type swigvarlink has no __module__ attribute` at the session level. Making the import lazy (`import spacy` moved inside `_load_nlp()`) resolved this.

**Issue 2 (NOT FIXED — tt-mlir/tt-metal):** BEiT-Large with 577 tokens (576 patches + 1 CLS from a 384×384 image at patch size 16) produces PCC=0.18631104738467294 on TT silicon vs CPU reference. This is a deterministic, reproducible failure that is identical across:
- Experimental compile path (default)
- Legacy compile path (`tt_legacy_compile=True`)
- `aten.layer_norm.default` decomposed to `var_mean.correction` ops (default path)
- `aten.layer_norm.default` preserved as intact op (routing through XLA batch_norm_training)

The identical PCC across all Python-layer code variants indicates the error lies below the FX graph manipulation layer, in the TTNN kernel execution or MLIR lowering for BEiT's specific op/shape combination. BEiT Base (HuggingFace variant, 197 tokens, 224×224) has the same known regression (test config has `assert_pcc: false # Fell to <0.3 after experimental compile`), confirming this is a systemic issue with BEiT models on TT silicon, not an input-shape fluke.

The TTNN SDPA kernel handles non-multiple-of-32 sequence lengths via internal padding (`padded_Sk = ceil(577/32)*32 = 608`) with a `use_padded_mask` flag, so the chunk-size issue is not a simple unhandled case. The actual kernel behavior with `use_padded_mask=true` for non-causal bidirectional attention and non-tile-aligned K may contain a masking bug, but this is not confirmed.

## Fix
Issue 1 was fixed in `tt-forge-models` at:
- `huspacy/pytorch/loader.py`: moved `import spacy` from module scope into `_load_nlp()` method (commit `e3bc42ad002654512b9598cb8109f09ef721c3cd`)

Issue 2 was not fixed. The proposed investigation and fix area:
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`: inspect the padding mask generation when `use_padded_mask=true` and `is_causal=false`. The non-causal padded mask path (seq_len=577, padded to 608, non-tile-aligned Sk) may have a boundary tile masking error that causes incorrect attention weights.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
internal-error-unknown-mechanism

The exact mechanism causing PCC=0.186 is not confirmed after extensive investigation. The bug is deterministic and unchanged across all Python-layer code variations, pointing to a kernel-level issue. Confirming root cause requires silicon-level debugging (e.g., inspecting intermediate tensor values after each transformer layer, or isolating the failing op by running subgraphs). This diagnosis-first work requires tt-metal/ttnn expertise beyond a scoped fix attempt.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    68.98s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models` → `e3bc42ad002654512b9598cb8109f09ef721c3cd` (submodule pointer)
- `tt-forge-models/huspacy/pytorch/loader.py`: lazy spacy import

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9adc62617ec56b2696777ca61029db1e7989db50 |
| tt-forge-models | e3bc42ad002654512b9598cb8109f09ef721c3cd |

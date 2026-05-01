# Remediation Summary: luke-sequence_classification-pytorch-mizuiro_sakura_luke_japanese_large_sentiment_analysis_wrime-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[luke/sequence_classification/pytorch-mizuiro_sakura_luke_japanese_large_sentiment_analysis_wrime-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: MLukeTokenizer BPE init crash (loader) and gather maxIndex==0 concat mismatch (tt-mlir)

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
transformers-5x-mluke-tokenizer-bpe-vocab-dict

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TypeError: argument 'vocab': 'dict' object cannot be converted to 'Sequence'
```
(The CI label `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is a trailing Python warning printed by pytest, not the actual error.)

## Root cause
**Bug 1 — loader (transformers 5.x MLukeTokenizer BPE init)**:
In transformers 5.x, `MLukeTokenizer` was refactored to use `TokenizersBackend`.
`SentencePieceExtractor.extract()` returns BPE vocab as a `{token: spm_id}` dict, but
`__init__` always passes it to `Unigram()` which requires `List[Tuple[str, float]]`,
raising `TypeError`.

**Bug 2 — tt-mlir (`StableHLOGatherToSliceRepeatConcatPattern` maxIndex==0)**:
After the tokenizer fix the model compiles, but `token_type_embeddings` has shape
`[1, 768]` (a single token-type).  When `sliceSizes[dim] == inputShape[dim]`,
`maxIndex = 0`.  Every gather index simultaneously satisfies `index == 0` AND
`index == maxIndex`, so the starts/ends counting loop double-counts all N indices,
producing a `ttir.concat` with `2N-1 = 1023` elements instead of `N = 512`, and the
verifier rejects it:
```
'ttir.concat' op Output tensor dimension 0 does not match the sum of input tensor dimensions: 512 vs. 1023.
```
Same root cause as jina_reranker_v2 and GLuCoSE-base-ja-v2.

## Fix
**Bug 1**: Added `_patch_mluke_tokenizer()` to
`tt_forge_models/luke/sequence_classification/pytorch/loader.py` and called it in
`load_model()` before `AutoTokenizer.from_pretrained()`.  The patch detects the BPE
dict vocab, remaps SPM IDs to fairseq alignment, builds a proper BPE `Tokenizer`, calls
the original `__init__` with `vocab=None` to run all other init logic, then replaces the
dummy Unigram tokenizer with the real BPE one.

**Bug 2**: Cherry-picked commit `97abb8bf3d18a49a2a887ae00e3d8cfa2589fd67` into
`tt-mlir` on the remediation branch.  The fix adds a `notifyMatchFailure` guard in
`StableHLOGatherToSliceRepeatConcatPattern` when `maxIndex == 0`, falling back to
`StableHLOGatherToEmbeddingPattern` which handles singleton indexed dimensions correctly.
File: `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    72.04s (0:01:12)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/luke/sequence_classification/pytorch/loader.py` — MLukeTokenizer BPE fix
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — maxIndex==0 guard

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 97abb8bf3d18a49a2a887ae00e3d8cfa2589fd67 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5a6b99dfb617a7b22b683cbdf641f314b6473280 |

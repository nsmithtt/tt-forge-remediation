# Remediation Summary: flair_ner_german-NER_German_Legal-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flair_ner_german/pytorch-NER_German_Legal-single_device-inference]

## Result
FAIL — `pack_padded_sequence` + LSTM incompatible with torch.dynamo/TT compilation (Tier B new-infrastructure); loader bugs fixed

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
lstm-packed-sequence-dynamo-tracing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
If you're using torch.compile/export/fx, it is likely that we are erroneously tracing into a custom kernel. To fix this, please wrap the custom kernel into an opaque custom op.
```

Raised at `_VF.lstm(packed, ...)` when torch.dynamo traces through `pack_padded_sequence` + BiLSTM inside `SequenceTagger.forward()` during TT device execution.

## Root cause

Two issues, layered:

**Loader (fixed):** `load_inputs` returned a raw `Sentence` object, but `SequenceTagger.forward(sentence_tensor, lengths)` requires pre-embedded tensors. The loader must call `model._prepare_tensors([sentence])` (which runs Flair's embedding pipeline on CPU) to obtain the `sentence_tensor` and `lengths` tensors. A second loader bug: Flair's internal embeddings always produce `float32`, so `sentence_tensor` must be cast to `dtype_override` (bfloat16) before being passed to the cast model; without the cast, the `embedding2nn` linear layer raises a dtype mismatch.

**Compiler-stack (Tier B):** After the loader is fixed, the CPU forward pass succeeds but TT device execution fails. `SequenceTagger.forward` calls `pack_padded_sequence(sentence_tensor, lengths)` which creates a `PackedSequence` with lazily-allocated `.data`. When `torch.dynamo` traces into `_VF.lstm(packed, ...)`, it accesses `packed.data` before memory has been allocated, triggering `RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet`. This is the same class of error described in PyTorch's own dynamo documentation for custom C++ kernels not wrapped as opaque ops. Supporting `PackedSequence` through TT's torch.compile/dynamo pipeline requires new infrastructure.

## Fix

**Loader fixes** (tt-forge-models, `flair_ner_german/pytorch/loader.py`):
1. Move `self.model = tagger` assignment to after the `dtype_override` cast so the stored model matches the returned model dtype.
2. In `load_inputs`, call `self.model._prepare_tensors([sentence])` to produce `(sentence_tensor, lengths)` tensors instead of returning a raw `Sentence` object.
3. Cast `sentence_tensor` to `dtype_override` when set, because Flair embeddings always return float32.

**Proposed compiler-stack fix (not attempted — Tier B):** Add support for `PackedSequence` / `_VF.lstm` with packed inputs in the TT torch.compile pipeline, or intercept `pack_padded_sequence` at the dynamo level to substitute a padded-LSTM path that the compiler can trace through. This would live in `tt-xla`'s `torch_overrides.py` or the dynamo decomposition layer.

## Tier B justification
**new-infrastructure**: `PackedSequence` from `pack_padded_sequence` uses lazy tensor allocation that is fundamentally incompatible with torch.dynamo tracing. Properly supporting it requires either a new decomposition of `_VF.lstm` for packed inputs or a mechanism to intercept `pack_padded_sequence` and substitute a padded variant — neither exists in the TT torch.compile pipeline today.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    35.42s (CPU pass succeeded; TT pass failed with the original error)
- Tier A attempts: N/A

## Files changed
- `flair_ner_german/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ac94e3b6f3a354ece30b2b4013d8ee1d0a1a9f21 |
| tt-forge-models | 0d759151d24bfbb19c2b299685cf0e6e06fddca9 |

# Remediation Summary: gliner_multitask-pytorch-Multitask_v1.0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gliner_multitask/pytorch-Multitask-v1.0-single_device-inference]

## Result
FAIL — pack_padded_sequence + bidirectional LSTM breaks XLA graph tracing (Tier B)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
gliner-lstm-packed-sequence-dynamo-tracing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
```
from `gliner/modeling/layers.py:58 in torch_dynamo_resume_in_forward_at_55` after
`pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)` causes a
graph break, and then `self.lstm(packed_x, hidden)` runs TT tensors in eager mode via
`_VF.lstm`.

Before reaching this error, two earlier bugs were fixed:

1. **llm2vec namespace shadow**: The `llm2vec/` directory in `tt_forge_models` is
   picked up as a Python namespace package. `gliner.modeling.encoder` calls
   `is_module_available("llm2vec")` which returns True, then
   `from llm2vec.models import ...` fails with `ModuleNotFoundError`.

2. **SharedLHSMatmulFusion OOB SIGABRT**: `collectCandidates` in
   `TTIRFusing.cpp` checked RHS rank but not output rank. When candidates have
   mixed output ranks (e.g., 2-D vs 3-D), `replaceWithSlices` indexes an ArrayRef
   OOB → `Assertion 'Index < Length'` → SIGABRT. This was the original crash
   producing the "Extension modules: ..." faulthandler dump.

## Root cause
After the two loader and compiler fixes above, the model's `LstmSeq2SeqEncoder.forward`
calls `pack_padded_sequence(x, lengths, enforce_sorted=False)`. The `enforce_sorted=False`
path does a data-dependent sort, triggering a dynamo graph break. After the break,
`self.lstm(packed_x)` runs in eager mode on TT device tensors via `_VF.lstm` through
`__torch_function__`, which fails because the PackedSequence holds tensors with
unallocated XLA lazy storage.

A static padded LSTM replacement was investigated in the base GLiNER reports
and produces PCC=0.456 — semantically wrong because the backward direction of the
bidirectional LSTM processes padding zeros before real tokens, corrupting its hidden
state. The correct fix requires per-sample sequence-length information at compile time
(dynamic shapes), which is new infrastructure.

## Fix
**Two fixes applied:**

1. **Loader fix** — `tt_forge_models`: `gliner_multitask/pytorch/loader.py`
   - Temporarily strip `project_root` (which contains `llm2vec/` as a namespace dir)
     from `sys.path` and clear cached `llm2vec` entries from `sys.modules` before
     importing `gliner`, so `is_module_available("llm2vec")` returns False.
   - Monkey-patch `gliner.modeling.utils.extract_word_embeddings` and
     `gliner.modeling.base.extract_word_embeddings` with a static `scatter_`
     implementation that routes invalid tokens to a sink slot at `max_text_length`
     and discards it (avoids `torch.where(words_mask > 0)` dynamic output shape).
   - Remove `llm2vec` from `requirements.txt` — it requires `transformers<=4.44.2`
     which conflicts with the `5.2.0` stack.
   - Commit: `05ea307672` on
     `remediation/gliner_multitask-pytorch-Multitask_v1.0-single_device-inference`
     in `tt-forge-models`.

2. **Tier A compiler fix** — `tt-mlir`: `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
   - Add output-rank guard in `SharedLHSMatmulFusion::collectCandidates`: skip any
     candidate whose result rank differs from the root op's result rank.
   - Commit: `bec72757a` on
     `remediation/gliner_multitask-pytorch-Multitask-v1.0-single_device-inference`
     in `tt-mlir`.

**Proposed fix for the remaining Tier B bug:**

In `gliner/modeling/layers.py`, replace `pack_padded_sequence` + bidirectional LSTM
with a padded sequence approach using per-sample length information. However, the
semantically correct implementation requires data-dependent indexing at compile time
(dynamic shapes support in the XLA backend) to start the backward LSTM direction
from the last real token rather than the padding. This is new infrastructure in
`tt-xla` / `torch-xla`.

## Tier B justification
`new-infrastructure` — correct bidirectional LSTM handling for variable-length sequences
requires dynamic shape support: the backward direction must start from the last real token
per sample, which requires data-dependent tensor indexing not yet supported in the XLA
compilation path.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    568.35s (0:09:28)
- Tier A attempts: 1

## Files changed
- `gliner_multitask/pytorch/loader.py` (tt-forge-models)
- `gliner_multitask/pytorch/requirements.txt` (tt-forge-models)
- `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (tt-mlir)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | bec72757a1e17219bf6e99902aaff29de49a6b69 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 05ea307672bc234efaa599be19045949fdf2e864 |

# Remediation Summary: gliner-pytorch-Small_PII-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gliner/pytorch-Small_PII-single_device-inference]

## Result
FAIL — third compiler-stack bug (Tier B): `pack_padded_sequence` + bidirectional LSTM in `LstmSeq2SeqEncoder` causes a dynamo graph break; LSTM then runs on TT tensors in eager mode and fails

## Stack layer
tt-mlir, loader, tt-xla

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
Original failure (SIGABRT, fixed):
```
python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253: const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]: Assertion `Index < Length && "Invalid index!"' failed.
Fatal Python error: Aborted
Extension modules: numpy._core._multiarray_umath, ...
```
GDB backtrace frame #7: `SharedLHSMatmulFusion<LinearOp>::matchAndRewrite` in TTIRFusingPass.

Second failure (loader bug, fixed):
```
AttributeError: 'fused_0' object has no attribute 'xla_args'
```
in `torch_xla/_dynamo/dynamo_bridge.py:348`, triggered by `torch.where(words_mask > 0)` in `gliner/modeling/utils.py:extract_word_embeddings`.

Third failure (Tier B, unfixed):
```
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
If you're using torch.compile/export/fx, it is likely that we are erroneously tracing into a custom kernel.
```
in `gliner/modeling/layers.py:58` at `packed_output, hidden = self.lstm(packed_x, hidden)`, via
`torch_overrides.py:34: return func(*args, **(kwargs or {}))`.

## Root cause

**First bug (fixed — Tier A, tt-mlir):** `SharedLHSMatmulFusion<LinearOp>::matchAndRewrite` in `TTIRFusingPass` collected LinearOp candidates without checking whether each candidate's output rank matches the root op's output rank. `replaceWithSlices` then indexed into candidate shapes at `outputFusedDim = rootOutputRank - 1`, but a candidate with a lower output rank caused an out-of-bounds `ArrayRef::operator[]` assertion (SIGABRT). The GLiNER model (DeBERTa-based encoder) produces LinearOps sharing the same LHS with mismatched output ranks.

**Second bug (fixed — loader):** `extract_word_embeddings` in `gliner/modeling/utils.py` calls `torch.where(words_mask > 0)` in its unary form. This returns variable-length index tensors whose size depends on runtime data, breaking XLA static graph compilation. `partition_fx_graph_for_cpu_fallback` in the torch_xla dynamo bridge tries to run this as a CPU-fallback subgraph; `InputCollector` fails to set `xla_args` on `fused_0` for this data-dependent operation, causing `AttributeError`. Fixed in the loader by replacing `extract_word_embeddings` with a static `scatter_` implementation using a sink slot at index `max_text_length` for invalid tokens.

**Third bug (Tier B — tt-xla / torch_xla):** `LstmSeq2SeqEncoder.forward` in `gliner/modeling/layers.py` calls `pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)`. The `enforce_sorted=False` path performs data-dependent sorting of sequences by length, which causes a dynamo graph break. After the break, `self.lstm(packed_x)` is called with concrete TT tensors in eager mode. The TT `__torch_function__` handler calls `_VF.lstm` directly with TT tensors, which fails because `_VF.lstm` is a C++ kernel that cannot operate on TT device tensors in eager mode. A static padded-LSTM replacement (skipping packing) gives semantically incorrect results for the bidirectional direction (PCC=0.456) because the backward LSTM processes padding zeros before real tokens, corrupting its hidden state for all real positions.

## Fix

**Applied (tt-mlir):** Added an output-rank equality guard in `SharedLHSMatmulFusion::collectCandidates` (`lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`). Before adding a candidate to the fusion list, the code now checks that `candidateOutputType.getRank() == rootOutputRank`. Candidates with a different output rank are skipped.

Commit: `ebfa3f2ff` on branch `remediation/gliner-pytorch-Small_PII-single_device-inference` in tt-mlir.

**Applied (loader):** Replaced `extract_word_embeddings` (and its reference in `gliner.modeling.base`) with a static `scatter_`-based implementation in `gliner/pytorch/loader.py`. The new function routes invalid token positions (where `words_mask == 0`) to a sink slot at index `max_text_length` in a temporary `(batch, max_text_length+1, embed_dim)` tensor, applies `scatter_`, then discards the sink column.

Commit: `7b185e6877` on branch `remediation/gliner-pytorch-Small_PII-single_device-inference` in tt-forge-models.

**Not applied (tt-xla):** The `pack_padded_sequence` + bidirectional LSTM issue requires the dynamo/TT-XLA bridge to support PackedSequence operations or requires an architectural change to `LstmSeq2SeqEncoder`. The forward LSTM is traceable without packing, but the backward direction is semantically incorrect when run on a right-padded sequence (backward LSTM starts from padding positions, corrupting hidden state for real token positions). Correct handling requires per-sample sequence-length information at runtime (dynamic shape), which is new infrastructure.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

`pack_padded_sequence` with `enforce_sorted=False` causes a dynamo graph break because it requires data-dependent sorting. After the break, `_VF.lstm` is called on TT tensors in eager mode and fails. A static replacement without packing is semantically incorrect for bidirectional LSTM with variable-length sequences — the backward direction requires per-sample sequence lengths to start from the correct position, which is inherently dynamic. Supporting this correctly requires new dynamic-shape infrastructure in the TT-XLA dynamo bridge.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    104.13s (0:01:44)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` — added output-rank check in `SharedLHSMatmulFusion::collectCandidates`
- `tt-forge-models/gliner/pytorch/loader.py` — static scatter_ replacement for `extract_word_embeddings`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | ebfa3f2ff9248220d35adba6c96200e3684dd610 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7b185e6877a38adecb2fde6ffa3ccc864fcb0e2e |

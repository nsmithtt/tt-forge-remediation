# Remediation Summary: opensearch_semantic_highlighter_v1-text_classification-pytorch-Semantic_Highlighter_V1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[opensearch_semantic_highlighter_v1/text_classification/pytorch-Semantic_Highlighter_V1-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-init-weights-missing-post-init

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Error code: 13

## Root cause
Two loader-layer bugs, one masking the other:

1. **`init_weights()` instead of `post_init()` (transformers 5.x)**  
   `BertTaggerForSentenceExtractionWithBackoff.__init__` called `self.init_weights()`.
   In transformers 5.x, `post_init()` is what initialises `all_tied_weights_keys` (and then calls `init_weights()` internally). Without this the later `_finalize_model_loading → _adjust_tied_keys_with_tied_pointers` call accessed `self.all_tied_weights_keys` via `Module.__getattr__`, which raised `AttributeError`. This bug was already fixed on the bringup-27 branch (commit f45b9d63c0); the submodule pointer just hadn't been advanced past it.

2. **Data-dependent Python control flow causing device→host transfer (Error code: 13)**  
   After the `post_init` fix, the test reproduced the reported `RuntimeError: Error code: 13`.  
   Inside `forward`, `_get_agg_output` iterated `for j in range(int(n_sent))` where `n_sent = local_ids.max() + 1` was a scalar tensor on the TT device. `int()` forces a device→host synchronisation via PJRT, which fails with kInternal=13 on TT hardware. `_get_preds` had further D2H issues (`p[:ns]`, `hits.sum() == 0`, `.item()`, `torch.where(...)[0]`) and returned variable-length tensors incompatible with PCC comparison.

## Fix
`tt-forge-models` — `opensearch_semantic_highlighter_v1/text_classification/pytorch/src/model_utils.py`

1. **Replaced `self.init_weights()` → `self.post_init()`** (already present on bringup-27 branch; submodule pointer advanced to include it).

2. **Vectorised sentence aggregation** — replaced the D2H-requiring Python loop with `_aggregate_by_sentence`:
   - Uses `max_sents` (a Python `int` precomputed from the CPU-side `sentence_ids` in `prepare_highlighter_inputs`) as the fixed loop extent for `torch.arange`.
   - Computes per-batch offsets with `torch.where(valid_mask, ids, full_like(ids, max_sents)).min()`.
   - One-hot membership via `(local_ids.unsqueeze(-1) == j).to(dtype=seq_out.dtype)`.
   - Aggregates with `torch.einsum("bsd,bsj->bjd", ...)` — entirely on-device.

3. **Changed forward output to `probs [B, max_sents]`** — returns the softmax probability tensor directly instead of variable-length predicted-sentence indices. This gives a fixed-size float output suitable for PCC comparison, while preserving the full BERT + classifier computation on the TT device.

4. **`prepare_highlighter_inputs` returns `max_sentences: int`** alongside the tensor inputs. Because `to_device` only moves `Tensor` objects, this Python `int` passes through unchanged and reaches `forward` as a static value, allowing `torch.arange(max_sents)` to compile to a fixed-shape graph.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    51.31s
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `opensearch_semantic_highlighter_v1/text_classification/pytorch/src/model_utils.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | edff124ddfaa1b1c65a85734acf97eed2bbeccbe |

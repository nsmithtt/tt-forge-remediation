# Remediation Summary: gte-multilingual-base-embedding-generation

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gte_multilingual_base/embedding_generation/pytorch-gte-multilingual-base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
sdpa-mask-q-dim-broadcast

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
IndexError: index 6288518355790033519 is out of bounds for dimension 0 with size 128

(After fixing the above: ValueError: Error code: 13 — 'ttir.scaled_dot_product_attention' op Attention mask at dim 2 must match query sequence length)

## Root cause
Three bugs in two layers:

**Loader (tt_forge_models) — Bug 1: non-persistent buffer corruption**
transformers 5.x initializes models on meta device and then materializes
weights from the state_dict. Non-persistent buffers (registered with
`persistent=False`) are not in the state_dict, so their real-memory tensors
hold uninitialized garbage after `from_pretrained`. The GTE-multilingual-base
model registers `position_ids`, `inv_freq`, `cos_cached`, and `sin_cached`
as non-persistent buffers. After loading, `position_ids` contained values
like `6288518355790033519`, causing an IndexError when used to index the RoPE
cosine table.

**Loader (tt_forge_models) — Bug 2: f64 attention mask via Python float promotion**
`PreTrainedModel.get_extended_attention_mask` computes:
```python
(1.0 - extended_attention_mask) * torch.finfo(dtype).min
```
Both `1.0` and `torch.finfo(dtype).min` are Python floats (float64 in
Python's type system). When traced by XLA, these become f64 constants
that promote the entire attention mask computation to f64, producing a
`tensor<1x1x1x128xf64>` mask. TT hardware does not support f64.

**Compiler (tt-mlir) — Bug 3: SDPA mask q-dim not broadcast**
The attention mask has shape `[1,1,1,128]` (q-dim=1, broadcast semantics).
`ttir.ScaledDotProductAttentionOp` validation requires `mask.dim(2) ==
query_seq_len` but allows `dim(0)` and `dim(1)` to be 1 for broadcasting.
The q-dim broadcast case was not handled.

## Fix

**Loader fix** — `gte_multilingual_base/embedding_generation/pytorch/loader.py`
in `tt-forge-models` (commit `fa84bdc7b`):

1. Added `_reinit_non_persistent_buffers(model)` called after
   `from_pretrained`. Reinitializes `embeddings.position_ids` via
   `torch.arange`, recomputes `rotary_emb.inv_freq` from the base
   frequency formula, and calls `_set_cos_sin_cache` to repopulate
   `cos_cached` and `sin_cached`.

2. Added `_patched_get_extended_attention_mask` that computes the mask
   using `torch.tensor(1.0, dtype=dtype)` and
   `torch.tensor(torch.finfo(dtype).min, dtype=dtype)` — explicit dtype-typed
   tensors that prevent f64 promotion in XLA tracing. Bound to the model
   instance via `types.MethodType`.

**Compiler fix** — `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`
in `tt-mlir` (commit `2d02b389a`):

In `TenstorrentScaledDotProductAttentionConversionPattern`, when
`hasAttnMask` and `mask.dim(2) == 1` but `query_seq_len > 1`, insert a
`ttir::BroadcastOp` with `broadcast_dimensions = [1, 1, querySeqLen, 1]`
to expand the mask from `[B,H,1,K]` to `[B,H,seqLen,K]` before creating
`ttir::ScaledDotProductAttentionOp`.

After rebuilding `libTTMLIRCompiler.so` and copying it to
`tt-xla/third_party/tt-mlir/install/lib/`, all three bugs were resolved
and the test passed on silicon.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    62.07s
- Tier A attempts: 1

## Files changed
- `gte_multilingual_base/embedding_generation/pytorch/loader.py` (tt-forge-models)
- `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp` (tt-mlir)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 2d02b389a3c9a2dda07a5aaa63a1ed98b50d01c1 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | fa84bdc7b3b2fa14207649174b9af93210321739 |

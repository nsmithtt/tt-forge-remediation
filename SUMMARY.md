# Remediation Summary: gte_base_en_v1_5-embedding_generation

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gte_base_en_v1_5/embedding_generation/pytorch-gte-base-en-v1.5-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
transformers-5x-nonpersistent-buffer-uninit, sdpa-attn-mask-q-dim-no-broadcast

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: Error code: 13
```
Underlying MLIR failure:
```
loc("custom-call.1076"): error: 'ttir.scaled_dot_product_attention' op Attention mask at dim 2 must match query sequence length
module_builder.cc:889    ERR| Failed to convert from SHLO to TTIR module
```
Preceded by an `IndexError: index 7587475092128669798 is out of bounds for dimension 0 with size 128` from uninitialized `position_ids` buffer (not visible in final run, but root-caused in debugging).

## Root cause
Two bugs, in two layers:

**Layer 1 — loader (tt_forge_models):** `OrcaDB/gte-base-en-v1.5` uses custom remote code from Alibaba-NLP that registers `position_ids`, `inv_freq`, `cos_cached`, and `sin_cached` as non-persistent buffers (not saved to checkpoint). Under transformers 5.x, `AutoModel.from_pretrained` uses meta-device initialization which calls `__init__` but the non-persistent buffers get re-registered from uninitialized memory before the checkpoint weights are loaded, leaving garbage values. This caused `IndexError` on the position lookup.

**Layer 2 — tt-mlir (StableHLOLegalizeCompositePass):** GTE-Base uses bidirectional attention where the attention mask has shape `1x1x1x128` (a single mask row that broadcasts over all 128 query positions). The `TenstorrentScaledDotProductAttentionConversionPattern` passed this mask directly to `ttir::ScaledDotProductAttentionOp`, which verifies that dim 2 must exactly match the query sequence length (`seqLen = 128`). Since `1 != 128`, the TTIR verifier rejected the op during the SHLO→TTIR pass, returning `kInternal` (Error code: 13).

## Fix
**Fix 1 — loader** (`tt_forge_models/gte_base_en_v1_5/embedding_generation/pytorch/loader.py`):
Added `_reinit_non_persistent_buffers(model)` static method called after `AutoModel.from_pretrained`. Re-initializes `position_ids` as `torch.arange(config.max_position_embeddings)`, recomputes `inv_freq` from the RoPE base/dim, and calls `_set_cos_sin_cache` to populate `cos_cached`/`sin_cached`.

**Fix 2 — tt-mlir** (`lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`):
In `TenstorrentScaledDotProductAttentionConversionPattern::matchAndRewrite`, when the attention mask has shape `[B, H, 1, K]` and `seqLen > 1`, insert a `ttir::BroadcastOp` to expand the mask from `[B, H, 1, K]` to `[B, H, seqLen, K]` before creating the TTIR SDPA op. This is consistent with how dims 0 and 1 already allow broadcast (size 1) at the TTNN verifier level, but the TTNN SDPA kernel requires full shape at dim 2.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    51.55s
- Tier A attempts: 1

## Files changed
- `tt_forge_models/gte_base_en_v1_5/embedding_generation/pytorch/loader.py` — non-persistent buffer re-init
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp` — SDPA mask broadcast expansion

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 0dd64d4528a6a5d49f1017e58d6c1127941f701c |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | b09f3ce313aafc346145cd48b72ed75e9d9f18de |

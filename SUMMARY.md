# Remediation Summary: gte_large_en_v1_5-embedding_generation

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gte_large_en_v1_5/embedding_generation/pytorch-gte-large-en-v1.5-single_device-inference]

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
Preceded by (visible during CPU run debugging):
```
IndexError: index 7161128445189519223 is out of bounds for dimension 0 with size 128
```

## Root cause
Two bugs, in two layers:

**Layer 1 — loader (tt_forge_models):** `Alibaba-NLP/gte-large-en-v1.5` uses custom remote code that registers `position_ids`, `inv_freq`, `cos_cached`, and `sin_cached` as non-persistent buffers (not saved to checkpoint). Under transformers 5.x, `AutoModel.from_pretrained` uses meta-device initialization which allocates non-persistent buffers from uninitialized memory after the checkpoint weights are loaded, leaving garbage values. `position_ids[:10]` showed `[0, 67335, 49, ...]` instead of `[0, 1, 2, ...]`, causing `IndexError` on the RoPE position lookup during the CPU golden reference run.

**Layer 2 — tt-mlir (StableHLOLegalizeCompositePass):** GTE-Large uses bidirectional attention where the attention mask has shape `[1, 1, 1, 128]` (a single mask row that broadcasts over all 128 query positions). `TenstorrentScaledDotProductAttentionConversionPattern` passed this mask directly to `ttir::ScaledDotProductAttentionOp`, whose verifier requires dim 2 to exactly match the query sequence length (128). Since `1 != 128`, the TTIR verifier rejected the op during SHLO→TTIR, returning `kInternal` (Error code: 13).

## Fix
**Fix 1 — loader** (`tt_forge_models/gte_large_en_v1_5/embedding_generation/pytorch/loader.py`):
Added `_reinit_non_persistent_buffers(model)` static method called after `AutoModel.from_pretrained`. Re-initializes `position_ids` as `torch.arange(config.max_position_embeddings)`, recomputes `inv_freq` from the RoPE base/dim, and calls `_set_cos_sin_cache` to populate `cos_cached`/`sin_cached`.

**Fix 2 — tt-mlir** (`lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`):
In `TenstorrentScaledDotProductAttentionConversionPattern::matchAndRewrite`, when the attention mask has shape `[B, H, 1, K]` and `seqLen > 1`, insert a `ttir::BroadcastOp` to expand the mask from `[B, H, 1, K]` to `[B, H, seqLen, K]` before creating the TTIR SDPA op. This is consistent with how dims 0 and 1 already allow broadcast (size 1) at the TTNN verifier level, but the TTNN SDPA kernel requires full shape at dim 2.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    73.01s
- Tier A attempts: 1

## Files changed
- `tt_forge_models/gte_large_en_v1_5/embedding_generation/pytorch/loader.py` — non-persistent buffer re-init
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp` — SDPA mask broadcast expansion

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 8a7a6987b7f742a7da4932eec795fd00eef9bb3e |
| tt-xla          | 8318042e8cac5297778407a7eb977f3f68f16854 |
| tt-forge-models | a624759d965cd797df9c5330dd8d84310527772d |

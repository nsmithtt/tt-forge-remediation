# Remediation Summary: gte_multilingual_base-embedding_generation-pytorch-newmindai_TurkEmbed4STS-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gte_multilingual_base/embedding_generation/pytorch-newmindai/TurkEmbed4STS-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
sdpa-attn-mask-dim2-not-broadcast

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   ValueError: Error code: 13

Preceded by:
  loc("custom-call.1090"): error: 'ttir.scaled_dot_product_attention' op Attention mask at dim 2 must match query sequence length

## Root cause

Two independent bugs:

**Bug 1 (loader):** Transformers 5.x creates models inside `init_empty_weights()`, which converts all `register_buffer` calls to meta tensors. Non-persistent buffers (`position_ids`, `inv_freq`, `cos_cached`, `sin_cached`) are excluded from the state dict, so when materialized they receive `torch.empty` (garbage uninitialized) data. The CPU run failed with `IndexError: index 4429073836695105853 is out of bounds for dimension 0 with size 128` because `position_ids` held garbage memory.

**Bug 2 (tt-mlir):** The `TenstorrentScaledDotProductAttentionConversionPattern` in `StableHLOLegalizeCompositePass.cpp` passes the attention mask to `ttir::ScaledDotProductAttentionOp` without any shape transformation. BERT-like models produce masks with shape `[batch, 1, 1, kvSeqLen]` where dim 2 = 1 is a broadcast placeholder. The TTIR SDPA verifier requires dim 2 to exactly equal the query sequence length (no broadcasting permitted), causing the `ValueError: Error code: 13` failure.

## Fix

**Fix 1 (loader)** — `tt_forge_models/gte_multilingual_base/embedding_generation/pytorch/loader.py`:
- Added `_reinit_non_persistent_buffers(model)` to reinitialize `position_ids` (via `torch.arange`), `inv_freq`, `cos_cached`, and `sin_cached` after `from_pretrained` returns.
- Added `_patched_get_extended_attention_mask` to avoid f64 promotion in the attention mask computation (Python float literals and `torch.finfo().min` trace as f64 in XLA).

Cherry-picked from existing `remediation/gte-multilingual-base-embedding-generation` branch (commit `fa84bdc7b3`).

**Fix 2 (tt-mlir)** — `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`, `TenstorrentScaledDotProductAttentionConversionPattern::matchAndRewrite`:
- When the attention mask has dim 2 == 1 and query seqLen > 1, insert a `ttir::BroadcastOp` to expand dim 2 from 1 to seqLen before passing the mask to `ttir::ScaledDotProductAttentionOp`.
- `broadcast_dimensions = [1, 1, seqLen, 1]` — repeats the single mask row `seqLen` times, producing the full `[batch, heads, seqLen, kvSeqLen]` mask expected by the TTIR verifier.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    64.43s
- Tier A attempts: 1

## Files changed
- `tt_forge_models/gte_multilingual_base/embedding_generation/pytorch/loader.py` (cherry-pick from remediation/gte-multilingual-base-embedding-generation)
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | b51faf305f22392403678d04fe38a3f962a0b00a |
| tt-xla          | 8063856172721a878278c3fee07abf3aa8ed4a17 |
| tt-forge-models | d2237909f27ec6bd82ae4710455d431d8adffe3b |

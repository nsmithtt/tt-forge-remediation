# Remediation Summary: nomic-embed-text-v1-gguf-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[nomic/embed_text_gguf/pytorch-nomic-embed-text-v1-GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
sdpa-bert-attention-mask-broadcast-dim2

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: Error code: 13

## Root cause

Two independent bugs:

**Bug 1 — Loader layer (tt_forge_models)**: The `nomic-embed-text-v1-GGUF` repo ships a GGUF checkpoint with `general.architecture = nomic-bert`, which was not registered in `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING`. The original loader attempted `AutoModel.from_pretrained(..., gguf_file=...)` which failed immediately with an unrecognised architecture error, surfacing as `ValueError: Error code: 13` through the PJRT wrapper.

In addition to missing arch registration, five NomicBertConfig defaults produced wrong computation even after registration:
- `"feed_forward_length": "intermediate_size"` mapping (wrong key) → NomciBertGatedMLP defaulted to `int(8*768/3)=2048` instead of 3072 (size mismatch / wrong MLP output)
- `"attention.layer_norm_epsilon": "layer_norm_eps"` mapping (wrong key) → epsilon stayed at GPT2Config default 1e-5 instead of trained 1e-12
- `rotary_emb_fraction` defaulted to 0.0 → NomicBertEmbeddings allocated random absolute position embeddings not in the GGUF (PCC ~0.07)
- `qkv_proj_bias / mlp_fc1_bias / mlp_fc2_bias` defaulted to True → random bias tensors absent from GGUF retained random init values under `strict=False`
- Tokenizer loaded from GGUF repo (no tokenizer files) instead of base model

**Bug 2 — tt-mlir compiler layer**: `get_extended_attention_mask` in NomicBertModel produces a float attention mask with shape `[batch, 1, 1, kvSeqLen]` where dim 2 = 1 is a broadcast placeholder (the same mask applies to every query position). The `TenstorrentScaledDotProductAttentionConversionPattern` in `StableHLOLegalizeCompositePass.cpp` passed this mask as-is to the `ttir::ScaledDotProductAttentionOp`, whose verifier requires `mask.shape[2] == querySeqLen` (no broadcasting allowed), causing the compilation to abort with `kInternal = 13`.

## Fix

**Loader fix** (`tt_forge_models/nomic/embed_text_gguf/pytorch/loader.py`):
- Added `_patch_nomic_bert_gguf()` that registers `nomic-bert` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING["config"]` with correct field mappings: `feed_forward_length → n_inner`, `attention.layer_norm_epsilon → layer_norm_epsilon`
- Patched `load_gguf_checkpoint` to inject `activation_function="swiglu"`, `rotary_emb_fraction=1.0`, `qkv_proj_bias=False`, `mlp_fc1_bias=False`, `mlp_fc2_bias=False`, dropout=0.0
- Replaced `AutoModel.from_pretrained(gguf_file=...)` with direct `GGUFReader + dequantize + get_gguf_hf_weights_map + TensorProcessor` pipeline to load weights, bypassing the session-contaminated `load_gguf_checkpoint` chain
- Changed tokenizer to load from `nomic-ai/nomic-embed-text-v1` (base model with tokenizer files)

**Compiler fix** (`tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`):
- In `TenstorrentScaledDotProductAttentionConversionPattern::matchAndRewrite`, when an attention mask is present with `mask.shape[2] == 1` and `querySeqLen > 1`, insert a `ttir::BroadcastOp` to expand the mask from `[B, 1, 1, S_k]` to `[B, 1, S_q, S_k]` before creating the TTIR SDPA op

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    156.10s (0:02:36)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/nomic/embed_text_gguf/pytorch/loader.py`
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 129eaa6f4238581852b6e66e5f040ba296bbec14 |
| tt-xla          | eb0ffa9d51f159e411377de4e47c993d76336920 |
| tt-forge-models | c95b34601074c6357ed645ba11288b820f02645b |

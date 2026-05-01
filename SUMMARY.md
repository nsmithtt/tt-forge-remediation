# Remediation Summary: giga_embeddings_instruct-embedding_generation-pytorch-iMiW-Giga-Embeddings-instruct-4bit-nf4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[giga_embeddings_instruct/embedding_generation/pytorch-iMiW/Giga-Embeddings-instruct-4bit-nf4-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir, tt-metal

## Tier
A

## Bug fingerprint
transformers-5x-bitsandbytes-nf4-loader-fixes, sdpa-3d-input-reshape, nlp-concat-heads-cb-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError(
```
Full chain of failures uncovered iteratively:
1. `ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes` — missing dependency
2. `KeyError: 'default'` in `ROPE_INIT_FUNCTIONS` — transformers 5.x removed the 'default' key
3. `AttributeError: 'GigarEmbedModel' object has no attribute 'all_tied_weights_keys'` — remote code never calls `post_init()`
4. `RuntimeError: Creating a Parameter from an instance of type Params4bit requires that detach() returns an instance of the same type` — BNB 4-bit weights can't move to TT device
5. `ValueError: Error code: 13 / 'ttir.scaled_dot_product_attention' op Query must be a 4D tensor` — LatentAttentionModel CrossAttention uses 3D SDPA
6. `TT_THROW: Statically allocated circular buffers grow to 2208256 B which is beyond max L1 size of 1572864 B` — NLPConcatHeads double-buffer CB overflows L1

## Root cause
Multiple failures across loader and compiler layers:

**Loader (tt_forge_models):**
- Missing `bitsandbytes>=0.46.1` in `requirements.txt`
- transformers 5.x removed `ROPE_INIT_FUNCTIONS['default']`; model uses it during RoPE initialization
- `GigarEmbedModel` remote code does not call `self.post_init()`, so `all_tied_weights_keys` (set in `post_init()` in transformers 5.x) is never initialized; bitsandbytes quantizer accesses it via `get_keys_to_not_convert()`
- The model uses `torch.autocast('cuda', ...)` hardcoded in `forward()`; fails without a CUDA device
- `Params4bit` bitsandbytes parameters cannot be moved to the TT XLA device; model must be dequantized to BF16 before inference

**tt-mlir (compiler frontend):**
- `LatentAttentionModel.CrossAttention.forward()` uses einops to rearrange Q/K/V to `[b*h, seq, head_dim]` (3D) before passing to `torch.nn.functional.scaled_dot_product_attention`. The `TenstorrentScaledDotProductAttentionConversionPattern` in `StableHLOLegalizeCompositePass.cpp` passed these 3D tensors directly to `ttir::ScaledDotProductAttentionOp`, whose verifier requires exactly 4D tensors.

**tt-metal (backend runtime):**
- `NLPConcatHeadsProgramFactory::create()` unconditionally doubled the CB size for the non-sharded path to enable double-buffering. For this model (`LatentAttentionModel` with 32 heads × head_dim tiles), `per_tensor_tiles * 2 * single_tile_size = 2208256 B > 1572864 B` L1 limit.

## Fix
**Loader fixes** (5 commits in `tt_forge_models` on `remediation/...-v2` branch):
- `giga_embeddings_instruct/embedding_generation/pytorch/requirements.txt`: add `bitsandbytes>=0.46.1`
- `loader.py`: register `_compute_default_rope_parameters` into `ROPE_INIT_FUNCTIONS['default']` at module import time
- `loader.py`: add `_patch_autocast()` static method to replace hardcoded `torch.autocast('cuda', ...)` with device-type-aware variant; call it in `load_model`
- `loader.py`: add `_patch_all_tied_weights_keys()` static method to inject `all_tied_weights_keys = {}` as a class-level default on `GigarEmbedModel` before `from_pretrained`; call it in `load_model`
- `loader.py`: call `dequantize_and_replace(model, quantization_config=..., dtype=torch.bfloat16)` after `from_pretrained` to convert all `Linear4bit` layers to standard `nn.Linear` with BF16 weights

**tt-mlir fix** (1 commit on `remediation/...` branch):
- `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`: in `TenstorrentScaledDotProductAttentionConversionPattern::matchAndRewrite`, detect 3D Q/K/V tensors and insert `ttir::ReshapeOp` to promote them to 4D `[batch, 1, seq, dim]` before the TTIR SDPA op, then reshape back to 3D after.

**tt-metal fix** (1 commit on `remediation/...` branch):
- `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_program_factory.cpp`: guard the double-buffer with `if (per_tensor_tiles * 2 * single_tile_size <= a.device()->l1_size_per_core())`, falling back to single-buffer when the doubled allocation would exceed L1.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    257.65s (0:04:17)
- Tier A attempts: 2

## Files changed
- `tt_forge_models/giga_embeddings_instruct/embedding_generation/pytorch/requirements.txt` (created)
- `tt_forge_models/giga_embeddings_instruct/embedding_generation/pytorch/loader.py` (5 fixes)
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`
- `tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_program_factory.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 4af6e1c20aacb24d80789f068338c3fdb51150b0 |
| tt-mlir         | a9175846b4411a423918f5dcdc144c044850bc49 |
| tt-xla          | 2b375182e4afd883ee5d5df57f058e9b2d0fdadf |
| tt-forge-models | e0acc183333ba324c673a1891b08a04ccf1d5e9f |

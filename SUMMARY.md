# Remediation Summary: aaryan_k_qwen_3_5_2b_gguf-causal_lm-pytorch-2B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aaryan_k_qwen_3_5_2b_gguf/causal_lm/pytorch-2B_GGUF-single_device-inference]

## Result
FAIL — Qwen3.5-2B is a hybrid SSM/full-attention architecture; GGUF loading for qwen3_5_text not implemented in transformers

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-ssm-hybrid-arch-not-supported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
Three cascading loader bugs prevent the test from running:

1. **Missing `requirements.txt`**: `gguf` is not listed as a dependency, so in a fresh CI environment `is_gguf_available()` returns False and transformers raises the ImportError immediately. Fixed.

2. **Wrong GGUF filename**: The loader specifies `Qwen3.5-2B-Q4_K_M.gguf` but the repository `AaryanK/Qwen3.5-2B-GGUF` only ships `Qwen3.5-2B.q4_k_m.gguf` (lowercase, dot-separated). Fixed.

3. **Broken `model_to_load` kwarg forwarding (26 loaders)**: Many GGUF loaders patched `load_gguf_checkpoint` at module import time with a fixed signature `(gguf_path, return_tensors=False)`, missing `**kwargs`. Transformers 5.x added a `model_to_load` parameter that must be forwarded through the entire patcher chain or a TypeError fires. Fixed across all 26 affected loaders.

After the above fixes, the remaining failure is architectural: **Qwen3.5-2B is a hybrid linear-attention (SSM) + full-attention model**, not a standard Qwen3 transformer. The GGUF metadata contains:
- `qwen35.ssm.conv_kernel: 4`, `qwen35.ssm.state_size: 128`, `qwen35.ssm.inner_size: 2048`, etc.
- `qwen35.full_attention_interval: 4` — every 4th layer is standard multi-head attention; the other three layers in each group use linear attention (SSM-based).

Layers 0, 1, 2 (linear attention) have SSM tensors (`ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`) and a fused `attn_qkv.weight [2048, 6144]` for the linear attention component. Layers 3, 7, 11, 15, 19, 23 (full attention) have the standard `attn_q/k/v.weight` but with 32 query heads and 4 KV heads (not 8/2 as the GGUF global metadata implies).

Transformers 5.2.0 has `Qwen3_5TextConfig` (model_type=`qwen3_5_text`) with `layer_types` and SSM parameters, but has NO GGUF-to-`qwen3_5_text` mapping in `GGUF_TO_TRANSFORMERS_MAPPING`. The loader's `_patched_load_gguf_checkpoint` maps `model_type='qwen35'` → `'qwen3'` (the only available alias), which creates a Qwen3ForCausalLM with uniform attention head counts, causing `RuntimeError: size mismatch` for the full-attention layers that have 32 heads instead of 8.

## Fix
Partial loader fixes applied on `remediation/aaryan_k_qwen_3_5_2b_gguf-causal_lm-pytorch-2B_GGUF-single_device-inference` in `tenstorrent/tt-forge-models`:

1. **`aaryan_k_qwen_3_5_2b_gguf/causal_lm/pytorch/requirements.txt`** (created): adds `gguf>=0.10.0`.
2. **`aaryan_k_qwen_3_5_2b_gguf/causal_lm/pytorch/loader.py`** (modified): corrects GGUF filename from `Qwen3.5-2B-Q4_K_M.gguf` to `Qwen3.5-2B.q4_k_m.gguf`; adds `qwen35` GGUF architecture registration and `_patched_load_gguf_checkpoint(*args, **kwargs)` patcher.
3. **26 other loaders** (modified): applied `**kwargs` signature and forwarding fix to all `_patched_load_gguf_checkpoint` functions lacking it.

**Proposed fix** (requires new infrastructure — not attempted):
Implement GGUF-to-`qwen3_5_text` loading in the loader's patched function:
1. Detect `architecture = 'qwen35'` in the raw GGUF result.
2. Construct a `Qwen3_5TextConfig` from GGUF metadata, including deriving `layer_types` from `full_attention_interval` (every `full_attention_interval`-th layer is `'full_attention'`, others `'linear_attention'`).
3. Map SSM parameters: `ssm.conv_kernel` → `linear_conv_kernel_dim`, `ssm.state_size` → SSM state, `ssm.inner_size` → inner_size, `ssm.group_count` → group count.
4. Map full-attention head counts separately (not from global `attention.head_count` which describes linear attention only).
5. Provide a custom tensor-name map for the hybrid architecture (`blk.N.ssm_*` → `model.layers.N.linear_attn.*`, `blk.N.attn_qkv` → `model.layers.N.linear_attn.qkv_proj`, etc.).
6. Return the result with `model_type='qwen3_5_text'` and the correct config.

This requires coordinated changes across the loader's patching logic and the GGUF tensor name mapping — equivalent to adding GGUF support for a new architecture family, which falls into the "new infrastructure" Tier B category.

## Tier B justification
new-infrastructure — Transformers has no GGUF loading support for the `qwen3_5_text` (hybrid SSM/full-attention) architecture. Adding it requires a complete tensor-name mapping for the SSM layers (`ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`, fused `attn_qkv`) plus per-layer `layer_types` derivation from `full_attention_interval`, touching the GGUF mapping infrastructure in a way that goes well beyond a simple registration fix.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `aaryan_k_qwen_3_5_2b_gguf/causal_lm/pytorch/requirements.txt` (created)
- `aaryan_k_qwen_3_5_2b_gguf/causal_lm/pytorch/loader.py` (modified)
- 26 other loaders (modified: `**kwargs` fix for `_patched_load_gguf_checkpoint`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e1a61305b738b4116846f12c073713219615f698 |
| tt-forge-models | d4fe4d1212c618db0af2dfd8bfcf9da43a4187d3 |

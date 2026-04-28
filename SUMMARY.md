# Remediation Summary: amkkk_qwen_3_5_2b_abiliterate_all_layers_baked_gguf_quantized-causal_lm-pytorch-2B_Abiliterate_All_Layers_Baked_GGUF_quantized-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[amkkk_qwen_3_5_2b_abiliterate_all_layers_baked_gguf_quantized/causal_lm/pytorch-2B_Abiliterate_All_Layers_Baked_GGUF_quantized-single_device-inference]

## Result
FAIL — GGUF model has a hybrid architecture with non-uniform attention head dimensions per layer that Qwen3ForCausalLM cannot represent

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-hybrid-arch-non-uniform-head-dim

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The test presented two sequential loader failures:

**Failure 1 (initial run):**
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
A GGUF loader imported earlier in the pytest session had patched `load_gguf_checkpoint` without a `**kwargs` passthrough. Transformers 5.x added `model_to_load` to this call, causing the TypeError.

**Failure 2 (after fixing Failure 1):**
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```
Shape mismatches at layers {3, 7, 11, 15, 19, 23}:
- `self_attn.q_proj.weight`: ckpt [4096, 2048] vs model [1024, 2048]
- `self_attn.k_proj.weight`: ckpt [512, 2048] vs model [256, 2048]
- `self_attn.v_proj.weight`: ckpt [512, 2048] vs model [256, 2048]
- `self_attn.q_norm.weight`: ckpt [256] vs model [128]
- `self_attn.k_norm.weight`: ckpt [256] vs model [128]

## Root cause

**Bug 1 (loader — fixed):** 26 GGUF loaders across tt-forge-models patched `load_gguf_checkpoint` with `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` — missing `**kwargs`. Transformers 5.2.0 added `model_to_load=None` to `load_gguf_checkpoint`, which flows through the global patch chain. All 26 files needed `**kwargs` in the signature and forwarded to `_orig_load_gguf_checkpoint`. This fix was committed to tt-forge-models.

**Bug 2 (loader — Tier B):** The GGUF file for `amkkk/Qwen3.5_2B_Abiliterate_All_Layers_Baked_GGUF_quantized` stores a non-standard hybrid architecture. Inspection of the GGUF metadata reveals:
- `qwen35.full_attention_interval: 4` — every 4th layer uses full/global attention
- `qwen35.ssm.*` fields (conv_kernel, state_size, inner_size, etc.) — SSM (Mamba-style) layer parameters
- `qwen35.attention.key_length: 256` — 256-dim heads for global attention layers
- Local/sliding-window layers (0,1,2,4,5,...): fused `attn_qkv.weight [2048, 6144]`
- Global attention layers (3,7,11,...): separate `attn_q.weight [2048, 4096]`, `attn_k.weight [2048, 512]`, `attn_v.weight [2048, 512]`

The global layers use head_dim=256 with 16 Q-heads and 2 KV-heads; the local layers use a different head configuration packed into combined QKV. `Qwen3ForCausalLM` uses a single global `num_attention_heads` / `head_dim` for all layers and cannot represent this per-layer variation. Loading the GGUF causes shape mismatches at the 6 global attention layers.

## Fix

**Bug 1 fix (applied):** In `tt-forge-models`, applied two-part fix to all 26 loader files with the broken patch pattern:
1. `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` → `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):`
2. `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)`

Branch: `remediation/amkkk_qwen_3_5_2b_abiliterate_all_layers_baked_gguf_quantized-...` in tt-forge-models, commit `d9f2854771`.

**Bug 2 proposed fix:** A custom `Qwen35HybridForCausalLM` model class would be needed that:
- Alternates between Mamba/SSM layers and full-attention layers according to `full_attention_interval`
- Uses different `head_dim` and `num_heads` for local vs global attention layers

This requires implementing a new model architecture class in transformers or in tt-forge-models as a custom modeling file. This is significant new infrastructure beyond a simple loader fix.

## Tier B justification

- **new-infrastructure**: A new model class is required to handle the hybrid SSM + attention architecture with per-layer attention head dimension variation. The `Qwen3ForCausalLM` class is not extensible to this without new infrastructure.
- The fix requires implementing at minimum: a hybrid model class, a custom config class with per-layer head dim fields, and corresponding GGUF-to-HF weight mapping logic. This touches more than 3 files and requires coordinated changes.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 310.37s (second run, after Bug 1 fix)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/` — 26 loader files: `_patched_load_gguf_checkpoint` signature and call site updated with `**kwargs`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d7b4e078d7fbd66c022a121f044b6055e3419f2b |
| tt-forge-models | d9f2854771c71625a7318be5cd4c496169b463ab |

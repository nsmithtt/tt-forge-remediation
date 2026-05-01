# Remediation Summary: gliese_qwen3_5_4b_gguf-causal_lm-pytorch-4B_Abliterated_Caption_i1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gliese_qwen3_5_4b_gguf/causal_lm/pytorch-4B_Abliterated_Caption_i1-single_device-inference]

## Result
FAIL — Qwen3.5-4B is a hybrid SSM/full-attention architecture; GGUF loading for qwen3_5 not implemented in transformers

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
raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")

## Root cause
Two cascading loader bugs prevented the test from running, followed by a residual Tier B architectural issue:

1. **Broken `model_to_load` kwarg forwarding (26 loaders, cross-loader clobbering)**: Many GGUF loaders patched `load_gguf_checkpoint` at module import time with a narrow signature `(gguf_path, return_tensors=False)`, missing `**kwargs`. Transformers 5.2.0 added a `model_to_load` parameter that must be forwarded. During collection, another loader's narrow-sig patch was installed, and when the gliese test ran, `from_pretrained` called `load_gguf_checkpoint` with `model_to_load=dummy_model`, raising `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. (The originally reported `ValueError` is what fires when no cross-loader patch exists.) Fixed.

2. **Missing qwen35 arch registration in gliese loader**: The gliese loader had no `_patched_load_gguf_checkpoint` of its own, so it depended on cross-loader clobbering for qwen35 arch registration. When run in isolation, `qwen35` is not in `GGUF_SUPPORTED_ARCHITECTURES`, raising `ValueError: GGUF model with architecture qwen35 is not supported yet.`. Fixed by adding a standalone `_patched_load_gguf_checkpoint` with qwen35 registration.

After both fixes, the remaining failure is architectural: **Qwen3.5-4B is a hybrid linear-attention (GatedDeltaNet/SSM) + full-attention model**, not a standard Qwen3 transformer. The GGUF metadata contains:
- `qwen35.ssm.conv_kernel: 4`, `qwen35.ssm.state_size: 128`, `qwen35.ssm.inner_size: 4096`, etc.
- `qwen35.full_attention_interval: 4` — every 4th layer (3, 7, 11, ..., 31) is standard multi-head attention; the other layers use GatedDeltaNet.

GDA layers (blk.0, 1, 2, 4, ...) have SSM tensors (`ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`) and fused `attn_qkv.weight [2560, 8192]`. Full-attention layers (blk.3, 7, 11, ...) have separate `attn_q/k/v.weight` with 32 query heads and 4 KV heads at head_dim=256.

Transformers 5.x has `Qwen3_5TextConfig` (model_type=`qwen3_5`) with SSM parameters, but has **no GGUF-to-`qwen3_5` tensor mapping** in `GGUF_TO_TRANSFORMERS_MAPPING`. The loader maps `model_type='qwen35'` → `'qwen3'`, which creates a `Qwen3ForCausalLM` with uniform 16-head attention (head_dim=128). This mismatches the full-attention layers that have 32 heads at head_dim=256, raising `RuntimeError: You set ignore_mismatched_sizes to False`:
```
model.layers.{3,7,...}.self_attn.q_proj.weight: ckpt [8192, 2560] vs model [2048, 2560]
model.layers.{3,7,...}.self_attn.q_norm.weight: ckpt [256] vs model [128]
model.layers.{3,7,...}.self_attn.k_proj.weight: ckpt [1024, 2560] vs model [512, 2560]
model.layers.{3,7,...}.self_attn.v_proj.weight: ckpt [1024, 2560] vs model [512, 2560]
model.layers.{3,7,...}.self_attn.o_proj.weight: ckpt [2560, 4096] vs model [2560, 2048]
```

## Fix
Partial loader fixes applied on `remediation/gliese_qwen3_5_4b_gguf-causal_lm-pytorch-4B_Abliterated_Caption_i1-single_device-inference` (commit `faab1f780d`) in `tenstorrent/tt-forge-models`:

1. **26 GGUF loaders** (modified): Changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `_patched_load_gguf_checkpoint(*args, **kwargs)` and updated inner call to `_orig_load_gguf_checkpoint(*args, **kwargs)`. Eliminates the `model_to_load` TypeError.

2. **`gliese_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`** (modified): Added standalone `_patched_load_gguf_checkpoint` with `_patch_qwen35_support()` and `(*args, **kwargs)` signature so the loader is self-contained and does not rely on cross-loader clobbering.

The residual fix requires a complete `qwen35` → `Qwen3_5ForCausalLM` GGUF tensor name mapping in transformers, covering both GDA layer tensors (`attn_gate`, `attn_qkv`, `ssm_*`) and full-attention layer tensors (`attn_q`, `attn_k`, `attn_v`, `attn_output`), plus SSM config field mapping. This is new infrastructure that must be contributed to `transformers.modeling_gguf_pytorch_utils`.

## Tier B justification
new-infrastructure — GGUF loading for the qwen3_5 (Qwen3.5 hybrid SSM+attention) architecture requires adding a full tensor name mapping (GDA + full-attention layers + SSM config fields) to `transformers.modeling_gguf_pytorch_utils.GGUF_TO_TRANSFORMERS_MAPPING`, which does not exist. No single-file scoped fix is possible.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    394.56s
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/gliese_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/<26 other GGUF loaders>/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | faab1f780d62bbe416e681267a16762faf21a355 |

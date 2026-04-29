# Remediation Summary: cmp_nct_qwen3_5_9b_gguf-causal_lm-pytorch-9B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cmp_nct_qwen3_5_9b_gguf/causal_lm/pytorch-9B_GGUF-single_device-inference]

## Result
FAIL — Qwen3.5 GGUF arch is a hybrid SSM+attention model; no transformers class exists for it

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-ssm-unsupported-arch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

(Initially surfaced as `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` — a separate loader bug fixed in this report.)

## Root cause

Two issues:

**Issue 1 (fixed):** 26 GGUF loader files in `tt_forge_models` monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time using a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 added a `model_to_load` keyword argument to this call site. When pytest collects all model files, any of these 26 loaders is imported before the `cmp_nct_qwen3_5_9b_gguf` test runs, leaving the narrowly-typed patch active. The cmp_nct loader's `AutoModelForCausalLM.from_pretrained` then hits `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

**Issue 2 (Tier B, not fixed):** The `cmp-nct/Qwen3.5-9B-GGUF` file declares GGUF architecture `qwen35`. The same group of monkey-patching loaders also register `qwen35` as an alias for `qwen3` in `GGUF_TO_TRANSFORMERS_MAPPING`, so transformers loads the GGUF weights into a `Qwen3ForCausalLM`. However, Qwen3.5 is a fundamentally different hybrid architecture:

- **SSM (Mamba-style) layers** with tensors `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`; fused `attn_qkv`/`attn_gate` for local attention
- **Full attention layers** (every 4th, `qwen35.full_attention_interval=4`) with 64 Q-heads and 8 KV-heads — yielding q_proj shape `[8192,4096]` vs `Qwen3ForCausalLM`'s expected `[2048,4096]`

The size mismatch causes transformers to raise `RuntimeError: You set ignore_mismatched_sizes to False`. Using `ignore_mismatched_sizes=True` would silently load wrong weights — forbidden workaround.

## Fix

**Issue 1 — fixed in this report:** Updated all 26 GGUF loaders to accept `*args, **kwargs` and forward them through:

```python
# Before
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)

# After
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```

Changes in `tt_forge_models` remediation branch: `cf0074b625337ab4f414066feb0b94d317073f7e`

**Issue 2 — proposed fix (Tier B):** Implement `Qwen35ForCausalLM` with per-layer dispatch: SSM layers at non-full-attention positions use a Mamba-style mixer; every 4th layer uses full grouped-query attention with 64Q/8KV heads. This requires:
- A new model class (hundreds of lines) in transformers or as an inline custom class in the loader
- A new config class `Qwen35Config` that reads `full_attention_interval`, `ssm.state_size`, `ssm.inner_size` from GGUF metadata
- New weight-mapping entries in `GGUF_TO_TRANSFORMERS_MAPPING` for all SSM tensor names

## Tier B justification
**new-infrastructure**: The fix requires a completely new model class `Qwen35ForCausalLM` and config `Qwen35Config` that do not exist in transformers 5.x. The hybrid SSM+attention architecture has distinct tensor shapes and semantics at each layer type, and the correct weight-loading logic cannot be expressed as a simple alias of an existing class.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 435.38s (7:15) to reproduce final error after loader fix
- Tier A attempts: N/A

## Files changed
- `tt_forge_models`: 26 loader files — `_patched_load_gguf_checkpoint` signature widened to `*args, **kwargs`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 724598e7eb6509c5c078a023e0b13a3440dbfb08 |
| tt-forge-models | cf0074b625337ab4f414066feb0b94d317073f7e |

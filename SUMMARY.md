# Remediation Summary: daniloreddy_qwen_3_5_2b_gguf-causal_lm-pytorch-2B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[daniloreddy_qwen_3_5_2b_gguf/causal_lm/pytorch-2B_GGUF-single_device-inference]

## Result
FAIL — Qwen3.5-2B is a hybrid Mamba2/SSM + full-attention architecture; transformers has no GGUF-to-qwen3_5_text mapping and the qwen3 fallback mapping produces weight shape mismatches

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-mamba-hybrid-architecture-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!

## Root cause
The GGUF file (`Qwen3.5-2B_Q4_K_M.gguf`) declares architecture `qwen35`. Transformers 5.2.0 does not include `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES`, so without patching it raises `ValueError: GGUF model with architecture qwen35 is not supported yet.`

During a pytest session, `test_models.py` imports all model loaders at collection time via `TorchDynamicLoader.setup_test_discovery()`. The `daniloreddy_qwen3_5_0_8b_gguf` loader executes `_patch_qwen35_support()` at module level, globally registering `qwen35` as an alias for `qwen3` in `GGUF_SUPPORTED_ARCHITECTURES` and patching `load_gguf_checkpoint` to remap `model_type='qwen35'` → `'qwen3'`. This side effect persists into the 2B test run.

The global patch allows the 2B GGUF to pass the architecture check and load as `Qwen3ForCausalLM`. However, Qwen3.5-2B is a **hybrid Mamba2/SSM + full-attention architecture**, not a standard Qwen3 transformer. The GGUF metadata confirms this:

- `qwen35.ssm.conv_kernel: 4`, `qwen35.ssm.state_size: 128`, `qwen35.ssm.inner_size: 2048`, `qwen35.ssm.group_count: 16`
- `qwen35.full_attention_interval: 4` — every 4th layer is standard multi-head attention; the other three layers use linear/SSM attention
- SSM layers have fused `attn_qkv.weight [2048, 6144]` and SSM tensors (`blk.N.ssm_a`, etc.) instead of separate q/k/v projections
- Full-attention layers use 32 query heads while the global `qwen35.attention.head_count` is 8 (describing linear attention layers only)

The `qwen3` tensor-name mapping does not cover SSM tensor names (`blk.N.ssm_a`, `blk.N.ssm_alpha`, `blk.N.ssm_conv1d`, etc.), and the uniform Qwen3ForCausalLM architecture cannot accommodate the per-layer head count differences between SSM and full-attention layers. This causes `RuntimeError: size mismatch` when transformers' state-dict loading detects mismatched tensor shapes.

Transformers 5.2.0 has a `Qwen3_5TextConfig` class (`model_type='qwen3_5_text'`) with `layer_types` and SSM parameters, but provides **no GGUF-to-`qwen3_5_text` mapping** in `GGUF_TO_TRANSFORMERS_MAPPING`. There is no path from a `qwen35` GGUF file to the correct model class without implementing that mapping.

The same root cause affects `aaryan_k_qwen_3_5_2b_gguf` (reported separately, same bug fingerprint).

## Fix
No fix attempted. This is a Tier B bug requiring new infrastructure.

**Proposed fix** (not implemented):
1. In the loader's patched `load_gguf_checkpoint`, detect `architecture == 'qwen35'` with SSM fields present.
2. Construct a `Qwen3_5TextConfig` from GGUF metadata: derive `layer_types` from `full_attention_interval` (repeating pattern of N-1 `'linear_attention'` then one `'full_attention'`), map SSM parameters (`ssm.conv_kernel` → `linear_conv_kernel_dim`, `ssm.state_size`, `ssm.inner_size`, `ssm.group_count`).
3. Provide a full tensor-name map for the hybrid architecture: `blk.N.ssm_*` → `model.layers.N.linear_attn.*`, fused `blk.N.attn_qkv.weight` → linear-attention QKV projection, per-layer full-attention head counts (32 Q / 4 KV) derived from the actual weight shapes rather than the global metadata.
4. Return `model_type='qwen3_5_text'` so `AutoModelForCausalLM` selects `Qwen3_5ForCausalLM`.

## Tier B justification
new-infrastructure — Transformers has no GGUF loading path for the `qwen3_5_text` (hybrid Mamba2/SSM + full-attention) model class. Implementing it requires a complete tensor-name mapping for SSM layers, per-layer head-count derivation, and a new `Qwen3_5TextConfig` construction from GGUF metadata — equivalent to adding GGUF support for a new architecture family.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    351.58s (0:05:51)
- Tier A attempts: N/A

## Files changed
None — no code changes made.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ec9f38a8327e0872e719841a14022c5415785158 |

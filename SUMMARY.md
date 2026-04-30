# Remediation Summary: khtsly_qwen3_5_27b_abliterated_claude_4_6_opus_distilled_32k_gguf-causal_lm-pytorch-27B_Abliterated_Claude_4_6_Opus_Distilled_32k_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[khtsly_qwen3_5_27b_abliterated_claude_4_6_opus_distilled_32k_gguf/causal_lm/pytorch-27B_Abliterated_Claude_4_6_Opus_Distilled_32k_GGUF-single_device-inference]

## Result
FAIL — HF repository khtsly/Qwen3.5-27B-Abliterated-Claude-4.6-Opus-Distilled-32k-GGUF is 404 (deleted); original error not reproducible; multiple loader bugs present

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AttributeError: 'NoneType' object has no attribute 'config'

Observed in reproduction attempt (repo now 404):
E   OSError: khtsly/Qwen3.5-27B-Abliterated-Claude-4.6-Opus-Distilled-32k-GGUF is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'

## Root cause
The HuggingFace repository `khtsly/Qwen3.5-27B-Abliterated-Claude-4.6-Opus-Distilled-32k-GGUF` has been deleted (confirmed 404 with authenticated HF token). The test cannot be run in any form.

Root-causing the original `AttributeError: 'NoneType' object has no attribute 'config'` is therefore inferential:

1. **Deleted HF repo (primary)**: The model and GGUF file are no longer available. This is confirmed as the current failure mode. The original CI run must have accessed the repo while it still existed.

2. **GGUF arch `qwen35` not in `GGUF_CONFIG_MAPPING`** (loader bug): The Qwen3.5 model family uses GGUF architecture string `qwen35` (confirmed in `gguf/constants.py` line 837: `MODEL_ARCH.QWEN35: "qwen35"`). In transformers 5.2.0, `GGUF_CONFIG_MAPPING` in `transformers/integrations/ggml.py` has entries for `qwen2`, `qwen2_moe`, `qwen3`, and `qwen3_moe`, but NOT `qwen35`. Without this mapping, `AutoConfig.from_pretrained(..., gguf_file=...)` fails with `ValueError: GGUF model with architecture qwen35 is not supported yet.` at `modeling_gguf_pytorch_utils.py:477`. This ValueError in `load_config()` (called in the `finally` block of `_run_model_test_impl`) would replace the original exception from the `try` block, making pytest report the `load_config()` failure rather than the original one.

3. **`load_shard_spec` does not guard for hybrid layers**: `Qwen3_5DecoderLayer.__init__` sets either `self.linear_attn` OR `self.self_attn` depending on `layer_type`, never both. The loader's `load_shard_spec` unconditionally accesses `layer.self_attn.q_proj.weight` for every layer; for "linear_attention" layers where `self_attn` is never set, this raises `AttributeError: 'Qwen3_5DecoderLayer' object has no attribute 'self_attn'`.

The original `AttributeError: 'NoneType' object has no attribute 'config'` was likely produced when an older transformers version handled the unknown `qwen35` arch differently (returning None from some code path rather than raising ValueError), resulting in a None object being dereferenced at `.config`.

## Fix
No fix was applied. The required fixes are:

1. **Find a replacement model**: The `khtsly` repo is gone. A suitable replacement GGUF (e.g., `mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF` or `HeYujie/Qwen3.5-27B-abliterated-GGUF`) would need to be validated and the loader updated.

2. **Register `qwen35` in `GGUF_CONFIG_MAPPING`** in `transformers/integrations/ggml.py` with Qwen3.5-specific field mappings (including `full_attention_interval`, SSM state/inner size fields). This requires understanding the full Qwen3.5 GGUF metadata schema. The `Qwen3_5TextConfig` fields (`layer_types`, `linear_num_value_heads`, `linear_key_head_dim`, etc.) need to be populated from the GGUF metadata.

3. **Fix `load_shard_spec`**: Guard `layer.self_attn` access with `hasattr(layer, "self_attn")` per the pattern documented in memory for other Qwen3.5 hybrid loaders. Add shard specs for `linear_attn` (GatedDeltaNet) layers using `in_proj_qkv`, `in_proj_z`, `out_proj` weights.

4. **Add `use_cache=False`** in `load_inputs` to avoid `Qwen3_5DynamicCache` incompatibility with the test evaluator (same issue as documented for other Qwen3.5 hybrid models).

## Tier B justification
**new-infrastructure**: Registering `qwen35` in `GGUF_CONFIG_MAPPING` requires mapping Qwen3.5-specific GGUF metadata keys (SSM state sizes, `full_attention_interval`, per-layer type information) to `Qwen3_5TextConfig` fields. The Qwen3.5 GGUF spec is not yet mapped in transformers. This is a multi-field infrastructure addition with no existing template in `GGUF_CONFIG_MAPPING`. Additionally, without a live model repo, none of these fixes can be tested.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    29.77s (to first failure: OSError from missing HF repo)
- Tier A attempts: N/A

## Files changed
None.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5a4478ae3efa381b59ad358dc84518e891aacb0f |

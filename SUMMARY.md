# Remediation Summary: ai_sage_gigachat3_10b_a1_8b_gguf-causal_lm-pytorch-10B_A1_8B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ai_sage_gigachat3_10b_a1_8b_gguf/causal_lm/pytorch-10B_A1_8B_GGUF-single_device-inference]

## Result
FAIL — DeepSeek V2 MoE RoPE uses complex tensor arithmetic (`torch.polar` + multiplication); TT compiler has no complex-tensor lowering

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
deepseek-v2-rope-complex-arithmetic-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TT_THROW: Complex tensor with num_dims == 0 is not supported.

While executing %mul : [num_users=26] = call_function[target=torch.ops.aten.mul.Tensor](args = (%polar, 1.0), kwargs = {})

Original traceback:
  File "transformers/models/deepseek_v2/modeling_deepseek_v2.py", line 532, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
  File "transformers/models/deepseek_v2/modeling_deepseek_v2.py", line 229, in forward
    freqs_cis = freqs_cis * self.attention_scaling
```

## Root cause

**Loader layer (fixed):** The GGUF file declares `general.architecture = deepseek2`. Transformers did not include `deepseek2` in `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING["config"]`. Additionally, the existing `glm_4_7_flash_gguf` loader patches `load_gguf_checkpoint` to remap `model_type` from `deepseek2` to `deepseek_v2`, but only registers `deepseek2` (not `deepseek_v2`) in `GGUF_TO_FAST_CONVERTERS`. The tokenizer loader reads the remapped `model_type = "deepseek_v2"` and fails with `KeyError: 'deepseek_v2'` when looking up the tokenizer converter.

**Compiler layer (Tier B):** After all loader bugs are fixed, the model loads and compilation begins. DeepSeek V2 uses a complex-number RoPE: `freqs_cis = torch.polar(torch.ones_like(freqs), freqs)` (line 228 in `modeling_deepseek_v2.py`) followed by `freqs_cis = freqs_cis * self.attention_scaling` (line 229). In the ATen FX graph this is `aten.mul.Tensor(polar_result, 1.0)`. When XLA lowers complex-tensor × real-scalar, the scalar is promoted to a 0-dim complex constant. tt-mlir's `buffer_instance.cc:282` asserts `Complex tensor with num_dims == 0 is not supported.` The underlying requirement is full complex-arithmetic support (split into real/imag or native complex ops) throughout the tt-mlir lowering pipeline — new infrastructure, Tier B.

## Fix

**Loader fix (committed):** `ai_sage_gigachat3_10b_a1_8b_gguf/causal_lm/pytorch/loader.py` in tt-forge-models (`remediation/ai_sage_gigachat3_10b_a1_8b_gguf-causal_lm-pytorch-10B_A1_8B_GGUF-single_device-inference`):

1. Register `deepseek2` in `GGUF_SUPPORTED_ARCHITECTURES` (idempotent guard).
2. Add `deepseek2` config key mapping to `GGUF_TO_TRANSFORMERS_MAPPING["config"]` (same fields as `glm_4_7_flash_gguf` but with corrected head dimension mappings for GigaChat3 MLA; intentionally omits `attention.head_count_kv` because GigaChat3 stores compressed-KV count=1 but HF needs `num_key_value_heads == num_attention_heads`).
3. Register **both** `deepseek2` and `deepseek_v2` in `GGUF_TO_FAST_CONVERTERS` → `GGUFQwen2Converter`.
4. Patch `load_gguf_checkpoint` to rewrite `model_type` from `deepseek2` → `deepseek_v2` and set `q_lora_rank=None` when absent.
5. Patch `get_gguf_hf_weights_map` to translate `model_type=deepseek_v2` back to `deepseek2` for gguf-py's `MODEL_ARCH_NAMES` lookup.
6. Also patch `tok_auto`, `config_utils`, and `modeling_utils` module-level bindings so `AutoConfig.from_pretrained` and `AutoModelForCausalLM.from_pretrained` pick up the rewrite.

**Compiler fix (proposed):** In tt-mlir, add a lowering pass to decompose complex tensors into real/imaginary pairs before type lowering, or add a StableHLO → TTIR lowering pattern for `stablehlo.complex_multiply` / `stablehlo.real` / `stablehlo.imag`. Would touch at minimum: type lowering, arithmetic op patterns, and the `buffer_instance.cc` guard.

## Tier B justification

**Indicator:** new-infrastructure

Implementing complex tensor arithmetic in the TT compiler requires (at minimum): (1) a lowering pass to split complex-typed ops into real/imag representations or introduce TTIR complex ops; (2) matching TTNN kernel support for complex arithmetic; (3) the `buffer_instance.cc` 0-dim complex buffer allocator. Changes span ≥3 files across tt-mlir and tt-metal.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    707.66s (0:11:47) — loader fixed, model loaded, compilation failed
- Tier A attempts: N/A

## Files changed
- `ai_sage_gigachat3_10b_a1_8b_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 814bc789d74b87056b0cb18d00386f3010a5e3cf |
| tt-forge-models | 1c5391d2dccd89723ab2d9d86ed7da3b9399ad33 |

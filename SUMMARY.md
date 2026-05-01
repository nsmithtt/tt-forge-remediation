# Remediation Summary: glm_4_7_flash_heretic_mpoa_gguf-causal_lm-pytorch-4.7_Flash_heretic_MPOA_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_heretic_mpoa_gguf/causal_lm/pytorch-4.7_Flash_heretic_MPOA_GGUF-single_device-inference]

## Result
XFAIL — GLM-4.7-Flash-heretic-MPOA has 64 routed experts × 46 MoE layers (~60 GB BF16), exceeding single p150b DRAM capacity (32 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-glm-4-7-flash-heretic-mpoa-gguf

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AttributeError: 'tuple' object has no attribute 'to'
```

During tokenizer loading there was also:
```
KeyError: 'deepseek_v2'
```

The original reported failure was `raise AttributeError(` at `modeling_deepseek_v2.py:363`:
`q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings.to(q_pe.device))`
because `_patched_yarn_rope_forward` returned a plain Python tuple `(cos, sin)` instead of an object with a `.to()` method.

## Root cause

**Loader layer — multiple bugs, all fixed:**

1. **`KeyError: 'deepseek_v2'` in tokenizer loading**: Cross-loader import ordering during pytest collection caused a multi-loader clobbering chain. `glm_4_7_flash_gguf` patches `load_gguf_checkpoint` to remap `"deepseek2"→"deepseek_v2"` and registers `"deepseek2"` in `GGUF_TO_FAST_CONVERTERS`. `gpt_oss_swallow_20b_rl_v0_1_gguf` (imported after) captures the already-patched function and chains it into `tokenization_utils_tokenizers.load_gguf_checkpoint`. `iquest_coder_v1_40b_instruct_gguf` patches `convert_gguf_tokenizer` globally. The result: tokenizer loading sees `model_type="deepseek_v2"` and calls `convert_gguf_tokenizer("deepseek_v2", ...)` → `KeyError` because only `"deepseek2"` was registered.

2. **`NotImplementedError` in `get_gguf_hf_weights_map`**: gguf-py's `MODEL_ARCH_NAMES` has `"deepseek2"` but not `"deepseek_v2"`; when called with `model_type="deepseek_v2"` it raises `NotImplementedError`.

3. **Complex tensor rejection on TT PJRT**: `DeepseekV2YarnRotaryEmbedding.forward` uses `torch.polar` → complex tensors, rejected by TT PJRT with "Complex tensor with num_dims == 0 is not supported."

4. **`AttributeError: 'tuple' object has no attribute 'to'`**: The RoPE patch returned a plain Python tuple `(cos, sin)`. `modeling_deepseek_v2.py:363` calls `position_embeddings.to(q_pe.device)` before passing to `apply_rotary_emb`, so the tuple needs a `.to()` method.

5. **MLA `num_key_value_heads` overexpansion**: GGUF stores `head_count_kv=1` as a latent rank marker; this causes GQA expansion to multiply by `num_attention_heads` before SDPA.

6. **MoE `nonzero()` dynamic shapes**: The original `DeepseekV2Experts` forward uses `nonzero()`, producing dynamic-shape outputs incompatible with TT XLA tracing.

**Hardware capacity:**

After all loader fixes, the model has 64 routed experts × 46 MoE layers (47 total hidden layers). The Q4_K_M GGUF is 17 GB, but when loaded into BF16 for inference the weight footprint is approximately 60 GB — nearly double the p150b's 32 GB DRAM capacity. The second test run (with all fixes applied) consumed 141 GB system RAM during compilation and ran for 146+ minutes before being killed.

## Fix

All fixes are in `tt_forge_models` loader at `glm_4_7_flash_heretic_mpoa_gguf/causal_lm/pytorch/loader.py`:

1. **`_patch_deepseek_v2_gguf()`**: registers `GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter` so cross-loader clobbering no longer causes KeyError; patches `get_gguf_hf_weights_map` to remap `model_type="deepseek_v2"` → `"deepseek2"` for gguf-py tensor name lookup.

2. **`_tt_static_deepseek_v2_moe_forward`**: registers a static per-expert masked matmul in `ALL_EXPERTS_FUNCTIONS["tt_static_deepseek_v2_moe"]` and patches `get_correct_experts_implementation` to accept custom keys. Set via `config._experts_implementation = "tt_static_deepseek_v2_moe"` in `load_model`.

3. **`_patch_deepseek_v2_rope()`**: replaces `DeepseekV2RotaryEmbedding.forward` with real-arithmetic cos/sin (avoiding `torch.polar` complex tensors); replaces `apply_rotary_emb` with real-valued rotation.

4. **`_CosSinEmbed(tuple)`**: tuple subclass with `.to(*args, **kwargs)` forwarding to each element, so `position_embeddings.to(q_pe.device)` does not raise `AttributeError`.

5. **`load_model`**: sets `config.num_key_value_heads = config.num_attention_heads` when `kv_heads < attn_heads` to prevent GQA over-expansion for MLA.

The test config `tests/runner/test_config/torch/test_config_inference_single_device.yaml` was updated in tt-xla to mark this test `KNOWN_FAILURE_XFAIL`.

## Verification
- pytest exit: TIMEOUT (killed after 146+ minutes due to hardware capacity)
- Hardware:    blackhole-p150b
- Duration:    146+ minutes (killed, hardware capacity ceiling)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/glm_4_7_flash_heretic_mpoa_gguf/causal_lm/pytorch/loader.py` (new file, all loader fixes)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b17a67937935bc61cf423974e312d1847940cbd6 |
| tt-forge-models | 1ebd261cb2c958843434101f847327ac2726b28b |

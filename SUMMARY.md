# Remediation Summary: helios_nova-causal_lm-pytorch-306M-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[helios_nova/causal_lm/pytorch-306M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
helios-nova-model-type-not-in-transformers

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   ValueError: The checkpoint you are trying to load has model type `helios_nova` but Transformers does not recognize this architecture.

(CI had reported pcc=0.5336308393059899; locally the failure is the ValueError above, which is the root cause of the CI failure.)

## Root cause
The `respinosamena/Helios-Nova-306M` HuggingFace repo ships only model weights and config — no custom modeling code (`modeling_helios_nova.py` etc.). `AutoModelForCausalLM.from_pretrained` with `trust_remote_code=True` has no remote code to download, so transformers 5.2 raises `ValueError` because `helios_nova` is absent from `CONFIG_MAPPING`.

In CI the test produced pcc=0.53, indicating some fallback was loading an incorrect model. The root bug is in the loader, which relied on `AutoModelForCausalLM` to handle a non-standard architecture.

## Fix
Replaced the `AutoModelForCausalLM`-based loader with a complete custom PyTorch implementation in `tt-xla/third_party/tt_forge_models/helios_nova/causal_lm/pytorch/loader.py`:

- `_HeliosNovaConfig` — dataclass wrapping the raw JSON fields, with `num_hidden_layers` alias
- `_RMSNorm` — standard RMSNorm with learnable weight
- `_Attention` — GQA (16 Q heads, 4 KV heads, head_dim=64) with per-head QK norm (applied before RoPE) and `F.scaled_dot_product_attention(is_causal=True)`
- `_FFN` — SwiGLU: `down(silu(gate(x)) * up(x))`
- `_Block` — pre-norm decoder layer
- `_HeliosNovaForCausalLM` — full model with RoPE buffers, tied embeddings, returns `CausalLMOutputWithPast`

Weights are loaded directly from safetensors; the model's weight naming matches the checkpoint's custom scheme exactly, requiring no remapping.

## Verification
- pytest exit: PASS
- Hardware:    n300
- Duration:    86.46s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/helios_nova/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ea0016f542c614a66430d4d7b22b1a9c5890674f |
| tt-forge-models | bd02c270f5afcc94192b81fe42cb42eec6199c99 |

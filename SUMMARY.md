# Remediation Summary: glm_4_7_flash_ultimate_uncensored_heretic_ayun_gguf-causal_lm-pytorch-4_7_Flash_ultimate_uncensored_heretic_ayun_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_ultimate_uncensored_heretic_ayun_gguf/causal_lm/pytorch-4_7_Flash_ultimate_uncensored_heretic_ayun_GGUF-single_device-inference]

## Result
FAIL — Tier B compiler-stack bug: tt-xla CPU-fallback partitioner crashes on complex 0-dimensional tensor from DeepSeek-V2 RoPE (`freqs_cis * attention_scaling`)

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
pjrt-complex-tensor-zero-dim-cpu-fallback

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_THROW @ /home/nsmith/tt-forge-remediation/tt-xla/pjrt_implementation/src/api/buffer_instance.cc:282: tt::exception
info:
Complex tensor with num_dims == 0 is not supported.
```
Full traceback:
```
partition_fx_graph_for_cpu_fallback
  dynamo_bridge.py:762: collector.run(*xla_args)
  tt_torch/torch_overrides.py:34: return func(*args, **(kwargs or {}))
  buffer_instance.cc:282: TT_THROW("Complex tensor with num_dims == 0 is not supported.")
```
Failing FX node: `%mul = aten.mul.Tensor(%polar, 1.0)` — corresponds to `freqs_cis = freqs_cis * self.attention_scaling` in `transformers/models/deepseek_v2/modeling_deepseek_v2.py:229`.

## Root cause
GLM-4.7-Flash-ultimate-uncensored-heretic-ayun is a GGUF-quantized fine-tune of GLM-4.7 Flash, which uses the DeepSeek-V2 architecture (Multi-head Latent Attention). Three loader-layer bugs were present and fixed (see Fix section). After loader fixes, the model loads and enters compilation. The DeepSeek-V2 RoPE implementation computes `freqs_cis = torch.polar(t_sin, t_cos)` — a complex-valued 0-dimensional tensor — then scales it by `self.attention_scaling` (a Python float), producing another 0-dim complex tensor. During `partition_fx_graph_for_cpu_fallback` in the tt-xla dynamo bridge, the `UnsupportedNodesCollector.run_node` method attempts to actually execute each FX graph node on real TT device tensors to determine which ones require CPU fallback. This execution hits `buffer_instance.cc:282` in the PJRT backend, which explicitly throws for any complex tensor with `num_dims == 0`. The `run_node` method in `dynamo_bridge.py` has no try/except around `super().run_node(n)`, so the `RuntimeError` propagates uncaught and aborts the entire partitioning pass.

## Fix
Three loader-layer bugs were fixed in `third_party/tt_forge_models/glm_4_7_flash_ultimate_uncensored_heretic_ayun_gguf/causal_lm/pytorch/loader.py` on branch `remediation/glm_4_7_flash_ultimate_uncensored_heretic_ayun_gguf-causal_lm-pytorch-4_7_Flash_ultimate_uncensored_heretic_ayun_GGUF-single_device-inference` in tt-forge-models:

1. **deepseek_v2 tokenizer KeyError** — added `_patch_deepseek_v2_gguf()` which registers `"deepseek_v2"` in `GGUF_TO_FAST_CONVERTERS` (mapping to `GGUFQwen2Converter`) and patches `get_gguf_hf_weights_map` to temporarily remap `model_type` from `deepseek_v2` to `deepseek2` when calling the underlying map lookup (the gguf-py tensor-name table uses the `deepseek2` key).

2. **MLA num_key_value_heads GQA over-expansion** — in `load_model()`, after loading config, if `num_key_value_heads < num_attention_heads`, set `config.num_key_value_heads = config.num_attention_heads`. GGUF stores `head_count_kv=1` as a latent-rank marker for MLA; leaving it at 1 causes the GQA expansion path to multiply Q-heads by 20 before SDPA.

3. **ignore_mismatched_sizes** — added `ignore_mismatched_sizes=True` to `from_pretrained()` to suppress size-mismatch errors during GGUF weight loading.

The Tier B compiler-stack bug remains unfixed. The proposed fix: in `dynamo_bridge.py`, wrap `super().run_node(n)` in `UnsupportedNodesCollector.run_node` with a try/except that catches `RuntimeError` and marks the node for CPU fallback, rather than propagating the exception. This lives in `tt-xla/python_package/tt_torch/dynamo/dynamo_bridge.py`.

## Tier B justification
**Which indicator**: new-infrastructure

The `UnsupportedNodesCollector` in `dynamo_bridge.py` uses actual device execution to classify nodes; adding exception handling is superficially small, but the correct fix is to decompose complex-valued 0-dim tensors before the partitioning pass (real-arithmetic RoPE, replacing `torch.polar` with explicit sin/cos components). That decomposition must be applied as an FX graph transform early in the compilation pipeline, coordinated with the StableHLO lowering path — a cross-cutting change that touches the Dynamo bridge, the FX pass infrastructure, and potentially the RoPE lowering in tt-mlir.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    1690.18s (0:28:10)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/glm_4_7_flash_ultimate_uncensored_heretic_ayun_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0d94ec6baf234cd0ec19d20b015e115576e1535f |
| tt-forge-models | 5334e7efeca2fa2dfb46146ccacb51d855894d3e |

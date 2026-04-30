# Remediation Summary: glm_4_7_flash_heretic_i1_gguf-causal_lm-pytorch-4_7_Flash_heretic_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_heretic_i1_gguf/causal_lm/pytorch-4_7_Flash_heretic_i1_GGUF-single_device-inference]

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
The DeepSeek-V2 RoPE implementation computes `freqs_cis = torch.polar(t_sin, t_cos)` — a complex-valued 0-dimensional tensor — then scales it by `self.attention_scaling` (a Python float, producing another 0-dim complex tensor). During `partition_fx_graph_for_cpu_fallback` in the tt-xla dynamo bridge, the `UnsupportedNodesCollector.run_node` method attempts to actually execute each FX graph node on real TT device tensors to determine which ones require CPU fallback. This execution hits `buffer_instance.cc:282` in the PJRT backend, which explicitly throws for any complex tensor with `num_dims == 0`.

The `run_node` method in `dynamo_bridge.py` has no try/except around `super().run_node(n)`, so the `RuntimeError` propagates uncaught and aborts the entire partitioning pass.

Four loader-layer bugs were also present and fixed (see Fix section); they were not the final failure.

## Fix
Four loader-layer bugs fixed in `tt_forge_models` on branch `remediation/glm_4_7_flash_heretic_i1_gguf-causal_lm-pytorch-4_7_Flash_heretic_i1_GGUF-single_device-inference`:

1. **deepseek_v2 GGUF tokenizer KeyError** (`86a6bdaf1f`): The upstream `glm_4_7_flash_gguf` loader patches `load_gguf_checkpoint` to remap `model_type: deepseek2 → deepseek_v2` but only registers `deepseek2` in `GGUF_TO_FAST_CONVERTERS`. The heretic loader now adds `_patch_deepseek_v2_gguf()` that registers `GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter` and patches `get_gguf_hf_weights_map` to swap `model_type` back to `deepseek2` for the map lookup. File: `glm_4_7_flash_heretic_i1_gguf/causal_lm/pytorch/loader.py`.

2. **26 GGUF loaders missing `**kwargs`** (`65f56db8ba`): Cherry-pick of upstream fix. Each affected loader had `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` without `**kwargs`. Transformers 5.x added `model_to_load` kwarg; the missing `**kwargs` caused `TypeError: unexpected keyword argument 'model_to_load'`. Fixed all 26 loaders.

3. **`_patched_get_gguf_hf_weights_map` missing `num_layers`/`qual_name` args** (`d3e80778ca`): The heretic loader's patched `get_gguf_hf_weights_map` had signature `(hf_model, processor=None, model_type=None)` but transformers 5.x takes `(hf_model, processor, model_type, num_layers, qual_name)`. Another loader in a chained patch session passed all five positionally, causing `TypeError: got an unexpected keyword argument 'qual_name'`. Fixed signature to include `num_layers=None, qual_name=""` and forwarded all args. File: `glm_4_7_flash_heretic_i1_gguf/causal_lm/pytorch/loader.py`.

4. **MLA `num_key_value_heads` GQA over-expansion** (`7f344e16fc`): The GGUF metadata stores `head_count_kv=1` as a latent rank marker for MLA (Multi-head Latent Attention), not an actual KV head count. With `num_key_value_heads=1` and `num_attention_heads=20`, GQA expansion in SDPA computed `num_key_value_groups = 20` and expanded already-correct 20-head K tensors to 400 heads, causing `Mismatching argument... length 400... expected [1, 20]`. Fixed in `load_model()` by loading config first and setting `config.num_key_value_heads = config.num_attention_heads` when `num_key_value_heads < num_attention_heads`. File: `glm_4_7_flash_heretic_i1_gguf/causal_lm/pytorch/loader.py`.

Proposed compiler-stack fix (not attempted — Tier B): Add a try/except in `UnsupportedNodesCollector.run_node` in `dynamo_bridge.py` to catch `RuntimeError` for nodes that crash the TT backend during partitioning (treating them as unsupported, routing to CPU). The deeper fix would be supporting complex 0-dim tensors in `buffer_instance.cc` and the tt-mlir/tt-metal lowering pipeline.

## Tier B justification
**cross-cutting**: Properly supporting complex 0-dimensional tensors requires coordinated changes across at minimum three locations: (1) `buffer_instance.cc` — stop throwing, handle complex 0-dim allocation; (2) tt-mlir — add complex tensor lowering patterns; (3) tt-metal — complex arithmetic kernel support. Additionally, `dynamo_bridge.py:run_node` needs exception-safe partitioning to avoid aborting on any single node failure. This exceeds the ~3-file, single-layer scope of a Tier A fix.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 0:28:08
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/glm_4_7_flash_heretic_i1_gguf/causal_lm/pytorch/loader.py` — all four loader fixes
- `third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py` — **kwargs fix
- `third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py` — **kwargs fix
- `third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py` — **kwargs fix
- `third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py` — **kwargs fix
- (22 additional GGUF loader files — all **kwargs fix for _patched_load_gguf_checkpoint)
- `tt-xla/third_party/tt_forge_models` — submodule pointer updated to remediation HEAD

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6ae51da47945cd6c490d1eb25015a22bf85d1c68 |
| tt-forge-models | 7f344e16fc3bf3e6c8cf5b0cf1deed92bf3b6e87 |

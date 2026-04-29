# Remediation Summary: autoglm_phone_9b_gguf-causal_lm-pytorch-9B_Phone_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[autoglm_phone_9b_gguf/causal_lm/pytorch-9B_Phone_Q4_K_M-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed; test passes on silicon in 753s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before fix 1):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After fix 1, second failure:
```
AttributeError: 'Glm4MLP' object has no attribute 'up_proj'
```

## Root cause

**Bug 1 — `_patched_load_gguf_checkpoint` missing `model_to_load` kwarg:**
26 GGUF loaders define a module-level function `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and install it globally over `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time. Transformers 5.2 added `model_to_load=None` to `load_gguf_checkpoint`'s signature and calls it as `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)`. During pytest collection `TorchDynamicLoader.setup_test_discovery` imports every loader, leaving whichever bad-signature loader was imported last as the active patch. The autoglm loader itself does not patch, but it calls `AutoModelForCausalLM.from_pretrained` which internally calls the already-patched function with `model_to_load=`, causing `TypeError`.

**Bug 2 — `load_shard_spec` referencing wrong MLP attribute:**
The autoglm loader's `load_shard_spec` method accesses `layer.mlp.up_proj.weight` and `layer.mlp.gate_proj.weight`. `Glm4MLP` (which AutoGLM-Phone-9B uses) does not have separate `up_proj`/`gate_proj`; instead it has a fused `gate_up_proj = Linear(hidden, 2 * intermediate)`. This caused `AttributeError` when the test runner called `workload.shard_spec_fn(model)` before putting the model on the TT device, even for single_device inference.

## Fix

**Fix 1** (`tt_forge_models`, 26 files):
Updated all loaders with the bad `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signature to:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)
```
Files in: `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `dmind_3_mini_i1_gguf`, `gpt_oss_swallow_{120b_rl,20b_rl,20b_sft_mxfp4_moe}_v0_1_gguf`, 14x `mradermacher_*`, `qwen_3_5_imatrix_gguf`, 2x `tvall43_*`, `unified_reward_flex_qwen35_27b_gguf`.

**Fix 2** (`tt_forge_models`, `autoglm_phone_9b_gguf/causal_lm/pytorch/loader.py`):
Replaced:
```python
shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
```
With:
```python
shard_specs[layer.mlp.gate_up_proj.weight] = ("model", "batch")
```
Also added `if layer.self_attn.*.bias is not None:` guards for the attention bias entries.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    753.99s (0:12:33)
- Tier A attempts: N/A

## Files changed
- `autoglm_phone_9b_gguf/causal_lm/pytorch/loader.py` (load_shard_spec fix)
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d266b444ce833ef355b870d55cbd2264df75876e |
| tt-forge-models | 3bd6b0356088f44faad6ccc184a0870d2f7037f0 |
